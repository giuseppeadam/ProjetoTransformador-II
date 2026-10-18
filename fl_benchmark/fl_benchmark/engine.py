"""Treino local e avaliação, compartilhados por todas as estratégias.

Concentrar o laço de treino aqui é o que garante que a comparação entre
estratégias seja justa: todas usam o mesmo otimizador, a mesma função de perda
e o mesmo número de épocas locais. A única variação permitida é o termo
proximal, exigido pelo FedProx e inerte (`proximal_mu = 0`) nas demais.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """Monta um `DataLoader` sobre arrays tabulares.

    Quando `shuffle` está ativo, `seed` é obrigatório para que a execução seja
    reproduzível. Sem gerador explícito, o `DataLoader` consome o RNG global do
    processo — e numa simulação cada cliente roda em um processo próprio que não
    herda a semente definida no servidor, tornando a ordem dos batches
    diferente a cada execução e as métricas irreprodutíveis.
    """
    gerador = None
    if shuffle:
        if seed is None:
            raise ValueError("make_loader com shuffle=True exige seed, para reprodutibilidade")
        gerador = torch.Generator()
        gerador.manual_seed(int(seed))

    return DataLoader(
        TabularDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gerador,
    )


def derive_seed(*componentes: int) -> int:
    """Combina identificadores em uma semente estável.

    Permite que cada par (cliente, round) tenha sua própria ordem de batches —
    preservando a estocasticidade do treino — sem perder a reprodutibilidade,
    já que a semente é função apenas da configuração e não do relógio.
    """
    valor = 0
    for componente in componentes:
        valor = (valor * 1_000_003 + int(componente)) % (2**31 - 1)
    return valor


def resolve_device(preferencia: str = "auto") -> torch.device:
    """Resolve o dispositivo de treino.

    `auto` usa a GPU quando disponível. O benchmark registra em qual
    dispositivo cada execução rodou, já que isso afeta diretamente a métrica
    de tempo.
    """
    if preferencia == "cpu":
        return torch.device("cpu")
    if preferencia == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' solicitado, mas nenhuma GPU CUDA está disponível")
        return torch.device("cuda")
    if preferencia != "auto":
        raise ValueError(f"device desconhecido: {preferencia!r}. Use 'auto', 'cuda' ou 'cpu'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def train(
    net: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    proximal_mu: float = 0.0,
) -> tuple[float, float]:
    """Treina `net` localmente e devolve `(loss_média, tempo_em_segundos)`.

    Quando `proximal_mu > 0`, soma à perda o termo proximal do FedProx,
    `(mu/2) * ||w - w_global||^2`, que penaliza o afastamento dos pesos locais
    em relação aos pesos globais recebidos no início do round. A cópia dos
    pesos globais é feita depois de carregar o modelo recebido e antes do
    treino, como exige a formulação do método.

    O tempo medido cobre apenas o laço de otimização — carga de dados e
    construção do modelo ficam de fora, para que a métrica reflita o custo do
    aprendizado em si.
    """
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    pesos_globais = None
    if proximal_mu > 0:
        pesos_globais = [p.detach().clone() for p in net.parameters()]

    net.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inicio = time.perf_counter()

    soma_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)

            if pesos_globais is not None:
                proximal = torch.zeros((), device=device)
                for local, global_ in zip(net.parameters(), pesos_globais):
                    proximal = proximal + (local - global_).pow(2).sum()
                loss = loss + (proximal_mu / 2.0) * proximal

            loss.backward()
            optimizer.step()

            soma_loss += float(loss.item())
            num_batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duracao = time.perf_counter() - inicio

    media = soma_loss / num_batches if num_batches else float("nan")
    return media, duracao


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Avalia `net` e devolve perda, acurácia, F1 e ROC-AUC.

    O F1 acompanha a acurácia porque a partição non-IID desbalanceia as classes
    dentro de cada cliente, e a acurácia sozinha esconde o caso em que o modelo
    acerta a classe majoritária e erra a minoritária.

    O ROC-AUC usa a probabilidade, não a decisão em 0,5: continua discriminando
    modelos mesmo quando a acurácia satura. Fica indefinido quando o conjunto
    avaliado contém uma única classe, situação real na partição non-IID — nesse
    caso devolve `nan`, que a agregação ignora em vez de contaminar a média.
    """
    net.to(device)
    net.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    soma_loss, total = 0.0, 0
    probabilidades: list[np.ndarray] = []
    verdadeiros: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = net(X_batch)
        soma_loss += float(criterion(logits, y_batch).item())
        total += y_batch.numel()
        probabilidades.append(torch.sigmoid(logits).cpu().numpy().ravel())
        verdadeiros.append(y_batch.cpu().numpy().ravel())

    y_prob = np.concatenate(probabilidades)
    y_true = np.concatenate(verdadeiros)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "loss": soma_loss / total,
        "accuracy": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "num_examples": int(total),
    }
