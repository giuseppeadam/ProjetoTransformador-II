"""Agregação de métricas e indicadores derivados do benchmark.

O Flower agrega as métricas dos clientes com média ponderada pelo número de
exemplos. Isso é correto para perda e acurácia, mas errado para duas coisas que
este benchmark precisa medir:

* **tempo de treino** — o custo computacional de um round é a *soma* do tempo
  gasto pelos clientes, não a média;
* **métricas indefinidas** — na partição non-IID um cliente pode ter uma única
  classe na validação, o que torna o ROC-AUC indefinido. Uma média cega
  propagaria `nan` para o resultado do round inteiro.

Por isso a agregação é feita aqui, explicitamente.
"""

import math

import numpy as np
from flwr.app import MetricRecord, RecordDict

# Chaves tratadas por soma, não por média ponderada.
CHAVES_SOMADAS = ("train_time_s",)

# Acurácia a partir da qual o modelo global é considerado convergido, usada
# para medir eficiência de comunicação.
ALVO_CONVERGENCIA = 0.95

# Alvos reportados nos relatórios. O alvo frouxo serve de verificação — todas as
# estratégias devem alcançá-lo de imediato neste dataset, e uma que não alcance
# tem problema de implementação. O alvo exigido fica logo abaixo do teto
# centralizado e é o que de fato separa as estratégias.
ALVOS_RELATORIO = (0.95, 0.99)


def _metricas(record: RecordDict) -> dict:
    return dict(next(iter(record.metric_records.values())))


def _media_ponderada(valores: list[float], pesos: list[float]) -> float:
    """Média ponderada que descarta pares cujo valor é indefinido."""
    validos = [(v, p) for v, p in zip(valores, pesos) if not math.isnan(v)]
    if not validos:
        return float("nan")
    total = sum(p for _, p in validos)
    if total == 0:
        return float("nan")
    return sum(v * p for v, p in validos) / total


def aggregate(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Agrega as respostas dos clientes de um round.

    Métricas de qualidade saem como média ponderada pelo número de exemplos;
    tempos saem somados; e a dispersão da acurácia entre clientes é adicionada
    como `accuracy_std`, que mede equidade — se o modelo global serve bem todas
    as instituições ou apenas as que detêm mais dados.
    """
    por_cliente = [_metricas(r) for r in records]
    pesos = [float(m[weighting_metric_name]) for m in por_cliente]

    chaves = {k for m in por_cliente for k in m if k != weighting_metric_name}
    agregado = MetricRecord()

    for chave in sorted(chaves):
        valores = [float(m[chave]) for m in por_cliente if chave in m]
        if chave in CHAVES_SOMADAS:
            agregado[chave] = float(sum(valores))
        else:
            relevantes = [p for m, p in zip(por_cliente, pesos) if chave in m]
            agregado[chave] = _media_ponderada(valores, relevantes)

    if "accuracy" in chaves:
        acuracias = [float(m["accuracy"]) for m in por_cliente if "accuracy" in m]
        agregado["accuracy_std"] = float(np.std(acuracias)) if len(acuracias) > 1 else 0.0
        agregado["accuracy_min"] = float(np.min(acuracias))

    agregado["num_clients"] = float(len(por_cliente))
    return agregado


def rounds_to_target(curva: dict[int, float], alvo: float = ALVO_CONVERGENCIA) -> int | None:
    """Primeiro round em que a acurácia centralizada alcança `alvo`.

    Mede eficiência de comunicação: em Federated Learning real cada round é uma
    rodada de troca de pesos com todas as instituições, e é esse o recurso
    caro. Duas estratégias que terminam na mesma acurácia não são equivalentes
    se uma precisou de três rounds e a outra de nove. Devolve `None` quando o
    alvo não foi atingido na execução.
    """
    for rodada in sorted(curva):
        valor = curva[rodada]
        if not math.isnan(valor) and valor >= alvo:
            return rodada
    return None


def curva_central(run: dict) -> dict[int, float]:
    """Extrai a curva de acurácia centralizada por round de uma execução."""
    return {int(linha["round"]): float(linha["central_accuracy"]) for linha in run["rounds"]}


def rounds_por_alvo(run: dict, alvos=ALVOS_RELATORIO) -> dict[float, int | None]:
    """Rounds necessários para atingir cada alvo, calculados da curva salva.

    Derivar isto da curva, e não de um único valor registrado na execução,
    permite reavaliar a eficiência com outros alvos sem repetir o benchmark.
    """
    curva = curva_central(run)
    return {alvo: rounds_to_target(curva, alvo) for alvo in alvos}


def _media_desvio(valores: list[float]) -> tuple[float, float]:
    limpos = [v for v in valores if not math.isnan(v)]
    if not limpos:
        return float("nan"), float("nan")
    media = float(np.mean(limpos))
    desvio = float(np.std(limpos)) if len(limpos) > 1 else 0.0
    return media, desvio


def resumo_repeticoes(runs: list[dict], alvos=ALVOS_RELATORIO) -> dict:
    """Agrega as repetições de uma mesma configuração em média e dispersão.

    A agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de
    chegada das respostas dos clientes varia entre execuções, e diferenças da
    ordem de 1e-7 nos pesos se amplificam ao longo dos rounds. Reportar uma
    execução única confundiria essa variação com diferença entre métodos; o que
    distingue as estratégias é a média entre sementes, lida junto da dispersão.
    """
    if not runs:
        raise ValueError("nenhuma execução para resumir")

    campos_finais = (
        "central_accuracy",
        "central_f1",
        "central_auc",
        "central_loss",
        "federated_accuracy_std",
        "federated_accuracy_min",
    )
    resumo: dict = {"num_seeds": len(runs), "seeds": [r["config"]["seed"] for r in runs]}

    for campo in campos_finais:
        media, desvio = _media_desvio([float(r["final"][campo]) for r in runs])
        resumo[campo] = media
        resumo[f"{campo}_spread"] = desvio

    for alvo in alvos:
        rodadas = [rounds_por_alvo(r, (alvo,))[alvo] for r in runs]
        atingidos = [r for r in rodadas if r is not None]
        chave = f"rounds_to_{int(alvo * 100)}"
        if len(atingidos) < len(rodadas):
            # Uma repetição que não alcança o alvo não pode entrar na média como
            # se fosse um número: o resultado honesto é registrar em quantas
            # sementes o alvo foi atingido.
            resumo[chave] = float(np.mean(atingidos)) if atingidos else None
            resumo[f"{chave}_reached"] = f"{len(atingidos)}/{len(rodadas)}"
        else:
            media, desvio = _media_desvio([float(r) for r in atingidos])
            resumo[chave] = media
            resumo[f"{chave}_spread"] = desvio
            resumo[f"{chave}_reached"] = f"{len(atingidos)}/{len(rodadas)}"

    media_tempo, desvio_tempo = _media_desvio(
        [float(r["timing"]["train_effective_s"]) for r in runs]
    )
    resumo["train_effective_s"] = media_tempo
    resumo["train_effective_s_spread"] = desvio_tempo

    return resumo
