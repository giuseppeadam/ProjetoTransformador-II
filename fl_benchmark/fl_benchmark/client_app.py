"""ClientApp: a instituição simulada.

Há um único cliente para todas as estratégias, de propósito — o que o
benchmark compara é a agregação feita no servidor, então variar o cliente
tornaria os resultados incomparáveis. A única ramificação é o termo proximal,
que a estratégia ativa enviando `proximal-mu` na configuração do round e que
permanece inerte (`0.0`) para as demais.

O cliente nunca envia dados, apenas pesos e métricas — é essa a propriedade
que caracteriza Federated Learning.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_benchmark import datasets, engine, partition
from fl_benchmark.model import MLP

app = ClientApp()


def _contexto_local(context: Context):
    """Monta o modelo e os dados locais a partir da configuração do round."""
    cfg = context.run_config
    nome_dataset = str(cfg["dataset"])

    X_train, y_train, _, _ = datasets.load(nome_dataset)
    particoes = partition.build(
        y_train,
        num_partitions=int(context.node_config["num-partitions"]),
        regime=str(cfg["partition"]),
        alpha=float(cfg["dirichlet-alpha"]),
        seed=int(cfg["seed"]),
    )
    X_tr, y_tr, X_val, y_val = partition.client_arrays(
        X_train,
        y_train,
        particoes,
        partition_id=int(context.node_config["partition-id"]),
        val_fraction=float(cfg["val-fraction"]),
        seed=int(cfg["seed"]),
    )

    device = engine.resolve_device(str(cfg["device"]))
    modelo = MLP(datasets.num_features(nome_dataset))
    return cfg, modelo, device, (X_tr, y_tr, X_val, y_val)


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, modelo, device, (X_tr, y_tr, _, _) = _contexto_local(context)
    modelo.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # A estratégia injeta o número do round em toda mensagem de treino, o que
    # permite variar a ordem dos batches entre rounds sem perder a
    # reprodutibilidade da execução como um todo.
    server_round = int(msg.content["config"].get("server-round", 0))
    semente = engine.derive_seed(
        int(cfg["seed"]),
        int(context.node_config["partition-id"]),
        server_round,
    )
    torch.manual_seed(semente)

    loader = engine.make_loader(
        X_tr, y_tr, int(cfg["batch-size"]), shuffle=True, seed=semente
    )

    # Presente apenas quando a estratégia é FedProx; as outras treinam com o
    # objetivo local puro.
    proximal_mu = float(msg.content["config"].get("proximal-mu", 0.0))

    loss, duracao = engine.train(
        modelo,
        loader,
        epochs=int(cfg["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        device=device,
        proximal_mu=proximal_mu,
    )

    conteudo = RecordDict(
        {
            "arrays": ArrayRecord(modelo.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": loss,
                    "train_time_s": duracao,
                    "num-examples": float(len(X_tr)),
                }
            ),
        }
    )
    return Message(content=conteudo, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, modelo, device, (_, _, X_val, y_val) = _contexto_local(context)
    modelo.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    loader = engine.make_loader(X_val, y_val, int(cfg["eval-batch-size"]), shuffle=False)
    resultado = engine.evaluate(modelo, loader, device)

    conteudo = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "loss": resultado["loss"],
                    "accuracy": resultado["accuracy"],
                    "f1": resultado["f1"],
                    "auc": resultado["auc"],
                    "num-examples": float(resultado["num_examples"]),
                }
            )
        }
    )
    return Message(content=conteudo, reply_to=msg)
