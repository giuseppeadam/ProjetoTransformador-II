import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx, FedYogi

from fl_addiction.task import Net, load_centralized_dataset, load_data
from fl_addiction.task import test as test_fn
from fl_addiction.task import train as train_fn

# =========================
# Cliente (roda em cada "instituicao" simulada)
# =========================
client_app = ClientApp()


@client_app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


# =========================
# Servidor (coordena os rounds)
# =========================
server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    strategy_name = context.run_config["strategy"].lower()

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # troca de estrategia so muda esse if - scaffold nao ta aqui porque nao
    # e built-in no flwr, precisaria mexer no cliente tambem
    if strategy_name == "fedavg":
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    elif strategy_name == "fedprox":
        strategy = FedProx(
            fraction_evaluate=fraction_evaluate,
            proximal_mu=context.run_config.get("proximal-mu", 0.01),
        )
    elif strategy_name in ("fedadam", "fedyogi"):
        cls = FedAdam if strategy_name == "fedadam" else FedYogi
        strategy = cls(
            fraction_evaluate=fraction_evaluate,
            eta=context.run_config.get("eta", 1e-2),
            eta_l=context.run_config.get("eta-l", 1e-1),
            beta_1=context.run_config.get("beta-1", 0.9),
            beta_2=context.run_config.get("beta-2", 0.99),
            tau=context.run_config.get("tau", 1e-9),
        )
    else:
        raise ValueError(f"estrategia desconhecida: {strategy_name!r}")

    print(f"\n>>> estrategia: {strategy_name}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test_fn(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
