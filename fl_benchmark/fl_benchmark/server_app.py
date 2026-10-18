"""ServerApp: coordena os rounds e registra o resultado da execução.

Este módulo é deliberadamente fino. Ele lê a configuração, pede ao registro de
estratégias o objeto correspondente e cuida da instrumentação — avaliação
centralizada por round, medição de tempo e gravação dos relatórios. A lógica de
cada método federado vive em `strategies/<nome>.py`.
"""

import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fl_benchmark import datasets, engine, metrics, partition, report, strategies
from fl_benchmark.model import MLP

app = ServerApp()


def _config(context: Context) -> dict:
    """Normaliza a configuração da execução para registro no relatório."""
    cfg = context.run_config
    return {
        "strategy": str(cfg["strategy"]).lower(),
        "partition": str(cfg["partition"]).lower(),
        "dataset": str(cfg["dataset"]),
        "num-clients": int(cfg["num-clients"]),
        "num-server-rounds": int(cfg["num-server-rounds"]),
        "local-epochs": int(cfg["local-epochs"]),
        "learning-rate": float(cfg["learning-rate"]),
        "batch-size": int(cfg["batch-size"]),
        "eval-batch-size": int(cfg["eval-batch-size"]),
        "fraction-evaluate": float(cfg["fraction-evaluate"]),
        "dirichlet-alpha": float(cfg["dirichlet-alpha"]),
        "val-fraction": float(cfg["val-fraction"]),
        "seed": int(cfg["seed"]),
        "device": str(cfg["device"]),
        "proximal-mu": float(cfg["proximal-mu"]),
        "trim-beta": float(cfg["trim-beta"]),
        "eta": float(cfg["eta"]),
        "eta-l": float(cfg["eta-l"]),
        "beta-1": float(cfg["beta-1"]),
        "beta-2": float(cfg["beta-2"]),
        "tau": float(cfg["tau"]),
        "convergence-target": float(cfg["convergence-target"]),
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = _config(context)
    modulo = strategies.get(cfg["strategy"])

    torch.manual_seed(cfg["seed"])

    nome_dataset = cfg["dataset"]
    X_train, y_train, X_test, y_test = datasets.load(nome_dataset)
    device = engine.resolve_device(cfg["device"])
    loader_teste = engine.make_loader(X_test, y_test, cfg["eval-batch-size"], shuffle=False)

    # Registrado no relatório para documentar o desbalanceamento efetivo da
    # execução, em vez de apenas o valor nominal de alpha.
    particoes = partition.build(
        y_train,
        num_partitions=cfg["num-clients"],
        regime=cfg["partition"],
        alpha=cfg["dirichlet-alpha"],
        seed=cfg["seed"],
    )
    resumo_particao = partition.skew_summary(partition.class_matrix(y_train, particoes))

    modelo_global = MLP(datasets.num_features(nome_dataset))
    arrays = ArrayRecord(modelo_global.state_dict())

    avaliacoes_centrais: dict[int, dict] = {}

    def avaliar_centralizado(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Avalia o modelo global no conjunto de teste que nenhum cliente viu."""
        modelo = MLP(datasets.num_features(nome_dataset))
        modelo.load_state_dict(arrays.to_torch_state_dict())
        resultado = engine.evaluate(modelo, loader_teste, device)
        avaliacoes_centrais[server_round] = resultado
        return MetricRecord(
            {
                "accuracy": resultado["accuracy"],
                "f1": resultado["f1"],
                "auc": resultado["auc"],
                "loss": resultado["loss"],
            }
        )

    estrategia = modulo.build(cfg)

    rotulo_device = engine.device_label(device)
    print(
        f"\n>>> estrategia: {modulo.TITLE} | particao: {cfg['partition']} "
        f"| device: {rotulo_device}\n"
    )

    inicio = time.perf_counter()
    resultado = estrategia.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": cfg["learning-rate"]}),
        num_rounds=cfg["num-server-rounds"],
        evaluate_fn=avaliar_centralizado,
    )
    wall_clock = time.perf_counter() - inicio

    _registrar(
        cfg=cfg,
        modulo=modulo,
        rotulo_device=rotulo_device,
        resultado=resultado,
        avaliacoes_centrais=avaliacoes_centrais,
        resumo_particao=resumo_particao,
        wall_clock=wall_clock,
    )


def _registrar(
    *,
    cfg: dict,
    modulo,
    rotulo_device: str,
    resultado,
    avaliacoes_centrais: dict[int, dict],
    resumo_particao: dict,
    wall_clock: float,
) -> None:
    """Consolida as métricas da execução e grava os relatórios."""
    treino = resultado.train_metrics_clientapp
    federado = resultado.evaluate_metrics_clientapp

    # O round 0 da avaliação centralizada corresponde ao modelo inicial, antes
    # de qualquer treino. Fica fora da curva e do cálculo de convergência.
    rodadas = sorted(r for r in avaliacoes_centrais if r >= 1)

    linhas = []
    tempo_treino_total = 0.0
    indefinido = float("nan")

    for rodada in rodadas:
        central = avaliacoes_centrais[rodada]
        metricas_treino = treino.get(rodada, {})
        metricas_federadas = federado.get(rodada, {})

        tempo = float(metricas_treino.get("train_time_s", 0.0))
        tempo_treino_total += tempo

        linhas.append(
            {
                "round": rodada,
                "central_accuracy": central["accuracy"],
                "central_f1": central["f1"],
                "central_auc": central["auc"],
                "central_loss": central["loss"],
                "federated_accuracy": float(metricas_federadas.get("accuracy", indefinido)),
                "federated_accuracy_std": float(
                    metricas_federadas.get("accuracy_std", indefinido)
                ),
                "federated_accuracy_min": float(
                    metricas_federadas.get("accuracy_min", indefinido)
                ),
                "federated_f1": float(metricas_federadas.get("f1", indefinido)),
                "federated_auc": float(metricas_federadas.get("auc", indefinido)),
                "train_loss": float(metricas_treino.get("train_loss", indefinido)),
                "train_time_s": tempo,
            }
        )

    ultima = linhas[-1]
    curva_central = {linha["round"]: linha["central_accuracy"] for linha in linhas}
    alvo = cfg["convergence-target"]

    report.write(
        strategy=cfg["strategy"],
        partition=cfg["partition"],
        dataset=cfg["dataset"],
        device=rotulo_device,
        config=cfg,
        partition_summary=resumo_particao,
        rounds=linhas,
        final={
            "central_accuracy": ultima["central_accuracy"],
            "central_f1": ultima["central_f1"],
            "central_auc": ultima["central_auc"],
            "central_loss": ultima["central_loss"],
            "federated_accuracy": ultima["federated_accuracy"],
            "federated_accuracy_std": ultima["federated_accuracy_std"],
            "federated_accuracy_min": ultima["federated_accuracy_min"],
        },
        timing={
            "train_effective_s": tempo_treino_total,
            "wall_clock_s": wall_clock,
            "mean_round_s": wall_clock / len(linhas) if linhas else indefinido,
        },
        convergence={
            "target": alvo,
            "rounds_to_target": metrics.rounds_to_target(curva_central, alvo),
        },
    )

    print(
        f"\n>>> {modulo.TITLE} / {cfg['partition']}: "
        f"acuracia {ultima['central_accuracy']:.4f} | "
        f"F1 {ultima['central_f1']:.4f} | "
        f"AUC {ultima['central_auc']:.4f} | "
        f"treino efetivo {tempo_treino_total:.2f}s | "
        f"wall-clock {wall_clock:.2f}s\n"
    )
