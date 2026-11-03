"""ServerApp: coordena os rounds e registra o resultado da execução.

Este módulo é deliberadamente fino. Ele lê a configuração, pede ao registro de
estratégias o objeto correspondente e cuida da instrumentação — avaliação
centralizada por round, medição de tempo e gravação dos relatórios. A lógica de
cada método federado vive em `strategies/<nome>.py`.
"""

import math
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
    inicio_setup = time.perf_counter()

    torch.manual_seed(cfg["seed"])

    nome_dataset = cfg["dataset"]
    X_train, y_train, X_test, y_test = datasets.load(nome_dataset)
    device = engine.resolve_device(cfg["device"])
    loader_teste = engine.make_loader(X_test, y_test, cfg["eval-batch-size"], shuffle=False)

    # Registrado no relatório para documentar o desbalanceamento efetivo da
    # execução, em vez de apenas o valor nominal de alpha.
    particoes = partition.build_for(
        nome_dataset,
        y_train,
        num_partitions=cfg["num-clients"],
        regime=cfg["partition"],
        alpha=cfg["dirichlet-alpha"],
        seed=cfg["seed"],
    )
    resumo_particao = partition.skew_summary(partition.class_matrix(y_train, particoes))
    if cfg["partition"] == "natural":
        resumo_particao["group_names"] = partition.group_names(datasets.group_labels(nome_dataset))

    modelo_global = MLP(datasets.num_features(nome_dataset))
    arrays = ArrayRecord(modelo_global.state_dict())

    avaliacoes_centrais: dict[int, dict] = {}

    def avaliar_centralizado(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Avalia o modelo global no conjunto de teste que nenhum cliente viu."""
        inicio_avaliacao = time.perf_counter()
        modelo = MLP(datasets.num_features(nome_dataset))
        modelo.load_state_dict(arrays.to_torch_state_dict())
        resultado = engine.evaluate(modelo, loader_teste, device)
        # Inclui a construção do modelo e a carga dos pesos, não apenas o laço
        # de inferência: é o custo real que a avaliação central impõe ao round.
        resultado["central_eval_total_s"] = time.perf_counter() - inicio_avaliacao
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

    duracao_setup = time.perf_counter() - inicio_setup
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
        duracao_setup=duracao_setup,
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
    duracao_setup: float,
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
        tempo_cliente_mais_lento = float(metricas_treino.get("train_time_max_s", indefinido))
        tempo_avaliacao_clientes = float(metricas_federadas.get("eval_time_s", indefinido))
        tempo_avaliacao_central = float(central.get("central_eval_total_s", indefinido))

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
                "train_time_slowest_client_s": tempo_cliente_mais_lento,
                "eval_time_clients_s": tempo_avaliacao_clientes,
                "eval_time_central_s": tempo_avaliacao_central,
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
        timing=_resumo_tempos(linhas, tempo_treino_total, wall_clock, duracao_setup),
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


def _resumo_tempos(
    linhas: list[dict], tempo_treino_total: float, wall_clock: float, duracao_setup: float
) -> dict:
    """Consolida os tempos da execução.

    `train_effective_s` é o custo computacional agregado do treino; já
    `train_slowest_total_s` soma, por round, apenas o cliente mais lento, e é a
    parcela do tempo de parede que o treino de fato impõe, dado que os clientes
    treinam em paralelo. A diferença entre os dois cresce com a desigualdade de
    tamanho entre as partições.

    `setup_s` cobre carga do dataset, construção da partição e do modelo inicial
    — tudo o que antecede o primeiro round. Em datasets grandes deixa de ser
    desprezível, e sem medi-lo o custo por execução fica subestimado.
    """

    def soma(chave: str) -> float:
        valores = [
            float(l[chave]) for l in linhas if not math.isnan(float(l.get(chave, float("nan"))))
        ]
        return float(sum(valores)) if valores else float("nan")

    return {
        "train_effective_s": tempo_treino_total,
        "train_slowest_total_s": soma("train_time_slowest_client_s"),
        "eval_clients_total_s": soma("eval_time_clients_s"),
        "eval_central_total_s": soma("eval_time_central_s"),
        "setup_s": duracao_setup,
        "wall_clock_s": wall_clock,
        "mean_round_s": wall_clock / len(linhas) if linhas else float("nan"),
    }
