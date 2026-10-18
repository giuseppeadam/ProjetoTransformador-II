"""Argumentos comuns a todas as estratégias do benchmark.

Tudo o que é idêntico entre as estratégias é montado aqui. É isso que garante
que a comparação meça a forma de agregar os pesos, e não uma diferença
acidental de configuração entre uma execução e outra.
"""

from fl_benchmark import metrics


def common_kwargs(cfg: dict) -> dict:
    """Argumentos repassados ao construtor de qualquer estratégia.

    `min_train_nodes` e `min_evaluate_nodes` são fixados no número total de
    clientes para que todos participem de todos os rounds: o benchmark compara
    estratégias de agregação, não políticas de amostragem de clientes.
    """
    num_clients = int(cfg["num-clients"])
    return {
        "fraction_train": 1.0,
        "fraction_evaluate": float(cfg["fraction-evaluate"]),
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
        "weighted_by_key": "num-examples",
        "train_metrics_aggr_fn": metrics.aggregate,
        "evaluate_metrics_aggr_fn": metrics.aggregate,
    }
