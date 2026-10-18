"""Benchmark de estratégias de Federated Learning com Flower.

O pacote separa o que varia entre execuções do que precisa permanecer
constante para que a comparação seja justa:

`datasets`
    Registro de datasets, carga, binarização do alvo e normalização.
`partition`
    Divisão do treino entre as instituições simuladas, nos regimes IID e
    non-IID.
`model`
    A rede neural, idêntica em todas as execuções.
`engine`
    Laço de treino local e avaliação.
`metrics`
    Agregação das métricas dos clientes e indicadores derivados.
`strategies`
    Um módulo por estratégia federada comparada.
`report`
    Persistência dos resultados em JSON e Markdown.
"""

__all__ = [
    "datasets",
    "engine",
    "metrics",
    "model",
    "partition",
    "report",
    "strategies",
]
