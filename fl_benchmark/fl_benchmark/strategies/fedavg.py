"""FedAvg — média ponderada dos pesos."""

from flwr.serverapp.strategy import FedAvg

from ._base import common_kwargs

NAME = "fedavg"
TITLE = "FedAvg"
REFERENCE = "McMahan et al., 2017 — arxiv.org/abs/1602.05629"
SUMMARY = (
    "A estratégia de referência de Federated Learning. A cada round o servidor "
    "calcula a média dos pesos recebidos, ponderada pelo número de exemplos de "
    "cada cliente — um cliente com dez vezes mais dados pesa dez vezes mais no "
    "modelo global. Simples e eficiente quando os dados são homogêneos, mas "
    "sofre com *client drift*: se as distribuições locais divergem, os modelos "
    "treinados localmente apontam para mínimos diferentes e a média entre eles "
    "pode não ser boa para nenhum dos clientes."
)


def hyperparams(cfg: dict) -> dict:
    return {}


def build(cfg: dict) -> FedAvg:
    return FedAvg(**common_kwargs(cfg))
