"""FedTrimmedAvg — média truncada, descartando os extremos."""

from flwr.serverapp.strategy import FedTrimmedAvg

from ._base import common_kwargs

NAME = "fedtrimmedavg"
TITLE = "FedTrimmedAvg"
REFERENCE = "Yin et al., 2018 — arxiv.org/abs/1803.01498"
SUMMARY = (
    "Estratégia de agregação robusta. Em vez de promediar todos os valores "
    "recebidos, ordena os clientes parâmetro por parâmetro e descarta uma "
    "fração `beta` das pontas antes de calcular a média. A motivação original é "
    "resistência a clientes defeituosos ou maliciosos, cujos pesos aparecem "
    "como valores extremos e seriam eliminados pelo corte."
)
NOTES = (
    "O corte é alheio ao motivo pelo qual um cliente é extremo. Sob partição "
    "non-IID, o cliente cujos dados são os mais atípicos é legítimo e, ainda "
    "assim, é exatamente o que tende a ser descartado — o que pode piorar o "
    "resultado em vez de protegê-lo.",
    "Com 5 clientes e `beta = 0,2`, o corte remove 1 cliente de cada extremo e "
    "agrega apenas os 3 centrais, descartando 40% dos participantes do round.",
)


def hyperparams(cfg: dict) -> dict:
    return {"beta": float(cfg["trim-beta"])}


def build(cfg: dict) -> FedTrimmedAvg:
    return FedTrimmedAvg(**common_kwargs(cfg), beta=float(cfg["trim-beta"]))
