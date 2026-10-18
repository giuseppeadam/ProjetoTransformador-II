"""FedProx — FedAvg com termo proximal no cliente."""

from flwr.serverapp.strategy import FedProx

from ._base import common_kwargs

NAME = "fedprox"
TITLE = "FedProx"
REFERENCE = "Li et al., 2018 — arxiv.org/abs/1812.06127"
SUMMARY = (
    "Ataca o *client drift* na origem: em vez de mudar a agregação, restringe o "
    "quanto cada cliente pode se afastar do modelo global. O cliente soma à sua "
    "perda o termo proximal `(mu/2) * ||w - w_global||^2`, que penaliza o "
    "distanciamento dos pesos recebidos no início do round. Com `mu = 0` o "
    "método se reduz exatamente ao FedAvg; quanto maior o `mu`, mais conservador "
    "o treino local. É a estratégia projetada para dados heterogêneos (non-IID)."
)
NOTES = (
    "A agregação no servidor é idêntica à do FedAvg. O que distingue o FedProx "
    "acontece no cliente: a estratégia apenas envia `proximal-mu` na configuração "
    "do round, e cabe ao cliente aplicar a penalidade na função de perda. Sem "
    "essa aplicação, selecionar FedProx produz resultados indistinguíveis de "
    "FedAvg.",
)


def hyperparams(cfg: dict) -> dict:
    return {"proximal_mu": float(cfg["proximal-mu"])}


def build(cfg: dict) -> FedProx:
    return FedProx(**common_kwargs(cfg), proximal_mu=float(cfg["proximal-mu"]))
