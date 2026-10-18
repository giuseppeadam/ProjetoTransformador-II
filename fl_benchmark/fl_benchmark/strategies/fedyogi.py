"""FedYogi — passo de Yogi aplicado no servidor."""

from flwr.serverapp.strategy import FedYogi

from ._base import common_kwargs

NAME = "fedyogi"
TITLE = "FedYogi"
REFERENCE = "Reddi et al., 2020 — arxiv.org/abs/2003.00295"
SUMMARY = (
    "Mesma ideia do FedAdam — otimizador adaptativo no servidor — trocando Adam "
    "por Yogi. A diferença está em como o momento de segunda ordem é atualizado: "
    "o Adam o faz de forma multiplicativa, o que pode fazer o tamanho efetivo do "
    "passo crescer rápido demais; o Yogi usa atualização aditiva, controlando "
    "esse crescimento. Na prática tende a ser mais estável que o FedAdam quando "
    "os updates dos clientes variam muito de magnitude entre rounds — o que é "
    "justamente o caso sob partição non-IID."
)


NOTES = (
    "`eta`, a taxa de aprendizado do servidor, foi calibrada por varredura na partição "
    "IID; o registro está em `calibration_eta_fedyogi.json`.",
    "A comparação mais informativa é com o FedAdam, de quem difere apenas na forma de "
    "atualizar o momento de segunda ordem. Sob partição non-IID o FedYogi apresentou "
    "dispersão de acurácia entre clientes menor que a do FedAdam e atendeu melhor o "
    "cliente de pior desempenho, o que é consistente com a atualização aditiva: ela "
    "limita o crescimento do passo efetivo quando os updates dos clientes variam muito "
    "de magnitude, situação que a heterogeneidade dos dados provoca.",
    "Como o FedAdam, não alcança 99% de acurácia em 10 rounds neste dataset: a média "
    "simples do FedAvg já é adequada para um problema linearmente separável, e o passo "
    "adaptativo no servidor adiciona inércia sem benefício correspondente. A vantagem "
    "dessa família aparece quando a agregação simples é ruim, não num problema fácil.",
)


def hyperparams(cfg: dict) -> dict:
    return {
        "eta": float(cfg["eta"]),
        "eta_l": float(cfg["eta-l"]),
        "beta_1": float(cfg["beta-1"]),
        "beta_2": float(cfg["beta-2"]),
        "tau": float(cfg["tau"]),
    }


def build(cfg: dict) -> FedYogi:
    return FedYogi(**common_kwargs(cfg), **hyperparams(cfg))
