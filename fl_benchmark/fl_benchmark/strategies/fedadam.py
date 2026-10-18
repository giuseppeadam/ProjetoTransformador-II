"""FedAdam — passo de Adam aplicado no servidor."""

from flwr.serverapp.strategy import FedAdam

from ._base import common_kwargs

NAME = "fedadam"
TITLE = "FedAdam"
REFERENCE = "Reddi et al., 2020 — arxiv.org/abs/2003.00295"
SUMMARY = (
    "Trata a diferença entre os pesos agregados e os pesos globais anteriores "
    "como se fosse um gradiente, e aplica sobre ela um passo do otimizador Adam "
    "no servidor — com momento de primeira e segunda ordem acumulados entre "
    "rounds. Enquanto o FedAvg sempre dá um passo do tamanho da média recebida, "
    "o FedAdam adapta o tamanho do passo por parâmetro, o que normalmente reduz "
    "o número de rounds até a convergência. O preço é um conjunto maior de "
    "hiperparâmetros para ajustar."
)
NOTES = (
    "`eta` é a taxa de aprendizado do servidor — parâmetro que o FedAvg não possui, "
    "por não dar passo algum no servidor. Seu valor foi calibrado por varredura na "
    "partição IID; o registro está em `calibration_eta_fedadam.json`. Com `eta = 1,0` "
    "o treinamento divergiu, colapsando para 50% de acurácia nos últimos rounds.",
    "`eta_l` declara a taxa do cliente à estratégia, mas não controla o treino local "
    "nesta implementação: o cliente usa a `learning-rate` recebida via `train_config`. "
    "`tau` é o termo de estabilidade numérica, análogo ao epsilon do Adam convencional.",
    "Neste dataset o método converge mais devagar que o FedAvg, e nenhum `eta` testado "
    "alcançou 99% de acurácia em 10 rounds. O resultado é coerente com o problema: "
    "sendo linearmente separável, a média simples dos pesos já aponta direto para o "
    "ótimo, e o momento acumulado no servidor só adiciona inércia. A vantagem dos "
    "otimizadores adaptativos aparece quando a agregação simples é ruim — gradientes de "
    "magnitude muito variável entre clientes ou heterogeneidade severa — e não num "
    "problema fácil.",
)


def hyperparams(cfg: dict) -> dict:
    return {
        "eta": float(cfg["eta"]),
        "eta_l": float(cfg["eta-l"]),
        "beta_1": float(cfg["beta-1"]),
        "beta_2": float(cfg["beta-2"]),
        "tau": float(cfg["tau"]),
    }


def build(cfg: dict) -> FedAdam:
    return FedAdam(**common_kwargs(cfg), **hyperparams(cfg))
