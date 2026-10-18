"""Registro das estratégias federadas comparadas no benchmark.

Cada estratégia vive em seu próprio módulo, com a construção do objeto e a
descrição do método lado a lado. Adicionar uma sexta estratégia é criar um
módulo com a mesma interface — `NAME`, `TITLE`, `REFERENCE`, `SUMMARY`,
`hyperparams()` e `build()` — e incluí-lo em `_MODULOS`.
"""

from types import ModuleType

from . import fedadam, fedavg, fedprox, fedtrimmedavg, fedyogi

_MODULOS: tuple[ModuleType, ...] = (fedavg, fedprox, fedadam, fedyogi, fedtrimmedavg)

# A ordem do registro é a ordem em que as estratégias aparecem nos relatórios:
# da referência (FedAvg) para as variações mais especializadas.
REGISTRY: dict[str, ModuleType] = {m.NAME: m for m in _MODULOS}
NAMES: tuple[str, ...] = tuple(REGISTRY)


def get(name: str) -> ModuleType:
    chave = name.strip().lower()
    if chave not in REGISTRY:
        disponiveis = ", ".join(NAMES)
        raise ValueError(f"estratégia desconhecida: {name!r}. Disponíveis: {disponiveis}")
    return REGISTRY[chave]


def build(name: str, cfg: dict):
    return get(name).build(cfg)


def notes(name: str) -> tuple[str, ...]:
    return tuple(getattr(get(name), "NOTES", ()))
