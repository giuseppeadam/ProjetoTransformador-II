"""Particionamento do conjunto de treino entre as instituições simuladas.

O benchmark compara dois regimes:

`iid`
    Cada cliente recebe uma amostra aleatória de tamanho igual. A distribuição
    de classes é praticamente idêntica entre clientes, então o modelo local de
    cada um já é próximo do modelo global ideal. É o cenário de controle.

`noniid`
    Desbalanceamento de rótulos governado por uma distribuição de Dirichlet.
    Para cada classe, a proporção que vai a cada cliente é sorteada de
    `Dirichlet(alpha)`: com `alpha` alto o resultado tende ao IID; com `alpha`
    baixo alguns clientes ficam quase com uma única classe. É o cenário que
    separa as estratégias, porque o modelo local passa a divergir do global —
    o fenômeno conhecido como *client drift*.

A semente é fixa nos dois regimes, de modo que todas as estratégias são
avaliadas exatamente sobre a mesma divisão de dados.
"""

import numpy as np

REGIMES = ("iid", "noniid")

# Com alpha baixo o sorteio de Dirichlet pode deixar um cliente quase vazio.
# Abaixo deste número de amostras o sorteio é repetido, prática usual nas
# implementações de referência de partição non-IID.
MIN_SAMPLES_POR_CLIENTE = 40
MAX_TENTATIVAS = 100


def _iid(num_samples: int, num_partitions: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    return [parte.copy() for parte in np.array_split(indices, num_partitions)]


def _dirichlet(y: np.ndarray, num_partitions: int, alpha: float, seed: int) -> list[np.ndarray]:
    classes = np.unique(y)

    for tentativa in range(MAX_TENTATIVAS):
        rng = np.random.default_rng(seed + tentativa)
        por_cliente: list[list[np.ndarray]] = [[] for _ in range(num_partitions)]

        for classe in classes:
            indices_classe = np.where(y == classe)[0]
            rng.shuffle(indices_classe)

            proporcoes = rng.dirichlet(np.repeat(alpha, num_partitions))
            cortes = (np.cumsum(proporcoes)[:-1] * len(indices_classe)).astype(int)
            for cliente, bloco in enumerate(np.split(indices_classe, cortes)):
                por_cliente[cliente].append(bloco)

        particoes = [np.concatenate(blocos) for blocos in por_cliente]
        if min(len(p) for p in particoes) >= MIN_SAMPLES_POR_CLIENTE:
            return [rng.permutation(p) for p in particoes]

    raise RuntimeError(
        f"não foi possível particionar com alpha={alpha} e {num_partitions} clientes "
        f"garantindo {MIN_SAMPLES_POR_CLIENTE} amostras por cliente; aumente alpha"
    )


def build(
    y: np.ndarray,
    num_partitions: int,
    regime: str,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Devolve, para cada cliente, os índices do conjunto de treino que lhe cabem."""
    if regime not in REGIMES:
        raise ValueError(f"regime de partição desconhecido: {regime!r}. Use um de {REGIMES}")
    if regime == "iid":
        return _iid(len(y), num_partitions, seed)
    return _dirichlet(y, num_partitions, alpha, seed)


def class_matrix(y: np.ndarray, particoes: list[np.ndarray]) -> np.ndarray:
    """Matriz `clientes × classes` com a contagem de amostras.

    Serve para documentar o grau de desbalanceamento efetivo de cada execução,
    em vez de apenas registrar o valor de `alpha`.
    """
    classes = np.unique(y)
    return np.array(
        [[int((y[p] == classe).sum()) for classe in classes] for p in particoes],
        dtype=int,
    )


def skew_summary(matriz: np.ndarray) -> dict:
    """Resume o desbalanceamento de uma matriz `clientes × classes`.

    `max_class_share` é a maior fração de uma única classe observada em algum
    cliente: 0,5 indica cliente equilibrado, valores próximos de 1,0 indicam
    cliente praticamente de classe única.
    """
    tamanhos = matriz.sum(axis=1)
    fracoes = matriz / tamanhos[:, None]
    return {
        "client_sizes": tamanhos.tolist(),
        "class_counts": matriz.tolist(),
        "max_class_share": round(float(fracoes.max()), 4),
        "mean_max_class_share": round(float(fracoes.max(axis=1).mean()), 4),
    }


def client_arrays(
    X: np.ndarray,
    y: np.ndarray,
    particoes: list[np.ndarray],
    partition_id: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separa a partição de um cliente em treino e validação locais.

    A validação é estratificada quando o cliente tem as duas classes. Sob
    partição non-IID isso nem sempre acontece: um cliente pode receber apenas
    uma classe, e nesse caso a divisão é feita sem estratificação.
    """
    from sklearn.model_selection import train_test_split

    indices = particoes[partition_id]
    X_parte, y_parte = X[indices], y[indices]

    estratificar = y_parte if len(np.unique(y_parte)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_parte,
        y_parte,
        test_size=val_fraction,
        random_state=seed,
        stratify=estratificar,
    )
    return X_tr, y_tr, X_val, y_val
