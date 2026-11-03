"""Registro de datasets do benchmark.

Cada dataset é declarado uma vez em `DATASETS` e passa a estar disponível para
todas as estratégias. Adicionar um novo dataset é adicionar uma entrada aqui —
nenhum código de estratégia, partição ou métrica precisa mudar.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.2

# Nome da pasta que guarda os CSVs na raiz do repositório.
DATA_DIRNAME = "archive"

# Variável de ambiente que, quando definida, tem prioridade sobre a busca
# automática. Útil para apontar os dados para outro lugar sem editar código.
DATA_DIR_ENV = "FL_BENCHMARK_DATA_DIR"


def _candidatos(nome_arquivo: str):
    """Locais onde o CSV é procurado, em ordem de prioridade.

    A busca existe porque o Flower empacota a aplicação e a executa de uma
    cópia em `~/.flwr/apps/`, de onde a raiz do repositório não é alcançável
    por caminho relativo ao módulo. Procurar em vários locais mantém o mesmo
    código funcionando tanto a partir do fonte quanto a partir do pacote
    instalado.
    """
    definido = os.environ.get(DATA_DIR_ENV)
    if definido:
        yield Path(definido) / nome_arquivo

    aqui = Path(__file__).resolve()

    # Executando a partir do repositório: fl_benchmark/fl_benchmark/ -> raiz.
    yield aqui.parents[2] / DATA_DIRNAME / nome_arquivo

    # Dados empacotados junto do próprio pacote.
    yield aqui.parent / "data" / nome_arquivo

    # Executando a partir do pacote instalado: sobe do diretório de trabalho
    # até encontrar uma pasta de dados que contenha o arquivo.
    atual = Path.cwd().resolve()
    for diretorio in [atual, *atual.parents]:
        yield diretorio / DATA_DIRNAME / nome_arquivo


def resolve_csv(nome_arquivo: str) -> Path:
    """Devolve o primeiro caminho existente para `nome_arquivo`."""
    procurados = []
    for candidato in _candidatos(nome_arquivo):
        if candidato not in procurados:
            procurados.append(candidato)
        if candidato.is_file():
            return candidato

    locais = "\n  ".join(str(p) for p in procurados)
    raise FileNotFoundError(
        f"CSV {nome_arquivo!r} não encontrado. Locais verificados:\n  {locais}\n"
        f"Defina {DATA_DIR_ENV} para apontar a pasta que contém o arquivo."
    )


@dataclass(frozen=True)
class DatasetSpec:
    """Descreve como carregar e preparar um dataset para o benchmark."""

    name: str
    csv_filename: str
    features: list[str]
    target_column: str
    # "csv": um arquivo plano, alvo derivado na carga.
    # "preparado": diretório já limpo, normalizado e dividido por cliente,
    # produzido por um script de preparação. Datasets grandes exigem isso: na
    # simulação cada cliente é um processo próprio, e repetir limpeza e
    # normalização em todos eles a cada round domina o custo.
    kind: str = "csv"
    # Coluna que identifica a unidade federada quando a partição é lida dos
    # dados em vez de sorteada (regime "natural").
    group_column: str | None = None
    # Coluna contínua é binarizada pela mediana, o que garante classes
    # equilibradas sem escolher um limiar arbitrário.
    binarize_by_median: bool = True
    description: str = ""
    notes: list[str] = field(default_factory=list)


PRODUCTIVITY = DatasetSpec(
    name="productivity",
    csv_filename="student_productivity_distraction_dataset_20000.csv",
    features=[
        "age",
        "study_hours_per_day",
        "sleep_hours",
        "phone_usage_hours",
        "social_media_hours",
        "youtube_hours",
        "gaming_hours",
        "breaks_per_day",
        "coffee_intake_mg",
        "exercise_minutes",
        "assignments_completed",
        "attendance_percentage",
        "stress_level",
        "focus_score",
    ],
    target_column="productivity_score",
    description=(
        "Hábitos de estudo e distração digital de 20.000 estudantes. "
        "Alvo: produtividade acima da mediana."
    ),
    notes=[
        "As features são observações de estudantes, mas o alvo `productivity_score` "
        "não é uma medição: é uma coluna **derivada por fórmula** a partir de 6 das 14 "
        "features (study_hours_per_day, focus_score, sleep_hours, phone_usage_hours, "
        "stress_level, attendance_percentage). Por isso uma regressão linear sobre o "
        "conjunto atinge R² = 1,0000 — o alvo é, por construção, função exata delas.",
        "As 8 features restantes não entram na fórmula do alvo e têm coeficiente zero "
        "no modelo — foram mantidas de propósito, pois saber ignorá-las faz parte do "
        "que o benchmark mede.",
        "Como o alvo é derivado, o problema é linearmente separável e qualquer "
        "estratégia satura perto de 100% de acurácia no regime IID. A comparação entre "
        "estratégias se dá no regime non-IID e nas métricas de eficiência.",
    ],
)

PISA_FEATURES = (
    [f"ST326Q0{i}JA" for i in range(1, 7)]
    + ["ST322Q01JA", "ST322Q02JA", "ST322Q06JA", "ST322Q07JA", "ST296Q04JA", "ST097Q06DA"]
    + ["ESCS", "HOMEPOS", "ICTRES", "BELONG", "BULLIED"]
)

PISA = DatasetSpec(
    name="pisa",
    csv_filename="pisa_preparado",
    features=PISA_FEATURES,
    target_column="acima_mediana",
    binarize_by_median=False,
    kind="preparado",
    group_column="CNT",
    description=(
        "PISA 2025, questionário do estudante: uso de recursos digitais, autocontrole "
        "digital, contexto socioeconômico e desempenho em matemática. "
        "Alvo: nota acima da mediana global."
    ),
    notes=[
        "Os dados de origem foram extraídos com `user_missing=True`, que preserva os "
        "códigos de ausência do SPSS em vez de convertê-los em nulo. Valores como 97 "
        "(não aplicável) e 99 (sem resposta) constavam como medições, e `isna()` "
        "reportava zero nulos. Sem a limpeza, a correlação de ESCS com o desempenho "
        "aparece como -0,150 quando na verdade é +0,404 — sinal invertido.",
        "A amostra efetiva é de 303.681 dos 755.721 registros, por exigir as 17 features "
        "e a nota presentes. A perda não é aleatória: países que não aplicaram partes do "
        "questionário saem, o que altera a composição da amostra.",
        "O alvo é a média das dez plausible values de matemática. Elas não são dez notas, "
        "mas dez imputações da mesma proficiência latente; usar a média subestima o erro "
        "de medida, o que o procedimento da OCDE trata com análise múltipla.",
        "O peso amostral `W_FSTUWT` não é utilizado. Os resultados comparam estratégias "
        "de agregação federada e não constituem estimativas populacionais nacionais.",
        "A normalização é global, ajustada no conjunto de treino. Em federated learning "
        "estrito, estatísticas globais exigiriam um round prévio de agregação.",
    ],
)

DATASETS: dict[str, DatasetSpec] = {
    PRODUCTIVITY.name: PRODUCTIVITY,
    PISA.name: PISA,
}

# Cache em nível de módulo: numa simulação os clientes compartilham o processo,
# então o CSV é lido e normalizado uma única vez por execução.
_CACHE: dict[str, tuple] = {}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        disponiveis = ", ".join(sorted(DATASETS))
        raise ValueError(f"dataset desconhecido: {name!r}. Disponíveis: {disponiveis}")
    return DATASETS[name]


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega o dataset e devolve `(X_train, y_train, X_test, y_test)`.

    O conjunto de teste é separado antes de qualquer partição entre clientes:
    nenhum cliente treina nem valida sobre ele, o que o torna uma base honesta
    para a avaliação centralizada do modelo global.

    O `StandardScaler` é ajustado somente no treino e depois aplicado ao teste,
    evitando que estatísticas do conjunto de teste vazem para o treinamento.
    """
    if name in _CACHE:
        return _CACHE[name]

    spec = get_spec(name)
    if spec.kind == "preparado":
        _CACHE[name] = _load_preparado(spec)
        return _CACHE[name]

    df = pd.read_csv(resolve_csv(spec.csv_filename))

    faltando = [c for c in [*spec.features, spec.target_column] if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas ausentes no CSV de {name!r}: {faltando}")

    X = df[spec.features].to_numpy(dtype=np.float32)
    bruto = df[spec.target_column].to_numpy(dtype=np.float64)
    if spec.binarize_by_median:
        y = (bruto > np.median(bruto)).astype(np.float32)
    else:
        y = bruto.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    _CACHE[name] = (X_train, y_train, X_test, y_test)
    return _CACHE[name]


def num_features(name: str) -> int:
    return len(get_spec(name).features)


def repo_root() -> Path:
    """Raiz do repositório, deduzida da localização da pasta de dados.

    Serve de âncora para escrever os relatórios no repositório, e não na cópia
    da aplicação que o Flower mantém em `~/.flwr/apps/`.
    """
    qualquer_spec = next(iter(DATASETS.values()))
    return resolve_csv(qualquer_spec.csv_filename).resolve().parent.parent


def resolve_dir(nome: str) -> Path:
    """Localiza um diretório de dados, com a mesma busca aplicada aos CSVs."""
    procurados = []
    for candidato in _candidatos(nome):
        if candidato not in procurados:
            procurados.append(candidato)
        if candidato.is_dir():
            return candidato
    locais = "\n  ".join(str(p) for p in procurados)
    raise FileNotFoundError(
        f"diretório {nome!r} não encontrado. Locais verificados:\n  {locais}\n"
        f"Para o PISA, execute `python preparar_pisa.py` antes de treinar."
    )


# Grupos do conjunto de treino, por dataset. Preenchido junto com o cache de
# dados, para que a partição natural possa ser montada sem reler os arquivos.
_GRUPOS: dict[str, np.ndarray] = {}


def _load_preparado(spec: DatasetSpec):
    """Carrega um dataset já preparado: um arquivo por cliente mais o teste central.

    O treino é a concatenação dos arquivos de cliente, na ordem alfabética dos
    nomes — o que torna a composição reproduzível. A coluna de agrupamento é
    preservada em `_GRUPOS` para alimentar o regime de partição natural.
    """
    base = resolve_dir(spec.csv_filename)
    pasta_clientes = base / "clientes"
    teste_path = base / "teste_central.parquet"

    if not pasta_clientes.is_dir() or not teste_path.is_file():
        raise FileNotFoundError(
            f"diretório {base} incompleto: esperados 'clientes/' e 'teste_central.parquet'. "
            f"Execute `python preparar_pisa.py`."
        )

    arquivos = sorted(pasta_clientes.glob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum arquivo de cliente em {pasta_clientes}")

    colunas = spec.features + [spec.target_column] + (
        [spec.group_column] if spec.group_column else []
    )
    treino = pd.concat(
        [pd.read_parquet(a, columns=colunas) for a in arquivos], ignore_index=True
    )
    teste = pd.read_parquet(teste_path, columns=colunas)

    X_train = treino[spec.features].to_numpy(dtype=np.float32)
    y_train = treino[spec.target_column].to_numpy(dtype=np.float32)
    X_test = teste[spec.features].to_numpy(dtype=np.float32)
    y_test = teste[spec.target_column].to_numpy(dtype=np.float32)

    if spec.group_column:
        _GRUPOS[spec.name] = treino[spec.group_column].to_numpy()

    return X_train, y_train, X_test, y_test


def group_labels(name: str) -> np.ndarray:
    """Rótulos de grupo do conjunto de treino, para a partição natural."""
    spec = get_spec(name)
    if not spec.group_column:
        raise ValueError(
            f"dataset {name!r} não declara `group_column`; partição natural indisponível"
        )
    if name not in _GRUPOS:
        load(name)
    return _GRUPOS[name]


def has_groups(name: str) -> bool:
    return get_spec(name).group_column is not None
