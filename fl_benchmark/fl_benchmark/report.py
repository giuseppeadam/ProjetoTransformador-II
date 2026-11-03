"""Persistência dos resultados do benchmark.

Cada execução grava dois artefatos em `results/`:

* `<estrategia>_<particao>.json` — registro completo e legível por máquina:
  curva por round, hiperparâmetros, tempos e o desbalanceamento efetivo da
  partição. É o insumo da consolidação comparativa.
* `<ESTRATEGIA>.md` — resumo da estratégia, regenerado a partir de todos os
  JSONs dela. Regenerar em vez de acrescentar mantém o arquivo idempotente:
  repetir uma execução atualiza a linha correspondente em vez de duplicá-la.
"""

import json
import math
import os
from pathlib import Path

from fl_benchmark import datasets, metrics, strategies

# Variável de ambiente que, quando definida, decide onde os relatórios vão.
RESULTS_DIR_ENV = "FL_BENCHMARK_RESULTS_DIR"


def results_dir() -> Path:
    """Diretório onde os relatórios são gravados.

    O Flower executa a aplicação a partir de uma cópia em `~/.flwr/apps/`, e um
    caminho relativo ao módulo gravaria os resultados lá dentro, fora do
    repositório. Por isso o destino é ancorado na raiz do projeto, localizada a
    partir da pasta de dados.
    """
    definido = os.environ.get(RESULTS_DIR_ENV)
    if definido:
        return Path(definido)
    try:
        return datasets.repo_root() / "fl_benchmark" / "results"
    except FileNotFoundError:
        return Path.cwd() / "results"

PARTICAO_LABEL = {"iid": "IID", "noniid": "non-IID", "natural": "natural (por país)"}

# Ordem de apresentação: o controle primeiro, depois os regimes heterogêneos.
ORDEM_PARTICOES = {"iid": 0, "noniid": 1, "natural": 2}

COLUNAS = (
    ("Partição", "particao"),
    ("Sementes", "seeds"),
    ("Acurácia", "accuracy"),
    ("F1", "f1"),
    ("ROC-AUC", "auc"),
    ("Rounds p/ 95% do teto", "rounds_95"),
    ("Rounds p/ 99% do teto", "rounds_99"),
    ("Treino efetivo", "train_effective"),
    ("Desvio entre clientes", "accuracy_std"),
    ("Pior cliente", "accuracy_min"),
)


def _num(valor, casas: int = 4) -> str:
    if valor is None:
        return "—"
    if isinstance(valor, float) and math.isnan(valor):
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def _segundos(valor) -> str:
    if valor is None:
        return "—"
    return f"{valor:.2f} s".replace(".", ",")


def dataset_dir(dataset: str) -> Path:
    """Diretório de resultados de um dataset.

    Os artefatos são separados por dataset porque o nome de uma execução é
    `<estrategia>_<particao>_seed<N>` e dois datasets podem compartilhar o mesmo
    regime de partição: sem a separação, executar `iid` no segundo dataset
    sobrescreve silenciosamente os resultados `iid` do primeiro.
    """
    return results_dir() / dataset


def _json_path(dataset: str, strategy: str, partition: str, seed: int) -> Path:
    return dataset_dir(dataset) / f"{strategy}_{partition}_seed{seed}.json"


def datasets_com_resultados() -> list[str]:
    """Datasets que já têm execuções gravadas."""
    base = results_dir()
    if not base.is_dir():
        return []
    encontrados = []
    for caminho in sorted(base.iterdir()):
        if caminho.is_dir() and caminho.name != "logs":
            if any(caminho.glob("*_seed*.json")):
                encontrados.append(caminho.name)
    return encontrados


def load_runs(strategy: str | None = None, dataset: str | None = None) -> list[dict]:
    """Carrega os JSONs de execução, filtrando por estratégia e/ou dataset.

    Só considera arquivos cujo nome segue `<estrategia>_<particao>_seed<N>.json`,
    de modo que outros artefatos — como o baseline centralizado ou os registros
    de calibração — não sejam confundidos com execuções federadas.
    """
    alvos = [strategy] if strategy else list(strategies.NAMES)
    bases = [dataset_dir(dataset)] if dataset else [
        dataset_dir(d) for d in datasets_com_resultados()
    ]
    registros = []
    for base in bases:
        for nome in alvos:
            for caminho in sorted(base.glob(f"{nome}_*_seed*.json")):
                with caminho.open(encoding="utf-8") as arquivo:
                    registros.append(json.load(arquivo))
    return registros


def group_runs(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Agrupa execuções por estratégia e partição, reunindo as repetições.

    Cada grupo contém uma execução por semente. Reportar a média e a dispersão
    entre sementes, em vez de uma execução única, é o que permite distinguir
    diferença de método de variação de execução — necessário aqui porque a
    agregação dos pesos no servidor não é bit-a-bit reproduzível: a ordem de
    chegada das respostas dos clientes varia, e diferenças da ordem de 1e-7 nos
    pesos se amplificam ao longo dos rounds.
    """
    grupos: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grupos.setdefault((run["strategy"], run["partition"]), []).append(run)
    for grupo in grupos.values():
        grupo.sort(key=lambda r: r["config"]["seed"])
    return grupos


def mais_menos(media, desvio, casas: int = 4, sufixo: str = "") -> str:
    """Formata `média ± dispersão`, omitindo o sufixo quando há uma só semente."""
    if media is None or (isinstance(media, float) and math.isnan(media)):
        return "—"
    texto = f"{media:.{casas}f}".replace(".", ",")
    if desvio is not None and not (isinstance(desvio, float) and math.isnan(desvio)) and desvio > 0:
        texto += " ± " + f"{desvio:.{casas}f}".replace(".", ",")
    return texto + sufixo


def teto_centralizado(dataset: str) -> float | None:
    """Acurácia do baseline centralizado do dataset, se já foi executado.

    Serve de referência para os alvos de convergência. Acurácia absoluta não
    transfere entre datasets — 0,99 é exigente onde o teto é 0,993 e impossível
    onde o teto é 0,719 — então os alvos são frações deste valor.
    """
    caminho = dataset_dir(dataset) / "baseline.json"
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        if dados.get("dataset") != dataset:
            return None
        return float(dados["final"]["accuracy"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _rounds_celula(resumo: dict, alvo: int) -> str:
    """Formata a coluna de rounds, explicitando repetições que não atingiram o alvo."""
    chave = f"rounds_to_{alvo}"
    media = resumo.get(chave)
    alcancado = resumo.get(f"{chave}_reached", "")
    num_seeds = resumo["num_seeds"]

    if media is None:
        return "não atingido"

    texto = mais_menos(media, resumo.get(f"{chave}_spread"), casas=1)
    if alcancado and alcancado != f"{num_seeds}/{num_seeds}":
        texto += f" (em {alcancado})"
    return texto


def linha_tabela(particao: str, runs: list[dict]) -> dict:
    teto = teto_centralizado(runs[0]["dataset"])
    resumo = metrics.resumo_repeticoes(runs, teto=teto)
    return {
        "particao": PARTICAO_LABEL.get(particao, particao),
        "seeds": str(resumo["num_seeds"]),
        "accuracy": mais_menos(resumo["central_accuracy"], resumo["central_accuracy_spread"]),
        "f1": mais_menos(resumo["central_f1"], resumo["central_f1_spread"]),
        "auc": mais_menos(resumo["central_auc"], resumo["central_auc_spread"]),
        "rounds_95": _rounds_celula(resumo, 95),
        "rounds_99": _rounds_celula(resumo, 99),
        "train_effective": mais_menos(
            resumo["train_effective_s"], resumo["train_effective_s_spread"], casas=2, sufixo=" s"
        ),
        "accuracy_std": mais_menos(
            resumo["federated_accuracy_std"], resumo["federated_accuracy_std_spread"]
        ),
        "accuracy_min": mais_menos(
            resumo["federated_accuracy_min"], resumo["federated_accuracy_min_spread"]
        ),
    }


def _tabela(runs: list[dict]) -> list[str]:
    grupos = group_runs(runs)
    ordem = ORDEM_PARTICOES
    cabecalho = [titulo for titulo, _ in COLUNAS]

    linhas = ["| " + " | ".join(cabecalho) + " |"]
    linhas.append("|" + "|".join(["---"] * len(cabecalho)) + "|")
    for (_, particao), grupo in sorted(grupos.items(), key=lambda kv: ordem.get(kv[0][1], 9)):
        valores = linha_tabela(particao, grupo)
        linhas.append("| " + " | ".join(valores[chave] for _, chave in COLUNAS) + " |")
    return linhas


def _curva(runs: list[dict]) -> list[str]:
    """Curva por round, com a média entre as sementes de uma configuração."""
    import numpy as np

    rodadas = sorted({linha["round"] for run in runs for linha in run["rounds"]})
    uma_semente = len(runs) == 1

    linhas = [
        "| Round | Acurácia central | F1 central | ROC-AUC central "
        "| Acurácia federada | Tempo de treino |",
        "|---|---|---|---|---|---|",
    ]

    def agregar(rodada: int, campo: str) -> tuple[float, float]:
        valores = [
            float(linha[campo])
            for run in runs
            for linha in run["rounds"]
            if linha["round"] == rodada and not math.isnan(float(linha[campo]))
        ]
        if not valores:
            return float("nan"), float("nan")
        return float(np.mean(valores)), (float(np.std(valores)) if len(valores) > 1 else 0.0)

    for rodada in rodadas:
        celulas = []
        for campo, casas, sufixo in (
            ("central_accuracy", 4, ""),
            ("central_f1", 4, ""),
            ("central_auc", 4, ""),
            ("federated_accuracy", 4, ""),
            ("train_time_s", 2, " s"),
        ):
            media, desvio = agregar(rodada, campo)
            celulas.append(
                mais_menos(media, 0.0 if uma_semente else desvio, casas=casas, sufixo=sufixo)
            )
        linhas.append(f"| {rodada} | " + " | ".join(celulas) + " |")
    return linhas


def render_strategy(strategy: str, dataset: str) -> str:
    """Monta o conteúdo do README de uma estratégia, para um dataset."""
    modulo = strategies.get(strategy)
    runs = load_runs(strategy, dataset)
    if not runs:
        raise RuntimeError(
            f"nenhum resultado para a estratégia {strategy!r} no dataset {dataset!r}"
        )

    grupos = group_runs(runs)
    referencia = runs[-1]
    cfg = referencia["config"]
    spec = datasets.get_spec(referencia["dataset"])
    sementes = sorted({r["config"]["seed"] for r in runs})

    partes = [f"# {modulo.TITLE}", ""]
    partes.append(f"> {modulo.SUMMARY}")
    partes.append(">")
    partes.append(f"> **Referência:** {modulo.REFERENCE}")

    hiper = modulo.hyperparams(cfg)
    if hiper:
        formatado = " · ".join(f"`{k} = {v}`" for k, v in hiper.items())
        partes.append(">")
        partes.append(f"> **Hiperparâmetros próprios:** {formatado}")
    partes.append("")

    partes.append("## Resultados")
    partes.append("")
    partes.extend(_tabela(runs))
    partes.append("")
    rotulo_sementes = ", ".join(str(s) for s in sementes)
    partes.append(
        f"Dataset `{spec.name}` · {cfg['num-clients']} clientes · "
        f"{cfg['num-server-rounds']} rounds · {cfg['local-epochs']} épocas locais · "
        f"α de Dirichlet = {str(cfg['dirichlet-alpha']).replace('.', ',')} · "
        f"sementes {rotulo_sementes} · dispositivo `{referencia['device']}`"
    )
    partes.append("")
    teto = teto_centralizado(referencia["dataset"])
    if teto is not None:
        absolutos = metrics.alvos_absolutos(referencia, teto)
        detalhe = " · ".join(
            f"{int(f * 100)}% do teto = {_num(v)}" for f, v in sorted(absolutos.items())
        )
        partes.append(
            f"Teto centralizado do dataset: {_num(teto)}. Alvos de convergência: {detalhe}."
        )
        partes.append("")
    if len(sementes) > 1:
        partes.append(
            f"Os valores são a média entre {len(sementes)} execuções independentes, "
            "seguida da dispersão entre elas. A dispersão existe porque a agregação dos "
            "pesos no servidor não é reproduzível bit a bit — a ordem de chegada das "
            "respostas dos clientes varia entre execuções. Diferenças entre estratégias "
            "menores que essa dispersão não são conclusivas."
        )
        partes.append("")

    ordem = ORDEM_PARTICOES
    for (_, particao), grupo in sorted(grupos.items(), key=lambda kv: ordem.get(kv[0][1], 9)):
        rotulo = PARTICAO_LABEL.get(particao, particao)
        partes.append(f"### Curva por round — partição {rotulo}")
        partes.append("")
        partes.extend(_curva(grupo))
        partes.append("")

    observacoes = strategies.notes(strategy)
    if observacoes:
        partes.append("## Comportamento do método")
        partes.append("")
        for nota in observacoes:
            partes.append(f"- {nota}")
        partes.append("")

    partes.append("## Como reproduzir")
    partes.append("")
    partes.append("```bash")
    partes.append("cd fl_benchmark")
    for (_, particao) in sorted(grupos, key=lambda k: ordem.get(k[1], 9)):
        for semente in sementes:
            partes.append(
                f"flwr run . --federation-config num-supernodes={cfg['num-clients']} "
                f"--run-config 'strategy=\"{strategy}\" partition=\"{particao}\" "
                f"dataset=\"{cfg['dataset']}\" num-clients={cfg['num-clients']} "
                f"seed={semente}'"
            )
    partes.append("```")
    partes.append("")

    return "\n".join(partes)


def write(
    *,
    strategy: str,
    partition: str,
    dataset: str,
    device: str,
    config: dict,
    partition_summary: dict,
    rounds: list[dict],
    final: dict,
    timing: dict,
    convergence: dict,
) -> tuple[Path, Path]:
    """Grava o JSON da execução e regenera o README da estratégia."""
    dataset_dir(dataset).mkdir(parents=True, exist_ok=True)

    registro = {
        "strategy": strategy,
        "partition": partition,
        "dataset": dataset,
        "device": device,
        "config": config,
        "partition_summary": partition_summary,
        "rounds": rounds,
        "final": final,
        "timing": timing,
        "convergence": convergence,
    }

    caminho_json = _json_path(dataset, strategy, partition, int(config["seed"]))
    with caminho_json.open("w", encoding="utf-8") as arquivo:
        json.dump(registro, arquivo, indent=2, ensure_ascii=False)

    caminho_md = dataset_dir(dataset) / f"{strategy.upper()}.md"
    caminho_md.write_text(render_strategy(strategy, dataset), encoding="utf-8")

    return caminho_json, caminho_md
