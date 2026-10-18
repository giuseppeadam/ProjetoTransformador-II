"""Consolida as execuções do benchmark em `results/README.md`.

Lê todos os JSONs gravados pelas execuções, agrupa as repetições de cada
configuração e monta a tabela comparativa das estratégias, separada por regime
de partição. Combinações ausentes aparecem como linha vazia, em vez de
desaparecer da tabela — uma tabela que omite o que não foi medido sugere
cobertura que não existe.

Uso:
    python make_summary.py
"""

import json
import math
from pathlib import Path

from fl_benchmark import datasets, metrics, report, strategies

PARTICOES = ("iid", "noniid")
PARTICAO_TITULO = {
    "iid": "Partição IID (controle)",
    "noniid": "Partição non-IID (Dirichlet)",
}


def _num(valor, casas: int = 4) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def _carregar_baseline() -> dict | None:
    caminho = report.results_dir() / "baseline.json"
    if not caminho.is_file():
        return None
    with caminho.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _indexar() -> dict[tuple[str, str], list[dict]]:
    """Agrupa as execuções por estratégia e partição, reunindo as repetições."""
    return report.group_runs(report.load_runs())


def _tabela(execucoes: dict[tuple[str, str], list[dict]], particao: str) -> list[str]:
    linhas = [
        "| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% "
        "| Rounds p/ ≥99% | Treino efetivo (s) | Desvio entre clientes | Pior cliente |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nome in strategies.NAMES:
        titulo = strategies.get(nome).TITLE
        grupo = execucoes.get((nome, particao))
        if not grupo:
            linhas.append(f"| {titulo} | — | — | — | — | — | — | — | — | — |")
            continue
        celulas = report.linha_tabela(particao, grupo)
        linhas.append(
            "| {e} | {seeds} | {acc} | {f1} | {auc} | {r95} | {r99} | {t} | {std} | {pior} |".format(
                e=titulo,
                seeds=celulas["seeds"],
                acc=celulas["accuracy"],
                f1=celulas["f1"],
                auc=celulas["auc"],
                r95=celulas["rounds_95"],
                r99=celulas["rounds_99"],
                t=celulas["train_effective"].replace(" s", ""),
                std=celulas["accuracy_std"],
                pior=celulas["accuracy_min"],
            )
        )
    return linhas


def _skew(execucoes: dict[tuple[str, str], list[dict]]) -> list[str]:
    """Documenta o desbalanceamento efetivo da partição non-IID."""
    referencia = next(
        (
            execucoes[(nome, "noniid")][0]
            for nome in strategies.NAMES
            if execucoes.get((nome, "noniid"))
        ),
        None,
    )
    if referencia is None:
        return []

    resumo = referencia["partition_summary"]
    linhas = [
        "Todas as execuções non-IID com a mesma semente compartilham a mesma partição, de",
        "modo que a diferença entre estratégias não vem de sorteios distintos. A divisão",
        f"efetiva com a semente {referencia['config']['seed']} foi:",
        "",
        "| Cliente | Amostras | Classe 0 | Classe 1 | Maior fração de uma classe |",
        "|---|---|---|---|---|",
    ]
    for indice, (tamanho, contagens) in enumerate(
        zip(resumo["client_sizes"], resumo["class_counts"])
    ):
        maior = max(contagens) / tamanho if tamanho else float("nan")
        linhas.append(
            f"| {indice} | {tamanho} | {contagens[0]} | {contagens[1]} | {_num(maior, 3)} |"
        )
    return linhas


def _calibracoes() -> list[str]:
    """Registra as varreduras de hiperparâmetro que precederam as execuções."""
    arquivos = sorted(report.results_dir().glob("calibration_eta_*.json"))
    if not arquivos:
        return []

    linhas = [
        "FedAdam e FedYogi aplicam um passo de otimizador no servidor, controlado por",
        "`eta`. FedAvg não tem passo no servidor — apenas promedia — e portanto não tem",
        "esse hiperparâmetro para ajustar. Comparar as três com um `eta` arbitrário mediria",
        "a qualidade do ajuste em vez da do método, então `eta` foi calibrado por varredura",
        "na partição IID, que é o cenário de controle. Calibrar no regime non-IID, onde a",
        "comparação acontece, enviesaria o resultado a favor das adaptativas.",
        "",
    ]
    for caminho in arquivos:
        with caminho.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        titulo = strategies.get(dados["strategy"]).TITLE
        escolhido = str(dados["chosen_eta"]).replace(".", ",")
        linhas.append(f"**{titulo}** — `eta` escolhido: {escolhido}")
        linhas.append("")
        linhas.append("| eta | Acurácia final | Rounds p/ ≥95% | Rounds p/ ≥99% |")
        linhas.append("|---|---|---|---|")
        for entrada in dados["sweep"]:
            linhas.append(
                "| {eta} | {acc} | {r95} | {r99} |".format(
                    eta=str(entrada["eta"]).replace(".", ","),
                    acc=_num(entrada["final_accuracy"]),
                    r95=entrada["rounds_to_95"] or "não atingido",
                    r99=entrada["rounds_to_99"] or "não atingido",
                )
            )
        linhas.append("")
    return linhas


def _lidera(
    valores: dict[str, float],
    dispersoes: dict[str, float],
    menor_e_melhor: bool,
) -> str | None:
    """Nome da estratégia que lidera de forma distinguível, ou `None`.

    A comparação é contra a **segunda colocada**, não contra a pior: afirmar que
    uma estratégia lidera exige que ela se separe de quem vem logo atrás. Testar
    contra a amplitude total deixaria passar afirmações como "a melhor é X"
    quando X e Y estão empatados no topo e apenas Z está distante.

    As faixas são consideradas separadas quando a diferença entre as médias
    excede a soma das duas dispersões.
    """
    if len(valores) < 2:
        return None

    ordenadas = sorted(valores, key=lambda n: valores[n], reverse=not menor_e_melhor)
    primeiro, segundo = ordenadas[0], ordenadas[1]

    diferenca = abs(valores[primeiro] - valores[segundo])
    margem = (dispersoes.get(primeiro) or 0.0) + (dispersoes.get(segundo) or 0.0)
    return primeiro if diferenca > margem else None


def _destaques(execucoes: dict[tuple[str, str], list[dict]]) -> list[str]:
    """Extrai dos próprios números o que distingue as estratégias.

    Derivar os destaques dos dados, em vez de descrevê-los em texto fixo, impede
    que a leitura dos resultados contradiga a tabela quando o benchmark é
    reexecutado com outra configuração ou outro dataset.
    """
    resumos = {
        nome: metrics.resumo_repeticoes(execucoes[(nome, "noniid")])
        for nome in strategies.NAMES
        if execucoes.get((nome, "noniid"))
    }
    if not resumos:
        return []

    def titulo(nome: str) -> str:
        return strategies.get(nome).TITLE

    linhas = []

    # Só entram na comparação de eficiência as estratégias que alcançaram o alvo
    # em todas as sementes: uma média calculada sobre parte das repetições não é
    # comparável com uma calculada sobre todas.
    num_seeds = max(r["num_seeds"] for r in resumos.values())
    completo = f"{num_seeds}/{num_seeds}"
    convergiram = {
        nome: resumo["rounds_to_99"]
        for nome, resumo in resumos.items()
        if resumo.get("rounds_to_99") is not None
        and resumo.get("rounds_to_99_reached") == completo
    }
    parciais = [
        nome
        for nome, resumo in resumos.items()
        if resumo.get("rounds_to_99_reached") not in (completo, None)
        or resumo.get("rounds_to_99") is None
    ]

    if len(convergiram) > 1:
        dispersoes = {
            nome: resumos[nome].get("rounds_to_99_spread") or 0.0 for nome in convergiram
        }
        lider = _lidera(convergiram, dispersoes, menor_e_melhor=True)
        ordenadas = sorted(convergiram, key=lambda n: convergiram[n])
        primeiro, segundo = ordenadas[0], ordenadas[1]
        valor_primeiro = report.mais_menos(convergiram[primeiro], dispersoes[primeiro], 1)
        valor_segundo = report.mais_menos(convergiram[segundo], dispersoes[segundo], 1)

        if lider is not None:
            linhas.append(
                f"- **Eficiência de comunicação:** na partição non-IID, {titulo(lider)} "
                f"alcançou 99% de acurácia em {valor_primeiro} rounds, contra "
                f"{valor_segundo} da segunda colocada ({titulo(segundo)}) — menos rodadas "
                f"de troca de pesos para o mesmo resultado."
            )
        else:
            linhas.append(
                f"- **Eficiência de comunicação:** {titulo(primeiro)} ({valor_primeiro} "
                f"rounds para 99%) e {titulo(segundo)} ({valor_segundo}) ficaram com "
                f"faixas sobrepostas; a diferença não é conclusiva com "
                f"{num_seeds} sementes."
            )

    if len(convergiram) == 1 and parciais:
        unica = next(iter(convergiram))
        dispersao = resumos[unica].get("rounds_to_99_spread") or 0.0
        linhas.append(
            f"- **Convergência consistente:** {titulo(unica)} foi a única estratégia a "
            f"alcançar 99% de acurácia em todas as {num_seeds} sementes, em "
            f"{report.mais_menos(resumos[unica]['rounds_to_99'], dispersao, 1)} rounds. "
            f"As demais atingiram o alvo em parte das sementes ou em nenhuma, dentro dos "
            f"10 rounds executados."
        )

    # Quando só uma estratégia convergiu em todas as sementes, o destaque acima
    # já diz que as outras não convergiram — listá-las de novo seria redundante.
    if parciais and len(convergiram) != 1:
        nomes = ", ".join(titulo(n) for n in parciais)
        verbo = "chegaram" if len(parciais) > 1 else "chegou"
        linhas.append(
            f"- **Alvo de 99% não alcançado de forma consistente:** {nomes} não {verbo} a "
            f"99% de acurácia em todas as sementes dentro dos 10 rounds, e por isso "
            f"{'ficaram' if len(parciais) > 1 else 'ficou'} fora da comparação de "
            f"eficiência. A coluna *Rounds p/ ≥99%* da tabela indica em quantas sementes o "
            f"alvo foi atingido."
        )

    equidade = {n: r["federated_accuracy_std"] for n, r in resumos.items()}
    disp_equidade = {n: r["federated_accuracy_std_spread"] for n, r in resumos.items()}
    ordem_equidade = sorted(equidade, key=lambda n: equidade[n])
    mais_justa, segunda_justa = ordem_equidade[0], ordem_equidade[1]
    menos_justa = ordem_equidade[-1]
    valor_justa = report.mais_menos(equidade[mais_justa], disp_equidade[mais_justa])
    valor_injusta = report.mais_menos(equidade[menos_justa], disp_equidade[menos_justa])

    if _lidera(equidade, disp_equidade, menor_e_melhor=True) is not None:
        linhas.append(
            f"- **Equidade entre instituições:** {titulo(mais_justa)} obteve a menor "
            f"dispersão de acurácia entre clientes ({valor_justa}), contra "
            f"{report.mais_menos(equidade[segunda_justa], disp_equidade[segunda_justa])} "
            f"da segunda colocada e {valor_injusta} de {titulo(menos_justa)}. Sob dados "
            f"heterogêneos é esta a métrica que mais distingue as estratégias, já que a "
            f"acurácia global satura."
        )
    else:
        linhas.append(
            f"- **Equidade entre instituições:** a dispersão de acurácia entre clientes "
            f"ficou entre {valor_justa} ({titulo(mais_justa)}) e {valor_injusta} "
            f"({titulo(menos_justa)}), mas as faixas das primeiras colocadas se sobrepõem. "
            f"Há separação clara apenas entre os extremos, não uma vencedora isolada."
        )

    pior_cliente = {n: r["federated_accuracy_min"] for n, r in resumos.items()}
    disp_pior = {n: r["federated_accuracy_min_spread"] for n, r in resumos.items()}
    pior_pior = min(pior_cliente, key=lambda n: pior_cliente[n])
    lider_pior = _lidera(pior_cliente, disp_pior, menor_e_melhor=False)
    if lider_pior is not None:
        linhas.append(
            f"- **Instituição menos favorecida:** o cliente de pior desempenho foi melhor "
            f"atendido por {titulo(lider_pior)} "
            f"({report.mais_menos(pior_cliente[lider_pior], disp_pior[lider_pior])}) e "
            f"pior por {titulo(pior_pior)} "
            f"({report.mais_menos(pior_cliente[pior_pior], disp_pior[pior_pior])})."
        )
    else:
        melhor_pior = max(pior_cliente, key=lambda n: pior_cliente[n])
        linhas.append(
            f"- **Instituição menos favorecida:** o pior cliente teve acurácia entre "
            f"{report.mais_menos(pior_cliente[pior_pior], disp_pior[pior_pior])} "
            f"({titulo(pior_pior)}) e "
            f"{report.mais_menos(pior_cliente[melhor_pior], disp_pior[melhor_pior])} "
            f"({titulo(melhor_pior)}); nenhuma estratégia protege a instituição "
            f"minoritária de forma destacada das demais."
        )

    iid = {
        nome: metrics.resumo_repeticoes(execucoes[(nome, "iid")])
        for nome in strategies.NAMES
        if execucoes.get((nome, "iid"))
    }
    if len(iid) > 1:
        valores = {n: r["central_accuracy"] for n, r in iid.items()}
        amplitude = max(valores.values()) - min(valores.values())
        linhas.append(
            f"- **Controle IID:** no regime homogêneo a diferença de acurácia entre a "
            f"melhor e a pior estratégia é de {_num(amplitude)}, confirmando que o "
            f"pipeline é o mesmo para todas e que a heterogeneidade dos dados, não a "
            f"implementação, é a origem das diferenças observadas no regime non-IID."
        )

    return linhas


def render() -> str:
    execucoes = _indexar()
    baseline = _carregar_baseline()

    primeiro_grupo = next(iter(execucoes.values()), None)
    partes = ["# Comparativo das estratégias", ""]

    if primeiro_grupo is None:
        partes.append("Nenhuma execução registrada ainda.")
        partes.append("")
        return "\n".join(partes)

    referencia = primeiro_grupo[0]
    cfg = referencia["config"]
    spec = datasets.get_spec(referencia["dataset"])
    sementes = sorted({r["config"]["seed"] for grupo in execucoes.values() for r in grupo})

    partes.append(
        f"Dataset `{spec.name}` · {cfg['num-clients']} clientes · "
        f"{cfg['num-server-rounds']} rounds · {cfg['local-epochs']} épocas locais · "
        f"sementes {', '.join(str(s) for s in sementes)} · "
        f"dispositivo `{referencia['device']}`"
    )
    partes.append("")

    if len(sementes) > 1:
        partes.append(
            f"Cada célula traz a média de {len(sementes)} execuções independentes seguida "
            "da dispersão entre elas. A dispersão existe porque a agregação dos pesos no "
            "servidor não é reproduzível bit a bit: a ordem de chegada das respostas dos "
            "clientes varia, e diferenças da ordem de 1e-7 nos pesos se amplificam ao "
            "longo dos rounds. **Diferenças entre estratégias menores que a dispersão não "
            "são conclusivas.**"
        )
    else:
        partes.append(
            "Os valores vêm de uma única execução por configuração. A agregação dos pesos "
            "no servidor não é reproduzível bit a bit — a ordem de chegada das respostas "
            "dos clientes varia entre execuções — então diferenças pequenas entre "
            "estratégias podem refletir variação de execução. Repita com outras sementes "
            "(`--run-config seed=N`) para distinguir."
        )
    partes.append("")

    if baseline is not None:
        final = baseline["final"]
        partes.append("## Teto de referência")
        partes.append("")
        partes.append(
            "O mesmo modelo treinado de forma centralizada, com acesso a todos os dados de "
            "treino de uma só vez. Nenhuma estratégia federada deve superá-lo de forma "
            "significativa — se superasse, haveria vazamento de dados no pipeline."
        )
        partes.append("")
        partes.append("| | Acurácia | F1 | ROC-AUC | Treino efetivo (s) |")
        partes.append("|---|---|---|---|---|")
        partes.append(
            f"| Centralizado ({baseline['config']['epochs']} épocas) | "
            f"{_num(final['accuracy'])} | {_num(final['f1'])} | {_num(final['auc'])} | "
            f"{_num(baseline['timing']['train_effective_s'], 2)} |"
        )
        partes.append("")

    for particao in PARTICOES:
        partes.append(f"## {PARTICAO_TITULO[particao]}")
        partes.append("")
        partes.extend(_tabela(execucoes, particao))
        partes.append("")

    destaques = _destaques(execucoes)
    if destaques:
        partes.append("## Leitura dos resultados")
        partes.append("")
        partes.extend(destaques)
        partes.append("")

    skew = _skew(execucoes)
    if skew:
        partes.append("## Heterogeneidade da partição non-IID")
        partes.append("")
        partes.extend(skew)
        partes.append("")

    calibracoes = _calibracoes()
    if calibracoes:
        partes.append("## Calibração dos hiperparâmetros")
        partes.append("")
        partes.extend(calibracoes)

    faltando = [
        f"`{nome}` / `{particao}`"
        for nome in strategies.NAMES
        for particao in PARTICOES
        if not execucoes.get((nome, particao))
    ]
    if faltando:
        partes.append("## Execuções pendentes")
        partes.append("")
        partes.append(
            "As combinações a seguir ainda não foram executadas e aparecem como linha "
            "vazia nas tabelas acima:"
        )
        partes.append("")
        for item in faltando:
            partes.append(f"- {item}")
        partes.append("")

    partes.append("## Observações sobre o dataset")
    partes.append("")
    for nota in spec.notes:
        partes.append(f"- {nota}")
    partes.append("")

    partes.append("## Como os números foram obtidos")
    partes.append("")
    partes.append(
        "Acurácia, F1 e ROC-AUC são medidos pelo servidor sobre o conjunto de teste "
        f"({spec.name}, {int(datasets.TEST_SIZE * 100)}% do total) que nenhum cliente "
        "recebeu em nenhum momento. *Rounds p/ ≥95%* e *Rounds p/ ≥99%* são os primeiros "
        "rounds em que a acurácia central alcança cada alvo, e medem eficiência de "
        "comunicação: o alvo de 95% é frouxo neste dataset e serve de verificação, "
        "enquanto o de 99% fica logo abaixo do teto centralizado e é o que separa as "
        "estratégias. *Treino efetivo* soma o tempo gasto dentro do laço de otimização em "
        "todos os clientes e rounds, excluindo carga de dados e inicialização. *Desvio "
        "entre clientes* e *pior cliente* vêm da avaliação que cada cliente faz do modelo "
        "global na sua própria validação, e medem equidade: se o modelo serve todas as "
        "instituições ou apenas as que detêm mais dados."
    )
    partes.append("")

    return "\n".join(partes)


def regenerar_relatorios() -> list[Path]:
    """Reescreve o relatório de cada estratégia a partir dos JSONs gravados.

    Permite ajustar a apresentação dos resultados — colunas, alvos de
    convergência, texto — sem repetir as execuções.
    """
    gerados = []
    for nome in strategies.NAMES:
        if not report.load_runs(nome):
            continue
        destino = report.results_dir() / f"{nome.upper()}.md"
        destino.write_text(report.render_strategy(nome), encoding="utf-8")
        gerados.append(destino)
    return gerados


def main() -> None:
    destino: Path = report.results_dir() / "README.md"
    destino.parent.mkdir(parents=True, exist_ok=True)

    for caminho in regenerar_relatorios():
        print(f"Relatorio atualizado: {caminho.name}")

    destino.write_text(render(), encoding="utf-8")
    print(f"Comparativo gravado em {destino}")


if __name__ == "__main__":
    main()
