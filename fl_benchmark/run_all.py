"""Executa o benchmark completo, uma estratégia por vez.

Cada combinação de estratégia e regime de partição é executada em um processo
`flwr run` independente, em sequência — não em paralelo. Isso é deliberado:
*tempo de treino* é uma das métricas comparadas, e execuções concorrentes
disputariam CPU entre si, contaminando a medição.

As estratégias também podem ser executadas individualmente, sem este script;
veja o README da raiz do repositório. Este arquivo apenas automatiza a
sequência completa.

Uso:
    python run_all.py
    python run_all.py --strategies fedavg fedprox --partitions noniid
    python run_all.py --rounds 20 --device cuda --federation local-gpu
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from fl_benchmark import report, strategies

AQUI = Path(__file__).resolve().parent
PARTICOES = ("iid", "noniid")

# Histórico do tempo de parede de cada processo `flwr run`. O JSON de resultado
# só registra a duração a partir do `strategy.start()`, e fica de fora o custo
# de construir o pacote da aplicação, criar o ambiente e subir o backend de
# simulação — que na prática responde pela maior parte do tempo em execuções
# pequenas. Sem este registro não há como estimar a duração de um lote.
HISTORICO = "tempos_execucao.csv"
COLUNAS_HISTORICO = (
    "estrategia",
    "particao",
    "semente",
    "duracao_processo_s",
    "sucesso",
    "num_clients",
    "rounds",
    "dataset",
)


def _dataset_padrao() -> str:
    """Dataset configurado no pyproject.toml, usado quando --dataset é omitido."""
    with (AQUI / "pyproject.toml").open("rb") as arquivo:
        return str(tomllib.load(arquivo)["tool"]["flwr"]["app"]["config"]["dataset"])


def registrar_tempo(log_dir: Path, linha: dict) -> None:
    """Acrescenta uma linha ao histórico de tempos, criando o cabeçalho se preciso."""
    destino = log_dir / HISTORICO
    novo = not destino.exists()
    log_dir.mkdir(parents=True, exist_ok=True)
    with destino.open("a", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(COLUNAS_HISTORICO))
        if novo:
            escritor.writeheader()
        escritor.writerow({c: linha.get(c, "") for c in COLUNAS_HISTORICO})


def _ambiente() -> dict:
    """Ambiente do subprocesso, com saída em UTF-8.

    O console do Windows usa uma codificação que não cobre todos os caracteres
    emitidos pelo Flower, e a execução falharia na primeira linha de log.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def executar(
    estrategia: str,
    particao: str,
    *,
    federacao: str,
    num_clients: int,
    rounds: int | None,
    device: str | None,
    seed: int,
    dataset: str | None,
    log_dir: Path,
) -> tuple[bool, float]:
    """Executa uma combinação e devolve `(sucesso, duração)`.

    O sucesso é julgado pelo artefato produzido, não pelo código de saída. O
    `flwr run` pode encerrar com código 0 mesmo quando o backend de simulação
    falha na inicialização — já observado com timeout do raylet — e confiar no
    código de saída faria uma execução perdida ser contada como concluída.
    """
    # `num-clients` precisa acompanhar `num-supernodes`: o primeiro é lido pelo
    # servidor para montar a partição, o segundo dimensiona a federação. Passar
    # só um dos dois faz a partição discordar do número real de clientes.
    run_config = [
        f'strategy="{estrategia}"',
        f'partition="{particao}"',
        f"seed={seed}",
        f"num-clients={num_clients}",
    ]
    if rounds is not None:
        run_config.append(f"num-server-rounds={rounds}")
    if device is not None:
        run_config.append(f'device="{device}"')
    if dataset is not None:
        run_config.append(f'dataset="{dataset}"')

    comando = [
        "flwr",
        "run",
        ".",
        federacao,
        "--federation-config",
        f"num-supernodes={num_clients}",
        "--run-config",
        " ".join(run_config),
        "--stream",
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    destino_log = log_dir / f"{estrategia}_{particao}_seed{seed}.log"

    # O servidor grava em `results/<dataset>/` (ver `report.dataset_dir`).
    # Conferir em `results/` acusaria falha mesmo com a execução concluída.
    artefato = (
        report.dataset_dir(dataset or _dataset_padrao())
        / f"{estrategia}_{particao}_seed{seed}.json"
    )
    marca_anterior = artefato.stat().st_mtime if artefato.is_file() else None

    print(f"[{estrategia} / {particao} / seed {seed}] iniciando", flush=True)
    inicio = time.perf_counter()
    with destino_log.open("w", encoding="utf-8") as log:
        processo = subprocess.run(
            comando,
            cwd=AQUI,
            env=_ambiente(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    duracao = time.perf_counter() - inicio

    gravou = artefato.is_file() and artefato.stat().st_mtime != marca_anterior
    ok = processo.returncode == 0 and gravou

    if ok:
        estado = "concluido"
    elif processo.returncode != 0:
        estado = f"FALHOU (codigo {processo.returncode})"
    else:
        estado = "FALHOU (nenhum resultado gravado)"

    registrar_tempo(
        log_dir,
        {
            "estrategia": estrategia,
            "particao": particao,
            "semente": seed,
            "duracao_processo_s": round(duracao, 2),
            "sucesso": int(ok),
            "num_clients": num_clients,
            "rounds": rounds if rounds is not None else "",
            "dataset": dataset or "",
        },
    )

    print(
        f"[{estrategia} / {particao} / seed {seed}] {estado} em {duracao:.1f}s "
        f"-> {destino_log.name}",
        flush=True,
    )
    return ok, duracao


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(strategies.NAMES),
        choices=list(strategies.NAMES),
        help="estratégias a executar (padrão: todas)",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=list(PARTICOES),
        choices=list(PARTICOES),
        help="regimes de partição a executar (padrão: ambos)",
    )
    parser.add_argument("--federation", default="local-simulation")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=None, help="sobrescreve num-server-rounds")
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dataset", default=None, help="sobrescreve o dataset configurado")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help=(
            "sementes a executar. Mais de uma permite reportar media e dispersao, "
            "necessario porque a agregacao dos pesos no servidor nao e reproduzivel "
            "bit a bit (padrao: 42 43 44)"
        ),
    )
    parser.add_argument("--log-dir", default=str(AQUI / "results" / "logs"))
    args = parser.parse_args()

    combinacoes = [
        (e, p, s) for e in args.strategies for p in args.partitions for s in args.seeds
    ]
    print(f"{len(combinacoes)} execucoes em sequencia\n", flush=True)

    falhas = []
    inicio_total = time.perf_counter()
    for estrategia, particao, semente in combinacoes:
        ok, _ = executar(
            estrategia,
            particao,
            federacao=args.federation,
            num_clients=args.num_clients,
            rounds=args.rounds,
            device=args.device,
            seed=semente,
            dataset=args.dataset,
            log_dir=Path(args.log_dir),
        )
        if not ok:
            falhas.append(f"{estrategia}/{particao}/seed{semente}")

    total = time.perf_counter() - inicio_total
    print(f"\n{len(combinacoes) - len(falhas)}/{len(combinacoes)} concluidas em {total / 60:.1f} min")

    if falhas:
        print("Falharam: " + ", ".join(falhas))
        print(f"Consulte os logs em {args.log_dir}")

    print("\nConsolidando comparativo...", flush=True)
    subprocess.run([sys.executable, "make_summary.py"], cwd=AQUI, env=_ambiente(), check=False)

    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
