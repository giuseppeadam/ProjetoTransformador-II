"""Executa o benchmark dentro de um limite de tempo, priorizando o que importa.

Pensado para rodar sem supervisão: mede a primeira execução, estima quantas
cabem no tempo disponível, e executa em ordem de prioridade até o prazo. Para no
limite e consolida o que completou, em vez de ser interrompido no meio.

A ordem de prioridade coloca primeiro o regime onde está o resultado — a
partição heterogênea — e, dentro dele, a primeira semente de cada estratégia.
Assim, se o tempo acabar, o que ficou pronto é um benchmark completo com uma
semente, e não metade das estratégias com três.

Uso:
    python run_noturno.py --limite-horas 7 --dataset pisa --partitions natural iid --num-clients 48
    python run_noturno.py --limite-horas 2 --dry-run
"""

import argparse
import json
import time
from pathlib import Path

from fl_benchmark import report, strategies
from run_all import executar

AQUI = Path(__file__).resolve().parent

# Margem de segurança: só inicia uma execução se o tempo restante couber a
# duração estimada mais esta folga. Execuções variam, e ser interrompido no meio
# desperdiça todo o tempo já gasto naquela execução.
FOLGA_S = 180.0


def plano(estrategias: list[str], particoes: list[str], sementes: list[int]) -> list[tuple]:
    """Ordena as execuções por prioridade.

    Primeiro a semente inicial de todas as estratégias no regime principal,
    depois o regime de controle, e só então as sementes adicionais. Um corte por
    tempo em qualquer ponto deixa um conjunto interpretável.
    """
    ordenado = []
    for indice, semente in enumerate(sementes):
        for particao in particoes:
            for estrategia in estrategias:
                ordenado.append((estrategia, particao, semente, indice))
    # Prioriza por rodada de semente, mantendo a ordem de partições declarada.
    ordenado.sort(key=lambda t: (t[3], particoes.index(t[1])))
    return [(e, p, s) for e, p, s, _ in ordenado]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limite-horas", type=float, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--strategies", nargs="+", default=list(strategies.NAMES), choices=list(strategies.NAMES)
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["natural", "iid"],
        choices=["iid", "noniid", "natural"],
        help="em ordem de prioridade; a primeira é executada antes",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num-clients", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--federation", default="local-simulation")
    parser.add_argument("--log-dir", default=str(AQUI / "results" / "logs"))
    parser.add_argument(
        "--dry-run", action="store_true", help="mostra o plano e sai, sem executar"
    )
    args = parser.parse_args()

    limite_s = args.limite_horas * 3600.0
    fila = plano(args.strategies, args.partitions, args.seeds)
    log_dir = Path(args.log_dir)

    print(f"plano: {len(fila)} execucoes, limite de {args.limite_horas:.1f}h", flush=True)
    for posicao, (e, p, s) in enumerate(fila[:6], 1):
        print(f"  {posicao}. {e} / {p} / seed {s}", flush=True)
    if len(fila) > 6:
        print(f"  ... e {len(fila) - 6} outras", flush=True)

    if args.dry_run:
        return 0

    inicio = time.perf_counter()
    duracoes: list[float] = []
    concluidas, falhas, nao_iniciadas = [], [], []

    for estrategia, particao, semente in fila:
        decorrido = time.perf_counter() - inicio
        restante = limite_s - decorrido

        estimativa = (sum(duracoes) / len(duracoes)) if duracoes else 0.0
        if duracoes and restante < estimativa + FOLGA_S:
            nao_iniciadas.append((estrategia, particao, semente))
            continue

        ok, duracao = executar(
            estrategia,
            particao,
            federacao=args.federation,
            num_clients=args.num_clients,
            rounds=args.rounds,
            device=args.device,
            seed=semente,
            dataset=args.dataset,
            log_dir=log_dir,
        )
        duracoes.append(duracao)
        (concluidas if ok else falhas).append((estrategia, particao, semente))

        media = sum(duracoes) / len(duracoes)
        sobra = limite_s - (time.perf_counter() - inicio)
        cabem = max(0, int(sobra // (media + FOLGA_S)))
        print(
            f"    media {media / 60:.1f} min/execucao | restam {sobra / 3600:.1f}h "
            f"| cabem ~{cabem}",
            flush=True,
        )

    total = time.perf_counter() - inicio
    print()
    print(f"=== encerrado em {total / 3600:.2f}h de {args.limite_horas:.1f}h ===")
    print(f"  concluidas:    {len(concluidas)}")
    print(f"  falharam:      {len(falhas)}")
    print(f"  nao iniciadas: {len(nao_iniciadas)} (tempo insuficiente)")

    if falhas:
        print("  falhas: " + ", ".join(f"{e}/{p}/s{s}" for e, p, s in falhas))
    if nao_iniciadas:
        print("  pendentes: " + ", ".join(f"{e}/{p}/s{s}" for e, p, s in nao_iniciadas))

    # Registro do que ficou de fora, para que o relatório não sugira cobertura
    # que não existe e para retomar de onde parou.
    resumo = {
        "limite_horas": args.limite_horas,
        "duracao_horas": round(total / 3600, 3),
        "dataset": args.dataset,
        "media_s_por_execucao": round(sum(duracoes) / len(duracoes), 1) if duracoes else None,
        "concluidas": [list(c) for c in concluidas],
        "falhas": [list(f) for f in falhas],
        "nao_iniciadas": [list(n) for n in nao_iniciadas],
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "resumo_noturno.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, indent=2, ensure_ascii=False)

    print("\nconsolidando...", flush=True)
    import subprocess
    import sys

    from run_all import _ambiente

    subprocess.run([sys.executable, "make_summary.py"], cwd=AQUI, env=_ambiente(), check=False)
    print(f"\nresumo da noite em {log_dir / 'resumo_noturno.json'}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
