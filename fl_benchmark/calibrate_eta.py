"""Calibra a taxa de aprendizado do servidor para FedAdam e FedYogi.

FedAdam e FedYogi aplicam um passo de otimizador no servidor, controlado por
`eta`. FedAvg não tem passo no servidor — apenas promedia — e portanto não tem
esse hiperparâmetro. Comparar as três com um `eta` arbitrário mediria a
qualidade do ajuste, não a do método: um `eta` pequeno demais faz a estratégia
adaptativa convergir devagar e parecer inferior ao FedAvg por um motivo que não
tem relação com a ideia do método.

Este script varre valores de `eta` na partição IID e registra a acurácia final
e os rounds necessários para atingir cada alvo. O valor escolhido é então usado
nas execuções definitivas.

A partição IID é usada para a varredura justamente porque é o cenário de
controle: calibrar no regime non-IID, onde a comparação de fato acontece,
enviesaria o resultado a favor das adaptativas.

Uso:
    python calibrate_eta.py --strategy fedadam
    python calibrate_eta.py --strategy fedyogi --etas 0.05 0.1 0.3
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from fl_benchmark import metrics, report

AQUI = Path(__file__).resolve().parent
ETAS_PADRAO = (0.01, 0.1, 0.3, 0.6, 1.0)

# Semente reservada à calibração. Ter uma semente própria impede que a varredura
# sobrescreva um resultado definitivo do benchmark, que usa 42, 43 e 44.
SEED_CALIBRACAO = 1000


def _ambiente() -> dict:
    """Ambiente do subprocesso, com saída em UTF-8.

    O console do Windows usa uma codificação que não cobre todos os caracteres
    emitidos pelo Flower, e a execução falharia na primeira linha de log.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def executar(estrategia: str, eta: float, num_clients: int, destino_dir: Path) -> dict | None:
    """Executa uma combinação de estratégia e `eta`, devolvendo o resultado lido.

    O artefato gravado é movido de `results/` para `destino_dir`, de modo que a
    varredura não deixe execuções intermediárias entre os resultados definitivos
    do benchmark — elas apareceriam no comparativo como se fossem repetições.
    """
    comando = [
        "flwr",
        "run",
        ".",
        "--federation-config",
        f"num-supernodes={num_clients}",
        "--run-config",
        f'strategy="{estrategia}" partition="iid" eta={eta} seed={SEED_CALIBRACAO}',
        "--stream",
    ]

    destino_dir.mkdir(parents=True, exist_ok=True)
    destino_log = destino_dir / f"calib_{estrategia}_eta{eta}.log"

    artefato = report.results_dir() / f"{estrategia}_iid_seed{SEED_CALIBRACAO}.json"
    marca_anterior = artefato.stat().st_mtime if artefato.is_file() else None

    print(f"[{estrategia} eta={eta}] iniciando", flush=True)
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

    # O `flwr run` pode encerrar com código 0 mesmo quando o backend de
    # simulação falha na inicialização, então o sucesso é julgado pelo artefato.
    gravou = artefato.is_file() and artefato.stat().st_mtime != marca_anterior
    if processo.returncode != 0 or not gravou:
        motivo = (
            f"codigo {processo.returncode}" if processo.returncode != 0 else "sem resultado"
        )
        print(f"[{estrategia} eta={eta}] FALHOU ({motivo}) em {duracao:.1f}s", flush=True)
        return None

    with artefato.open(encoding="utf-8") as arquivo:
        run = json.load(arquivo)

    preservado = destino_dir / f"sweep_{estrategia}_eta{eta}.json"
    shutil.move(str(artefato), str(preservado))

    alvos = metrics.rounds_por_alvo(run)
    resultado = {
        "eta": eta,
        "final_accuracy": run["final"]["central_accuracy"],
        "final_f1": run["final"]["central_f1"],
        "final_auc": run["final"]["central_auc"],
        "rounds_to_95": alvos[0.95],
        "rounds_to_99": alvos[0.99],
        "accuracy_curve": {
            str(rodada): round(valor, 4) for rodada, valor in metrics.curva_central(run).items()
        },
    }
    print(
        f"[{estrategia} eta={eta}] acuracia {resultado['final_accuracy']:.4f} | "
        f"rounds p/ 99% {resultado['rounds_to_99']} | {duracao:.1f}s",
        flush=True,
    )
    return resultado


def escolher(resultados: list[dict]) -> dict:
    """Escolhe o melhor `eta`: menos rounds até 99%, desempatando por acurácia.

    A prioridade é a eficiência de comunicação, porque é o que a família
    adaptativa promete melhorar. Quando nenhum valor alcança o alvo — o que
    acontece neste dataset — a decisão recai inteiramente sobre a acurácia final.
    """

    def chave(r: dict):
        rounds = r["rounds_to_99"] if r["rounds_to_99"] is not None else float("inf")
        return (rounds, -r["final_accuracy"])

    return min(resultados, key=chave)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="fedadam", choices=("fedadam", "fedyogi"))
    parser.add_argument("--etas", nargs="+", type=float, default=list(ETAS_PADRAO))
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument(
        "--sweep-dir",
        default=str(AQUI / "results" / "logs" / "calibracao"),
        help="onde guardar as execuções da varredura, fora dos resultados definitivos",
    )
    args = parser.parse_args()

    destino_dir = Path(args.sweep_dir)
    resultados = []
    for eta in args.etas:
        resultado = executar(args.strategy, eta, args.num_clients, destino_dir)
        if resultado is not None:
            resultados.append(resultado)

    if not resultados:
        print("nenhuma execucao concluiu; consulte os logs em " + str(destino_dir))
        return 1

    if len(resultados) < len(args.etas):
        faltaram = len(args.etas) - len(resultados)
        print(f"\nAtencao: {faltaram} de {len(args.etas)} valores de eta nao concluiram.")

    resultados.sort(key=lambda r: r["eta"])
    melhor = escolher(resultados)

    print(f"\nVarredura de {args.strategy}:")
    for r in resultados:
        marca = " <- escolhido" if r["eta"] == melhor["eta"] else ""
        alvo = r["rounds_to_99"] if r["rounds_to_99"] is not None else "nao atingido"
        print(f"  eta={r['eta']:<6} acuracia={r['final_accuracy']:.4f}  rounds p/ 99%={alvo}{marca}")

    destino = report.results_dir() / f"calibration_eta_{args.strategy}.json"
    registro = {
        "strategy": args.strategy,
        "partition_used": "iid",
        "seed": SEED_CALIBRACAO,
        "chosen_eta": melhor["eta"],
        "criterion": (
            "menos rounds ate 99% de acuracia, com desempate pela acuracia final. "
            "Quando nenhum valor alcanca o alvo, decide a acuracia final"
        ),
        "sweep": resultados,
    }
    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(registro, arquivo, indent=2, ensure_ascii=False)

    print(f"\nRegistro gravado em {destino}")
    print(
        f"Para usar o valor escolhido como padrao, ajuste `eta` em pyproject.toml "
        f"para {melhor['eta']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
