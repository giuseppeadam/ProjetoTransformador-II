"""Treina o modelo de forma centralizada, para servir de teto de referência.

Um benchmark federado sem referência centralizada não permite julgar os
números: 98% de acurácia é bom ou ruim? A resposta depende do que o mesmo
modelo alcança quando tem acesso a todos os dados de uma vez, sem a restrição
de nunca reuni-los.

O resultado também funciona como verificação de integridade: nenhuma estratégia
federada deve superar o centralizado de forma significativa. Se superar, há
vazamento de dados no pipeline.

Uso:
    python baseline.py
    python baseline.py --dataset productivity --epochs 30 --device cpu
"""

import argparse
import json
import time

import torch

from fl_benchmark import datasets, engine, report
from fl_benchmark.model import MLP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="productivity")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="épocas centralizadas; o padrão equivale a 10 rounds de 3 épocas locais",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    X_train, y_train, X_test, y_test = datasets.load(args.dataset)
    device = engine.resolve_device(args.device)

    modelo = MLP(datasets.num_features(args.dataset))
    treino = engine.make_loader(
        X_train, y_train, args.batch_size, shuffle=True, seed=args.seed
    )
    teste = engine.make_loader(X_test, y_test, args.eval_batch_size, shuffle=False)

    inicio = time.perf_counter()
    loss, tempo_treino = engine.train(
        modelo, treino, epochs=args.epochs, lr=args.learning_rate, device=device
    )
    wall_clock = time.perf_counter() - inicio

    resultado = engine.evaluate(modelo, teste, device)

    registro = {
        "kind": "centralized_baseline",
        "dataset": args.dataset,
        "device": engine.device_label(device),
        "config": {
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "learning-rate": args.learning_rate,
            "seed": args.seed,
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "final": {
            "accuracy": resultado["accuracy"],
            "f1": resultado["f1"],
            "auc": resultado["auc"],
            "loss": resultado["loss"],
            "train_loss": loss,
        },
        "timing": {
            "train_effective_s": tempo_treino,
            "wall_clock_s": wall_clock,
        },
    }

    destino = report.results_dir()
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / "baseline.json"
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(registro, arquivo, indent=2, ensure_ascii=False)

    print(
        f"Baseline centralizado ({args.epochs} épocas, {len(X_train)} amostras, "
        f"{engine.device_label(device)}):"
    )
    print(f"  acuracia  {resultado['accuracy']:.4f}")
    print(f"  F1        {resultado['f1']:.4f}")
    print(f"  ROC-AUC   {resultado['auc']:.4f}")
    print(f"  treino    {tempo_treino:.2f}s")
    print(f"\nGravado em {caminho}")


if __name__ == "__main__":
    main()
