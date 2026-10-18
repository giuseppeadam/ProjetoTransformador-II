"""Modelo compartilhado por todas as estratégias do benchmark.

Usar a mesma arquitetura em todas as execuções é o que torna a comparação
justa: a diferença medida entre estratégias vem da forma de agregar os pesos,
não da capacidade do modelo.
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Perceptron multicamadas para classificação binária tabular.

    Federated Learning agrega os clientes operando sobre os pesos do modelo a
    cada round. Modelos baseados em árvores (Random Forest, Gradient Boosting)
    não expõem um vetor de pesos contínuo que possa ser promediado — por isso o
    benchmark usa uma rede neural, cujas camadas `nn.Linear` oferecem
    exatamente isso.

    A saída é um logit único (sem sigmoid), consumido por `BCEWithLogitsLoss`,
    que é numericamente mais estável do que aplicar sigmoid e BCE em separado.
    """

    def __init__(self, num_features: int, hidden: tuple[int, int] = (32, 16)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(num_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
