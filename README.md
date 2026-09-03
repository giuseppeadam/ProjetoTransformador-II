# ProjetoTransformador

## Parte 2 — Federated Learning (Flower)

Continuação do projeto: a "parte 1" (`dataset.ipynb`) treinou modelos clássicos
(Bagging, Gradient Boosting, Decision Tree, Random Forest, SVM, KNN, Voting)
sobre o dataset de vício digital. A "parte 2", em `fl_addiction/`, simula
múltiplas **instituições/clientes** treinando localmente sobre pedaços do
mesmo dataset, usando **Federated Learning** com o framework **Flower (flwr)**.

### Por que trocar de modelo?

Federated Learning agrega os clientes fazendo alguma operação sobre os
**pesos** do modelo a cada round. Bagging/Gradient Boosting/Decision Tree não
têm um vetor de pesos contínuo pra promediar (são conjuntos de árvores) — por
isso o modelo virou uma **rede neural simples (MLP)** em PyTorch, que expõe
pesos (`nn.Linear`) prontos para essa agregação.

### Glossário rápido

- **Cliente (ClientApp)**: representa uma instituição. Recebe os pesos
  globais, treina localmente por algumas épocas na sua própria partição de
  dados, e devolve os pesos atualizados — nunca envia os dados em si.
- **Servidor (ServerApp)**: coordena os rounds e agrega os pesos recebidos
  de todos os clientes num novo modelo global, usando uma **estratégia**.
- **Round**: um ciclo completo de (enviar pesos → treinar localmente →
  agregar).
- **Estratégias implementadas** (trocar é só mudar `strategy` no
  `pyproject.toml`, ver o `if/elif` em `fl_addiction/fl_addiction/app.py`):
  - `fedavg` (padrão do MVP): média ponderada simples dos pesos.
  - `fedprox`: penaliza clientes que se afastam muito do modelo global —
    ajuda quando os dados entre instituições são muito diferentes (non-IID).
  - `fedadam` / `fedyogi`: em vez de só promediar, aplicam um passo de
    otimizador (Adam / Yogi) no servidor sobre a diferença agregada de pesos.
  - `scaffold` (citado na aula, **não implementado**): corrige o "client
    drift" com control-variates — exige mudanças também no cliente, não é
    built-in no Flower atual. Fica como próximo passo.

### Como rodar

flwr run . --federation-config num-supernodes=5

Isso simula 5 "instituições", roda `num-server-rounds` (padrão: 10) rounds de
FedAvg, e imprime a accuracy/loss do modelo global a cada round — avaliado
num conjunto de teste centralizado que nenhum cliente viu.

Para trocar de estratégia sem editar arquivo:

```bash
flwr run . --federation-config num-supernodes=5 --run-config strategy=\"fedprox\"
```
