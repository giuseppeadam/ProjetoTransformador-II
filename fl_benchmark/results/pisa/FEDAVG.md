# FedAvg

> A estratégia de referência de Federated Learning. A cada round o servidor calcula a média dos pesos recebidos, ponderada pelo número de exemplos de cada cliente — um cliente com dez vezes mais dados pesa dez vezes mais no modelo global. Simples e eficiente quando os dados são homogêneos, mas sofre com *client drift*: se as distribuições locais divergem, os modelos treinados localmente apontam para mínimos diferentes e a média entre eles pode não ser boa para nenhum dos clientes.
>
> **Referência:** McMahan et al., 2017 — arxiv.org/abs/1602.05629

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7184 ± 0,0006 | 0,7226 ± 0,0011 | 0,7922 ± 0,0004 | 1,0 | 2,3 ± 0,5 | 466,47 ± 41,85 s | 0,0143 ± 0,0002 | 0,6907 ± 0,0020 |
| natural (por país) | 3 | 0,7096 ± 0,0005 | 0,7058 ± 0,0028 | 0,7830 ± 0,0003 | 1,0 | não atingido | 383,39 ± 7,92 s | 0,0564 ± 0,0013 | 0,5847 ± 0,0090 |

Dataset `pisa` · 48 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,7190. Alvos de convergência: 95% do teto = 0,6830 · 99% do teto = 0,7118.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7067 ± 0,0009 | 0,7178 ± 0,0015 | 0,7788 ± 0,0004 | 0,7060 ± 0,0013 | 46,24 ± 4,18 s |
| 2 | 0,7126 ± 0,0008 | 0,7202 ± 0,0014 | 0,7853 ± 0,0002 | 0,7124 ± 0,0008 | 46,00 ± 5,26 s |
| 3 | 0,7159 ± 0,0007 | 0,7208 ± 0,0011 | 0,7889 ± 0,0000 | 0,7153 ± 0,0009 | 47,09 ± 3,40 s |
| 4 | 0,7169 ± 0,0006 | 0,7214 ± 0,0024 | 0,7905 ± 0,0001 | 0,7170 ± 0,0009 | 46,94 ± 4,56 s |
| 5 | 0,7173 ± 0,0003 | 0,7198 ± 0,0006 | 0,7911 ± 0,0000 | 0,7176 ± 0,0008 | 46,94 ± 3,62 s |
| 6 | 0,7177 ± 0,0004 | 0,7206 ± 0,0017 | 0,7916 ± 0,0002 | 0,7179 ± 0,0010 | 47,25 ± 4,01 s |
| 7 | 0,7180 ± 0,0004 | 0,7220 ± 0,0018 | 0,7917 ± 0,0003 | 0,7179 ± 0,0009 | 45,29 ± 4,33 s |
| 8 | 0,7182 ± 0,0005 | 0,7217 ± 0,0015 | 0,7919 ± 0,0003 | 0,7183 ± 0,0010 | 46,62 ± 5,08 s |
| 9 | 0,7182 ± 0,0002 | 0,7230 ± 0,0016 | 0,7921 ± 0,0003 | 0,7180 ± 0,0011 | 47,03 ± 3,60 s |
| 10 | 0,7184 ± 0,0006 | 0,7226 ± 0,0011 | 0,7922 ± 0,0004 | 0,7184 ± 0,0007 | 47,07 ± 4,13 s |

### Curva por round — partição natural (por país)

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6991 ± 0,0012 | 0,7033 ± 0,0016 | 0,7689 ± 0,0012 | 0,7010 ± 0,0012 | 38,49 ± 1,68 s |
| 2 | 0,7026 ± 0,0006 | 0,7086 ± 0,0026 | 0,7752 ± 0,0008 | 0,7048 ± 0,0013 | 38,32 ± 0,44 s |
| 3 | 0,7064 ± 0,0002 | 0,7086 ± 0,0031 | 0,7789 ± 0,0003 | 0,7071 ± 0,0016 | 39,60 ± 1,73 s |
| 4 | 0,7079 ± 0,0004 | 0,7085 ± 0,0019 | 0,7810 ± 0,0002 | 0,7096 ± 0,0025 | 38,30 ± 0,07 s |
| 5 | 0,7088 ± 0,0005 | 0,7100 ± 0,0024 | 0,7821 ± 0,0003 | 0,7102 ± 0,0025 | 38,19 ± 1,08 s |
| 6 | 0,7087 ± 0,0004 | 0,7075 ± 0,0017 | 0,7823 ± 0,0001 | 0,7106 ± 0,0027 | 37,31 ± 0,92 s |
| 7 | 0,7090 ± 0,0003 | 0,7056 ± 0,0014 | 0,7826 ± 0,0001 | 0,7110 ± 0,0023 | 39,27 ± 1,29 s |
| 8 | 0,7095 ± 0,0002 | 0,7053 ± 0,0023 | 0,7834 ± 0,0004 | 0,7116 ± 0,0026 | 38,20 ± 1,36 s |
| 9 | 0,7098 ± 0,0005 | 0,7064 ± 0,0016 | 0,7831 ± 0,0007 | 0,7117 ± 0,0029 | 37,65 ± 0,22 s |
| 10 | 0,7096 ± 0,0005 | 0,7058 ± 0,0028 | 0,7830 ± 0,0003 | 0,7116 ± 0,0024 | 38,07 ± 1,25 s |

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="iid" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="iid" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="iid" dataset="pisa" num-clients=48 seed=44'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="natural" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="natural" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="natural" dataset="pisa" num-clients=48 seed=44'
```
