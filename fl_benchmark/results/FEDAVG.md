# FedAvg

> A estratégia de referência de Federated Learning. A cada round o servidor calcula a média dos pesos recebidos, ponderada pelo número de exemplos de cada cliente — um cliente com dez vezes mais dados pesa dez vezes mais no modelo global. Simples e eficiente quando os dados são homogêneos, mas sofre com *client drift*: se as distribuições locais divergem, os modelos treinados localmente apontam para mínimos diferentes e a média entre eles pode não ser boa para nenhum dos clientes.
>
> **Referência:** McMahan et al., 2017 — arxiv.org/abs/1602.05629

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,9914 ± 0,0001 | 0,9914 ± 0,0001 | 0,9998 ± 0,0000 | 1,0 | 2,7 ± 0,5 | 25,15 ± 0,21 s | 0,0028 ± 0,0002 | 0,9885 ± 0,0007 |
| non-IID | 3 | 0,9884 ± 0,0001 | 0,9884 ± 0,0001 | 0,9997 ± 0,0000 | 1,0 | 4,5 (em 2/3) | 25,41 ± 1,76 s | 0,0062 ± 0,0018 | 0,9837 ± 0,0029 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,9861 ± 0,0018 | 0,9861 ± 0,0018 | 0,9995 ± 0,0002 | 0,9864 ± 0,0033 | 2,29 ± 0,03 s |
| 2 | 0,9905 ± 0,0030 | 0,9905 ± 0,0030 | 0,9997 ± 0,0001 | 0,9896 ± 0,0027 | 2,63 ± 0,12 s |
| 3 | 0,9905 ± 0,0025 | 0,9905 ± 0,0025 | 0,9997 ± 0,0001 | 0,9906 ± 0,0010 | 2,52 ± 0,05 s |
| 4 | 0,9888 ± 0,0014 | 0,9889 ± 0,0013 | 0,9998 ± 0,0000 | 0,9899 ± 0,0021 | 2,56 ± 0,06 s |
| 5 | 0,9925 ± 0,0014 | 0,9925 ± 0,0014 | 0,9998 ± 0,0000 | 0,9903 ± 0,0009 | 2,67 ± 0,14 s |
| 6 | 0,9907 ± 0,0009 | 0,9908 ± 0,0009 | 0,9998 ± 0,0001 | 0,9915 ± 0,0015 | 2,50 ± 0,05 s |
| 7 | 0,9909 ± 0,0017 | 0,9909 ± 0,0017 | 0,9998 ± 0,0000 | 0,9905 ± 0,0008 | 2,45 ± 0,07 s |
| 8 | 0,9905 ± 0,0016 | 0,9905 ± 0,0016 | 0,9998 ± 0,0000 | 0,9911 ± 0,0013 | 2,57 ± 0,09 s |
| 9 | 0,9917 ± 0,0008 | 0,9917 ± 0,0008 | 0,9998 ± 0,0000 | 0,9896 ± 0,0013 | 2,42 ± 0,07 s |
| 10 | 0,9914 ± 0,0001 | 0,9914 ± 0,0001 | 0,9998 ± 0,0000 | 0,9922 ± 0,0013 | 2,55 ± 0,17 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,9762 ± 0,0089 | 0,9760 ± 0,0094 | 0,9987 ± 0,0010 | 0,9772 ± 0,0128 | 2,13 ± 0,13 s |
| 2 | 0,9878 ± 0,0022 | 0,9878 ± 0,0023 | 0,9997 ± 0,0000 | 0,9875 ± 0,0031 | 2,59 ± 0,37 s |
| 3 | 0,9880 ± 0,0013 | 0,9880 ± 0,0014 | 0,9997 ± 0,0001 | 0,9895 ± 0,0041 | 2,60 ± 0,20 s |
| 4 | 0,9885 ± 0,0034 | 0,9885 ± 0,0034 | 0,9997 ± 0,0002 | 0,9872 ± 0,0059 | 2,53 ± 0,40 s |
| 5 | 0,9897 ± 0,0033 | 0,9897 ± 0,0033 | 0,9998 ± 0,0001 | 0,9909 ± 0,0009 | 2,52 ± 0,40 s |
| 6 | 0,9904 ± 0,0026 | 0,9904 ± 0,0026 | 0,9998 ± 0,0000 | 0,9883 ± 0,0047 | 2,55 ± 0,12 s |
| 7 | 0,9890 ± 0,0013 | 0,9890 ± 0,0014 | 0,9997 ± 0,0000 | 0,9888 ± 0,0029 | 2,63 ± 0,17 s |
| 8 | 0,9882 ± 0,0001 | 0,9882 ± 0,0002 | 0,9997 ± 0,0001 | 0,9897 ± 0,0031 | 2,86 ± 0,19 s |
| 9 | 0,9891 ± 0,0030 | 0,9891 ± 0,0030 | 0,9997 ± 0,0002 | 0,9897 ± 0,0028 | 2,68 ± 0,16 s |
| 10 | 0,9884 ± 0,0001 | 0,9884 ± 0,0001 | 0,9997 ± 0,0000 | 0,9898 ± 0,0023 | 2,33 ± 0,07 s |

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" seed=44'
```
