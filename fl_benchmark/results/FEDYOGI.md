# FedYogi

> Mesma ideia do FedAdam — otimizador adaptativo no servidor — trocando Adam por Yogi. A diferença está em como o momento de segunda ordem é atualizado: o Adam o faz de forma multiplicativa, o que pode fazer o tamanho efetivo do passo crescer rápido demais; o Yogi usa atualização aditiva, controlando esse crescimento. Na prática tende a ser mais estável que o FedAdam quando os updates dos clientes variam muito de magnitude entre rounds — o que é justamente o caso sob partição non-IID.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,9832 ± 0,0019 | 0,9833 ± 0,0018 | 0,9983 ± 0,0006 | 3,3 ± 0,5 | não atingido | 25,66 ± 0,55 s | 0,0036 ± 0,0004 | 0,9766 ± 0,0013 |
| non-IID | 3 | 0,9845 ± 0,0015 | 0,9845 ± 0,0014 | 0,9984 ± 0,0007 | 4,3 ± 0,5 | não atingido | 23,15 ± 0,43 s | 0,0075 ± 0,0017 | 0,9761 ± 0,0038 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,8466 ± 0,0101 | 0,8462 ± 0,0085 | 0,9345 ± 0,0127 | 0,8451 ± 0,0122 | 2,36 ± 0,08 s |
| 2 | 0,9083 ± 0,0052 | 0,9083 ± 0,0044 | 0,9616 ± 0,0059 | 0,9079 ± 0,0081 | 2,62 ± 0,09 s |
| 3 | 0,9496 ± 0,0019 | 0,9499 ± 0,0024 | 0,9867 ± 0,0019 | 0,9492 ± 0,0030 | 2,55 ± 0,10 s |
| 4 | 0,9714 ± 0,0049 | 0,9717 ± 0,0050 | 0,9963 ± 0,0021 | 0,9710 ± 0,0081 | 2,77 ± 0,23 s |
| 5 | 0,9695 ± 0,0030 | 0,9697 ± 0,0028 | 0,9957 ± 0,0015 | 0,9718 ± 0,0036 | 2,52 ± 0,06 s |
| 6 | 0,9742 ± 0,0013 | 0,9742 ± 0,0015 | 0,9976 ± 0,0004 | 0,9728 ± 0,0025 | 2,59 ± 0,07 s |
| 7 | 0,9841 ± 0,0024 | 0,9841 ± 0,0024 | 0,9991 ± 0,0000 | 0,9827 ± 0,0013 | 2,62 ± 0,19 s |
| 8 | 0,9836 ± 0,0025 | 0,9837 ± 0,0024 | 0,9980 ± 0,0014 | 0,9856 ± 0,0016 | 2,50 ± 0,06 s |
| 9 | 0,9799 ± 0,0035 | 0,9802 ± 0,0033 | 0,9967 ± 0,0010 | 0,9805 ± 0,0030 | 2,51 ± 0,09 s |
| 10 | 0,9832 ± 0,0019 | 0,9833 ± 0,0018 | 0,9983 ± 0,0006 | 0,9829 ± 0,0023 | 2,65 ± 0,10 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,8499 ± 0,0110 | 0,8546 ± 0,0041 | 0,9386 ± 0,0070 | 0,8494 ± 0,0109 | 2,31 ± 0,13 s |
| 2 | 0,9001 ± 0,0096 | 0,9017 ± 0,0089 | 0,9529 ± 0,0074 | 0,8975 ± 0,0143 | 2,47 ± 0,07 s |
| 3 | 0,9407 ± 0,0076 | 0,9402 ± 0,0089 | 0,9860 ± 0,0013 | 0,9434 ± 0,0095 | 2,37 ± 0,29 s |
| 4 | 0,9642 ± 0,0124 | 0,9636 ± 0,0134 | 0,9965 ± 0,0011 | 0,9625 ± 0,0159 | 2,44 ± 0,21 s |
| 5 | 0,9733 ± 0,0052 | 0,9730 ± 0,0057 | 0,9978 ± 0,0005 | 0,9711 ± 0,0094 | 2,32 ± 0,10 s |
| 6 | 0,8906 ± 0,1164 | 0,8618 ± 0,1577 | 0,9800 ± 0,0245 | 0,8933 ± 0,1096 | 2,16 ± 0,15 s |
| 7 | 0,9793 ± 0,0020 | 0,9795 ± 0,0020 | 0,9985 ± 0,0003 | 0,9804 ± 0,0037 | 2,15 ± 0,11 s |
| 8 | 0,9841 ± 0,0015 | 0,9840 ± 0,0015 | 0,9986 ± 0,0002 | 0,9829 ± 0,0032 | 2,49 ± 0,40 s |
| 9 | 0,9847 ± 0,0011 | 0,9846 ± 0,0011 | 0,9983 ± 0,0007 | 0,9814 ± 0,0021 | 2,13 ± 0,09 s |
| 10 | 0,9845 ± 0,0015 | 0,9845 ± 0,0014 | 0,9984 ± 0,0007 | 0,9810 ± 0,0020 | 2,33 ± 0,09 s |

## Comportamento do método

- `eta`, a taxa de aprendizado do servidor, foi calibrada por varredura na partição IID; o registro está em `calibration_eta_fedyogi.json`.
- A comparação mais informativa é com o FedAdam, de quem difere apenas na forma de atualizar o momento de segunda ordem. Sob partição non-IID o FedYogi apresentou dispersão de acurácia entre clientes menor que a do FedAdam e atendeu melhor o cliente de pior desempenho, o que é consistente com a atualização aditiva: ela limita o crescimento do passo efetivo quando os updates dos clientes variam muito de magnitude, situação que a heterogeneidade dos dados provoca.
- Como o FedAdam, não alcança 99% de acurácia em 10 rounds neste dataset: a média simples do FedAvg já é adequada para um problema linearmente separável, e o passo adaptativo no servidor adiciona inércia sem benefício correspondente. A vantagem dessa família aparece quando a agregação simples é ruim, não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" seed=44'
```
