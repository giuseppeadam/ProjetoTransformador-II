# FedTrimmedAvg

> Estratégia de agregação robusta. Em vez de promediar todos os valores recebidos, ordena os clientes parâmetro por parâmetro e descarta uma fração `beta` das pontas antes de calcular a média. A motivação original é resistência a clientes defeituosos ou maliciosos, cujos pesos aparecem como valores extremos e seriam eliminados pelo corte.
>
> **Referência:** Yin et al., 2018 — arxiv.org/abs/1803.01498
>
> **Hiperparâmetros próprios:** `beta = 0.2`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,9913 ± 0,0006 | 0,9913 ± 0,0006 | 0,9998 ± 0,0000 | 1,0 | 2,7 ± 0,5 | 25,33 ± 0,70 s | 0,0028 ± 0,0003 | 0,9896 ± 0,0007 |
| non-IID | 3 | 0,9895 ± 0,0040 | 0,9894 ± 0,0041 | 0,9997 ± 0,0001 | 1,7 ± 0,5 | 5,5 (em 2/3) | 23,14 ± 1,60 s | 0,0119 ± 0,0029 | 0,9672 ± 0,0055 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,9861 ± 0,0022 | 0,9861 ± 0,0022 | 0,9994 ± 0,0003 | 0,9850 ± 0,0051 | 2,41 ± 0,03 s |
| 2 | 0,9897 ± 0,0019 | 0,9897 ± 0,0019 | 0,9997 ± 0,0001 | 0,9889 ± 0,0010 | 2,47 ± 0,01 s |
| 3 | 0,9903 ± 0,0002 | 0,9904 ± 0,0002 | 0,9997 ± 0,0001 | 0,9908 ± 0,0004 | 2,53 ± 0,10 s |
| 4 | 0,9902 ± 0,0005 | 0,9902 ± 0,0004 | 0,9997 ± 0,0000 | 0,9902 ± 0,0016 | 2,71 ± 0,16 s |
| 5 | 0,9913 ± 0,0011 | 0,9913 ± 0,0011 | 0,9997 ± 0,0000 | 0,9917 ± 0,0011 | 2,55 ± 0,17 s |
| 6 | 0,9912 ± 0,0014 | 0,9912 ± 0,0014 | 0,9998 ± 0,0000 | 0,9909 ± 0,0009 | 2,57 ± 0,13 s |
| 7 | 0,9910 ± 0,0029 | 0,9910 ± 0,0028 | 0,9998 ± 0,0000 | 0,9910 ± 0,0015 | 2,55 ± 0,14 s |
| 8 | 0,9895 ± 0,0019 | 0,9895 ± 0,0019 | 0,9997 ± 0,0000 | 0,9909 ± 0,0014 | 2,57 ± 0,09 s |
| 9 | 0,9918 ± 0,0006 | 0,9918 ± 0,0006 | 0,9998 ± 0,0000 | 0,9915 ± 0,0001 | 2,44 ± 0,07 s |
| 10 | 0,9913 ± 0,0006 | 0,9913 ± 0,0006 | 0,9998 ± 0,0000 | 0,9939 ± 0,0005 | 2,52 ± 0,17 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,8555 ± 0,1484 | 0,7918 ± 0,2401 | 0,9947 ± 0,0042 | 0,8574 ± 0,1555 | 2,20 ± 0,11 s |
| 2 | 0,9795 ± 0,0070 | 0,9793 ± 0,0074 | 0,9994 ± 0,0002 | 0,9789 ± 0,0084 | 2,49 ± 0,24 s |
| 3 | 0,9825 ± 0,0040 | 0,9825 ± 0,0043 | 0,9995 ± 0,0001 | 0,9813 ± 0,0070 | 2,56 ± 0,39 s |
| 4 | 0,9844 ± 0,0068 | 0,9843 ± 0,0071 | 0,9995 ± 0,0004 | 0,9831 ± 0,0106 | 2,36 ± 0,41 s |
| 5 | 0,9823 ± 0,0026 | 0,9823 ± 0,0026 | 0,9997 ± 0,0001 | 0,9819 ± 0,0063 | 2,27 ± 0,17 s |
| 6 | 0,9862 ± 0,0053 | 0,9861 ± 0,0055 | 0,9997 ± 0,0001 | 0,9842 ± 0,0065 | 2,27 ± 0,12 s |
| 7 | 0,9832 ± 0,0108 | 0,9829 ± 0,0113 | 0,9997 ± 0,0001 | 0,9806 ± 0,0131 | 2,31 ± 0,29 s |
| 8 | 0,9836 ± 0,0066 | 0,9834 ± 0,0070 | 0,9997 ± 0,0002 | 0,9806 ± 0,0119 | 2,20 ± 0,07 s |
| 9 | 0,9849 ± 0,0049 | 0,9849 ± 0,0051 | 0,9997 ± 0,0001 | 0,9835 ± 0,0077 | 2,15 ± 0,23 s |
| 10 | 0,9895 ± 0,0040 | 0,9894 ± 0,0041 | 0,9997 ± 0,0001 | 0,9852 ± 0,0066 | 2,33 ± 0,28 s |

## Comportamento do método

- O corte é alheio ao motivo pelo qual um cliente é extremo. Sob partição non-IID, o cliente cujos dados são os mais atípicos é legítimo e, ainda assim, é exatamente o que tende a ser descartado — o que pode piorar o resultado em vez de protegê-lo.
- Com 5 clientes e `beta = 0,2`, o corte remove 1 cliente de cada extremo e agrega apenas os 3 centrais, descartando 40% dos participantes do round.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" seed=44'
```
