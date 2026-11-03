# FedTrimmedAvg

> Estratégia de agregação robusta. Em vez de promediar todos os valores recebidos, ordena os clientes parâmetro por parâmetro e descarta uma fração `beta` das pontas antes de calcular a média. A motivação original é resistência a clientes defeituosos ou maliciosos, cujos pesos aparecem como valores extremos e seriam eliminados pelo corte.
>
> **Referência:** Yin et al., 2018 — arxiv.org/abs/1803.01498
>
> **Hiperparâmetros próprios:** `beta = 0.2`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7184 ± 0,0006 | 0,7227 ± 0,0023 | 0,7923 ± 0,0005 | 1,0 | 2,0 | 498,40 ± 13,93 s | 0,0139 ± 0,0005 | 0,6927 ± 0,0017 |
| non-IID | 3 | 0,9895 ± 0,0040 | 0,9894 ± 0,0041 | 0,9997 ± 0,0001 | 1,3 ± 0,5 | 5,0 ± 3,6 | 23,14 ± 1,60 s | 0,0119 ± 0,0029 | 0,9672 ± 0,0055 |
| natural (por país) | 3 | 0,7068 ± 0,0009 | 0,6905 ± 0,0034 | 0,7815 ± 0,0009 | 1,0 | não atingido | 381,31 ± 1,04 s | 0,0629 ± 0,0014 | 0,5694 ± 0,0084 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,9932. Alvos de convergência: 95% do teto = 0,9436 · 99% do teto = 0,9833.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7068 ± 0,0009 | 0,7172 ± 0,0014 | 0,7789 ± 0,0004 | 0,7064 ± 0,0011 | 48,78 ± 1,47 s |
| 2 | 0,7129 ± 0,0006 | 0,7204 ± 0,0009 | 0,7852 ± 0,0003 | 0,7123 ± 0,0008 | 49,50 ± 1,22 s |
| 3 | 0,7161 ± 0,0006 | 0,7206 ± 0,0016 | 0,7890 ± 0,0001 | 0,7155 ± 0,0007 | 49,58 ± 3,03 s |
| 4 | 0,7172 ± 0,0002 | 0,7215 ± 0,0017 | 0,7906 ± 0,0002 | 0,7167 ± 0,0006 | 49,53 ± 1,54 s |
| 5 | 0,7176 ± 0,0001 | 0,7199 ± 0,0007 | 0,7911 ± 0,0001 | 0,7178 ± 0,0009 | 51,76 ± 1,31 s |
| 6 | 0,7175 ± 0,0001 | 0,7209 ± 0,0014 | 0,7917 ± 0,0002 | 0,7179 ± 0,0004 | 49,70 ± 2,69 s |
| 7 | 0,7179 ± 0,0004 | 0,7224 ± 0,0015 | 0,7918 ± 0,0003 | 0,7184 ± 0,0008 | 49,86 ± 0,91 s |
| 8 | 0,7181 ± 0,0006 | 0,7208 ± 0,0015 | 0,7920 ± 0,0003 | 0,7190 ± 0,0008 | 49,87 ± 0,98 s |
| 9 | 0,7185 ± 0,0005 | 0,7234 ± 0,0017 | 0,7921 ± 0,0004 | 0,7183 ± 0,0011 | 50,45 ± 2,24 s |
| 10 | 0,7184 ± 0,0006 | 0,7227 ± 0,0023 | 0,7923 ± 0,0005 | 0,7179 ± 0,0007 | 49,38 ± 1,37 s |

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

### Curva por round — partição natural (por país)

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6924 ± 0,0018 | 0,6691 ± 0,0039 | 0,7678 ± 0,0010 | 0,6955 ± 0,0010 | 38,75 ± 0,66 s |
| 2 | 0,6985 ± 0,0009 | 0,6870 ± 0,0014 | 0,7719 ± 0,0005 | 0,7019 ± 0,0013 | 38,03 ± 0,73 s |
| 3 | 0,7021 ± 0,0008 | 0,6899 ± 0,0016 | 0,7755 ± 0,0005 | 0,7043 ± 0,0017 | 37,78 ± 0,55 s |
| 4 | 0,7033 ± 0,0007 | 0,6875 ± 0,0025 | 0,7780 ± 0,0007 | 0,7051 ± 0,0023 | 38,57 ± 0,53 s |
| 5 | 0,7044 ± 0,0001 | 0,6880 ± 0,0016 | 0,7795 ± 0,0004 | 0,7067 ± 0,0024 | 38,04 ± 0,70 s |
| 6 | 0,7051 ± 0,0007 | 0,6882 ± 0,0010 | 0,7800 ± 0,0007 | 0,7070 ± 0,0029 | 38,99 ± 1,23 s |
| 7 | 0,7052 ± 0,0000 | 0,6882 ± 0,0013 | 0,7804 ± 0,0010 | 0,7074 ± 0,0026 | 38,53 ± 0,20 s |
| 8 | 0,7059 ± 0,0006 | 0,6884 ± 0,0032 | 0,7809 ± 0,0013 | 0,7071 ± 0,0029 | 37,72 ± 0,83 s |
| 9 | 0,7066 ± 0,0008 | 0,6911 ± 0,0038 | 0,7812 ± 0,0012 | 0,7084 ± 0,0026 | 37,06 ± 0,98 s |
| 10 | 0,7068 ± 0,0009 | 0,6905 ± 0,0034 | 0,7815 ± 0,0009 | 0,7080 ± 0,0026 | 37,83 ± 0,25 s |

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
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="natural" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="natural" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="natural" seed=44'
```
