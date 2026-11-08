# FedTrimmedAvg

> Estratégia de agregação robusta. Em vez de promediar todos os valores recebidos, ordena os clientes parâmetro por parâmetro e descarta uma fração `beta` das pontas antes de calcular a média. A motivação original é resistência a clientes defeituosos ou maliciosos, cujos pesos aparecem como valores extremos e seriam eliminados pelo corte.
>
> **Referência:** Yin et al., 2018 — arxiv.org/abs/1803.01498
>
> **Hiperparâmetros próprios:** `beta = 0.2`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,8649 ± 0,0110 | 0,8867 ± 0,0133 | 0,9175 ± 0,0179 | 6,0 ± 0,8 | 7,5 (em 2/3) | 1,37 ± 0,56 s | 0,0875 ± 0,0295 | 0,6944 ± 0,0393 |
| non-IID | 3 | 0,8198 ± 0,0169 | 0,8583 ± 0,0114 | 0,8792 ± 0,0227 | 7,0 (em 2/3) | 5,0 (em 1/3) | 1,67 ± 0,08 s | 0,1347 ± 0,0450 | 0,5698 ± 0,1205 |

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,8649. Alvos de convergência: 95% do teto = 0,8216 · 99% do teto = 0,8562.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7613 ± 0,0169 | 0,7889 ± 0,0139 | 0,8797 ± 0,0132 | 0,8556 ± 0,0416 | 0,61 ± 0,48 s |
| 2 | 0,7703 ± 0,0110 | 0,7847 ± 0,0126 | 0,8845 ± 0,0067 | 0,8333 ± 0,0471 | 0,10 ± 0,03 s |
| 3 | 0,7838 | 0,8110 ± 0,0021 | 0,8927 ± 0,0070 | 0,8222 ± 0,0567 | 0,09 ± 0,02 s |
| 4 | 0,7928 ± 0,0064 | 0,8203 ± 0,0045 | 0,8966 ± 0,0066 | 0,8222 ± 0,0437 | 0,08 ± 0,02 s |
| 5 | 0,8153 ± 0,0064 | 0,8428 ± 0,0079 | 0,9004 ± 0,0096 | 0,8278 ± 0,0478 | 0,12 ± 0,06 s |
| 6 | 0,8333 ± 0,0169 | 0,8565 ± 0,0187 | 0,9063 ± 0,0100 | 0,8444 ± 0,0393 | 0,11 ± 0,07 s |
| 7 | 0,8468 ± 0,0169 | 0,8687 ± 0,0180 | 0,9114 ± 0,0122 | 0,8444 ± 0,0393 | 0,08 ± 0,01 s |
| 8 | 0,8604 ± 0,0064 | 0,8819 ± 0,0086 | 0,9137 ± 0,0148 | 0,8278 ± 0,0283 | 0,06 ± 0,00 s |
| 9 | 0,8468 ± 0,0064 | 0,8701 ± 0,0076 | 0,9149 ± 0,0164 | 0,8222 ± 0,0342 | 0,07 ± 0,01 s |
| 10 | 0,8649 ± 0,0110 | 0,8867 ± 0,0133 | 0,9175 ± 0,0179 | 0,8000 ± 0,0236 | 0,06 ± 0,00 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7613 ± 0,0127 | 0,8153 ± 0,0151 | 0,8705 ± 0,0205 | 0,7760 ± 0,0430 | 1,03 ± 0,03 s |
| 2 | 0,7748 ± 0,0230 | 0,8118 ± 0,0290 | 0,8802 ± 0,0130 | 0,7760 ± 0,0204 | 0,07 ± 0,00 s |
| 3 | 0,7838 ± 0,0221 | 0,8172 ± 0,0314 | 0,8830 ± 0,0096 | 0,7760 ± 0,0204 | 0,06 ± 0,00 s |
| 4 | 0,8198 ± 0,0230 | 0,8526 ± 0,0211 | 0,8822 ± 0,0097 | 0,7814 ± 0,0279 | 0,07 ± 0,01 s |
| 5 | 0,8153 ± 0,0387 | 0,8475 ± 0,0374 | 0,8787 ± 0,0147 | 0,7978 ± 0,0279 | 0,09 ± 0,02 s |
| 6 | 0,8153 ± 0,0355 | 0,8483 ± 0,0304 | 0,8792 ± 0,0143 | 0,7978 ± 0,0279 | 0,07 ± 0,01 s |
| 7 | 0,8198 ± 0,0337 | 0,8521 ± 0,0316 | 0,8817 ± 0,0138 | 0,7923 ± 0,0337 | 0,06 ± 0,00 s |
| 8 | 0,7973 ± 0,0191 | 0,8412 ± 0,0079 | 0,8822 ± 0,0173 | 0,7650 ± 0,0204 | 0,08 ± 0,02 s |
| 9 | 0,8063 ± 0,0064 | 0,8467 ± 0,0074 | 0,8815 ± 0,0210 | 0,7760 ± 0,0279 | 0,07 ± 0,00 s |
| 10 | 0,8198 ± 0,0169 | 0,8583 ± 0,0114 | 0,8792 ± 0,0227 | 0,7705 ± 0,0268 | 0,06 ± 0,00 s |

## Comportamento do método

- O corte é alheio ao motivo pelo qual um cliente é extremo. Sob partição non-IID, o cliente cujos dados são os mais atípicos é legítimo e, ainda assim, é exatamente o que tende a ser descartado — o que pode piorar o resultado em vez de protegê-lo.
- Com 5 clientes e `beta = 0,2`, o corte remove 1 cliente de cada extremo e agrega apenas os 3 centrais, descartando 40% dos participantes do round.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid" dataset="social_media" num-clients=5 seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid" dataset="social_media" num-clients=5 seed=44'
```
