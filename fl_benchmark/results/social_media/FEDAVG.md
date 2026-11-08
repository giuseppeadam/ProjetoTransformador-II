# FedAvg

> A estratégia de referência de Federated Learning. A cada round o servidor calcula a média dos pesos recebidos, ponderada pelo número de exemplos de cada cliente — um cliente com dez vezes mais dados pesa dez vezes mais no modelo global. Simples e eficiente quando os dados são homogêneos, mas sofre com *client drift*: se as distribuições locais divergem, os modelos treinados localmente apontam para mínimos diferentes e a média entre eles pode não ser boa para nenhum dos clientes.
>
> **Referência:** McMahan et al., 2017 — arxiv.org/abs/1602.05629

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,8559 ± 0,0064 | 0,8778 ± 0,0054 | 0,9134 ± 0,0143 | 7,0 ± 1,4 | 8,0 (em 1/3) | 1,63 ± 0,75 s | 0,0754 ± 0,0155 | 0,7222 ± 0,0393 |
| non-IID | 3 | 0,8198 ± 0,0278 | 0,8521 ± 0,0206 | 0,8861 ± 0,0214 | 5,0 (em 2/3) | não atingido | 1,03 ± 0,22 s | 0,1334 ± 0,0522 | 0,5646 ± 0,1316 |

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,8649. Alvos de convergência: 95% do teto = 0,8216 · 99% do teto = 0,8562.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7613 ± 0,0169 | 0,7889 ± 0,0139 | 0,8817 ± 0,0146 | 0,8611 ± 0,0342 | 0,86 ± 0,69 s |
| 2 | 0,7658 ± 0,0064 | 0,7814 ± 0,0089 | 0,8876 ± 0,0078 | 0,8278 ± 0,0515 | 0,10 ± 0,01 s |
| 3 | 0,7883 ± 0,0127 | 0,8141 ± 0,0134 | 0,8912 ± 0,0077 | 0,8222 ± 0,0567 | 0,09 ± 0,03 s |
| 4 | 0,7973 | 0,8235 ± 0,0000 | 0,8943 ± 0,0082 | 0,8222 ± 0,0567 | 0,06 ± 0,00 s |
| 5 | 0,8063 ± 0,0064 | 0,8326 ± 0,0064 | 0,9004 ± 0,0092 | 0,8278 ± 0,0393 | 0,12 ± 0,06 s |
| 6 | 0,8243 ± 0,0191 | 0,8477 ± 0,0205 | 0,9032 ± 0,0104 | 0,8500 ± 0,0360 | 0,10 ± 0,04 s |
| 7 | 0,8288 ± 0,0230 | 0,8522 ± 0,0233 | 0,9086 ± 0,0123 | 0,8389 ± 0,0437 | 0,09 ± 0,02 s |
| 8 | 0,8423 ± 0,0230 | 0,8640 ± 0,0247 | 0,9114 ± 0,0127 | 0,8389 ± 0,0437 | 0,07 ± 0,00 s |
| 9 | 0,8468 ± 0,0169 | 0,8687 ± 0,0180 | 0,9132 ± 0,0123 | 0,8444 ± 0,0283 | 0,07 ± 0,00 s |
| 10 | 0,8559 ± 0,0064 | 0,8778 ± 0,0054 | 0,9134 ± 0,0143 | 0,8333 ± 0,0236 | 0,07 ± 0,01 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7162 ± 0,0221 | 0,7586 ± 0,0217 | 0,8679 ± 0,0292 | 0,7650 ± 0,0470 | 0,40 ± 0,12 s |
| 2 | 0,7793 ± 0,0337 | 0,8040 ± 0,0377 | 0,8766 ± 0,0183 | 0,7869 ± 0,0232 | 0,07 ± 0,01 s |
| 3 | 0,7748 ± 0,0169 | 0,7988 ± 0,0223 | 0,8787 ± 0,0145 | 0,7869 ± 0,0268 | 0,06 ± 0,00 s |
| 4 | 0,7883 ± 0,0064 | 0,8170 ± 0,0077 | 0,8792 ± 0,0122 | 0,8197 ± 0,0354 | 0,06 ± 0,00 s |
| 5 | 0,7748 ± 0,0127 | 0,8063 ± 0,0079 | 0,8828 ± 0,0138 | 0,8142 ± 0,0470 | 0,07 ± 0,01 s |
| 6 | 0,7838 ± 0,0221 | 0,8155 ± 0,0171 | 0,8851 ± 0,0172 | 0,7978 ± 0,0409 | 0,06 ± 0,01 s |
| 7 | 0,7838 ± 0,0221 | 0,8155 ± 0,0171 | 0,8840 ± 0,0204 | 0,7978 ± 0,0337 | 0,07 ± 0,01 s |
| 8 | 0,8108 ± 0,0191 | 0,8431 ± 0,0169 | 0,8856 ± 0,0211 | 0,7869 ± 0,0268 | 0,07 ± 0,02 s |
| 9 | 0,8153 ± 0,0169 | 0,8476 ± 0,0137 | 0,8874 ± 0,0215 | 0,7869 ± 0,0134 | 0,07 ± 0,01 s |
| 10 | 0,8198 ± 0,0278 | 0,8521 ± 0,0206 | 0,8861 ± 0,0214 | 0,7705 ± 0,0134 | 0,09 ± 0,05 s |

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="iid" dataset="social_media" num-clients=5 seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg" partition="noniid" dataset="social_media" num-clients=5 seed=44'
```
