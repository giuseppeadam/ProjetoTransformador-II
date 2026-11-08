# FedYogi

> Mesma ideia do FedAdam — otimizador adaptativo no servidor — trocando Adam por Yogi. A diferença está em como o momento de segunda ordem é atualizado: o Adam o faz de forma multiplicativa, o que pode fazer o tamanho efetivo do passo crescer rápido demais; o Yogi usa atualização aditiva, controlando esse crescimento. Na prática tende a ser mais estável que o FedAdam quando os updates dos clientes variam muito de magnitude entre rounds — o que é justamente o caso sob partição non-IID.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7252 ± 0,1401 | 0,7203 ± 0,1952 | 0,7524 ± 0,1188 | 7,5 (em 2/3) | não atingido | 1,49 ± 0,39 s | 0,0913 ± 0,0360 | 0,6111 ± 0,1416 |
| non-IID | 3 | 0,6937 ± 0,0718 | 0,7851 ± 0,0235 | 0,7185 ± 0,1159 | 5,0 (em 1/3) | não atingido | 1,50 ± 0,43 s | 0,2433 ± 0,1169 | 0,2794 ± 0,2741 |

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,8649. Alvos de convergência: 95% do teto = 0,8216 · 99% do teto = 0,8562.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6937 ± 0,0627 | 0,7532 ± 0,0145 | 0,7955 ± 0,0434 | 0,7500 ± 0,1089 | 0,57 ± 0,08 s |
| 2 | 0,4775 ± 0,1562 | 0,5444 ± 0,1455 | 0,4209 ± 0,3192 | 0,4944 ± 0,1983 | 0,09 ± 0,04 s |
| 3 | 0,6892 ± 0,0584 | 0,6766 ± 0,0702 | 0,8499 ± 0,0147 | 0,7667 ± 0,0828 | 0,13 ± 0,05 s |
| 4 | 0,6937 ± 0,0627 | 0,7758 ± 0,0163 | 0,7660 ± 0,0754 | 0,6889 ± 0,0831 | 0,17 ± 0,13 s |
| 5 | 0,6667 ± 0,1821 | 0,7531 ± 0,1213 | 0,6731 ± 0,2401 | 0,6000 ± 0,2126 | 0,11 ± 0,04 s |
| 6 | 0,6441 ± 0,2071 | 0,6728 ± 0,1933 | 0,6715 ± 0,2765 | 0,6222 ± 0,2640 | 0,10 ± 0,03 s |
| 7 | 0,6712 ± 0,0230 | 0,7351 ± 0,0229 | 0,7229 ± 0,0801 | 0,6444 ± 0,1173 | 0,09 ± 0,04 s |
| 8 | 0,7117 ± 0,0459 | 0,7924 ± 0,0220 | 0,7789 ± 0,0723 | 0,7056 ± 0,0208 | 0,08 ± 0,00 s |
| 9 | 0,6577 ± 0,0355 | 0,7326 ± 0,0564 | 0,7557 ± 0,0794 | 0,6833 ± 0,0850 | 0,08 ± 0,02 s |
| 10 | 0,7252 ± 0,1401 | 0,7203 ± 0,1952 | 0,7524 ± 0,1188 | 0,7111 ± 0,1030 | 0,07 ± 0,01 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7027 ± 0,0689 | 0,7749 ± 0,0356 | 0,7346 ± 0,1797 | 0,7158 ± 0,0950 | 0,60 ± 0,15 s |
| 2 | 0,5721 ± 0,1240 | 0,4981 ± 0,3224 | 0,7244 ± 0,1594 | 0,6284 ± 0,1807 | 0,15 ± 0,08 s |
| 3 | 0,4820 ± 0,0919 | 0,3457 ± 0,3122 | 0,6536 ± 0,1087 | 0,4918 ± 0,0937 | 0,13 ± 0,05 s |
| 4 | 0,6171 ± 0,0230 | 0,7566 ± 0,0090 | 0,5742 ± 0,0751 | 0,6120 ± 0,0204 | 0,08 ± 0,02 s |
| 5 | 0,7342 ± 0,0674 | 0,8015 ± 0,0447 | 0,7794 ± 0,0543 | 0,6995 ± 0,0891 | 0,10 ± 0,05 s |
| 6 | 0,6396 ± 0,1784 | 0,5458 ± 0,3597 | 0,7881 ± 0,0518 | 0,6284 ± 0,1663 | 0,10 ± 0,05 s |
| 7 | 0,6441 ± 0,0951 | 0,6003 ± 0,1699 | 0,8111 ± 0,0375 | 0,6721 ± 0,1144 | 0,11 ± 0,07 s |
| 8 | 0,7207 ± 0,0775 | 0,7201 ± 0,1095 | 0,8255 ± 0,0141 | 0,7268 ± 0,1154 | 0,07 ± 0,00 s |
| 9 | 0,6622 ± 0,0862 | 0,7694 ± 0,0449 | 0,6132 ± 0,1412 | 0,6557 ± 0,0696 | 0,07 ± 0,01 s |
| 10 | 0,6937 ± 0,0718 | 0,7851 ± 0,0235 | 0,7185 ± 0,1159 | 0,6885 ± 0,0613 | 0,11 ± 0,06 s |

## Comportamento do método

- `eta`, a taxa de aprendizado do servidor, foi calibrada por varredura na partição IID; o registro está em `calibration_eta_fedyogi.json`.
- A comparação mais informativa é com o FedAdam, de quem difere apenas na forma de atualizar o momento de segunda ordem. Sob partição non-IID o FedYogi apresentou dispersão de acurácia entre clientes menor que a do FedAdam e atendeu melhor o cliente de pior desempenho, o que é consistente com a atualização aditiva: ela limita o crescimento do passo efetivo quando os updates dos clientes variam muito de magnitude, situação que a heterogeneidade dos dados provoca.
- Como o FedAdam, não alcança 99% de acurácia em 10 rounds neste dataset: a média simples do FedAvg já é adequada para um problema linearmente separável, e o passo adaptativo no servidor adiciona inércia sem benefício correspondente. A vantagem dessa família aparece quando a agregação simples é ruim, não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="iid" dataset="social_media" num-clients=5 seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi" partition="noniid" dataset="social_media" num-clients=5 seed=44'
```
