# FedAdam

> Trata a diferença entre os pesos agregados e os pesos globais anteriores como se fosse um gradiente, e aplica sobre ela um passo do otimizador Adam no servidor — com momento de primeira e segunda ordem acumulados entre rounds. Enquanto o FedAvg sempre dá um passo do tamanho da média recebida, o FedAdam adapta o tamanho do passo por parâmetro, o que normalmente reduz o número de rounds até a convergência. O preço é um conjunto maior de hiperparâmetros para ajustar.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7297 ± 0,0292 | 0,7729 ± 0,0140 | 0,8010 ± 0,0644 | 7,0 (em 1/3) | não atingido | 0,83 ± 0,03 s | 0,1164 ± 0,0181 | 0,5556 ± 0,0786 |
| non-IID | 3 | 0,7703 ± 0,0292 | 0,8026 ± 0,0287 | 0,8469 ± 0,0296 | 7,0 (em 1/3) | não atingido | 1,00 ± 0,14 s | 0,1026 ± 0,0626 | 0,5884 ± 0,1371 |

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,8649. Alvos de convergência: 95% do teto = 0,8216 · 99% do teto = 0,8562.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6982 ± 0,0647 | 0,7585 ± 0,0080 | 0,8209 ± 0,0235 | 0,7444 ± 0,1091 | 0,22 ± 0,02 s |
| 2 | 0,5360 ± 0,1975 | 0,6267 ± 0,1511 | 0,5854 ± 0,3274 | 0,5667 ± 0,2520 | 0,06 ± 0,00 s |
| 3 | 0,6712 ± 0,1180 | 0,6291 ± 0,1990 | 0,8525 ± 0,0340 | 0,6778 ± 0,1612 | 0,07 ± 0,02 s |
| 4 | 0,6757 ± 0,0552 | 0,7659 ± 0,0114 | 0,7179 ± 0,1420 | 0,6722 ± 0,0550 | 0,06 ± 0,00 s |
| 5 | 0,7027 ± 0,0689 | 0,7862 ± 0,0347 | 0,7801 ± 0,0596 | 0,7000 ± 0,0680 | 0,10 ± 0,05 s |
| 6 | 0,6892 ± 0,0981 | 0,7483 ± 0,0563 | 0,7974 ± 0,0731 | 0,7333 ± 0,0471 | 0,06 ± 0,01 s |
| 7 | 0,7162 ± 0,1342 | 0,6864 ± 0,2136 | 0,7837 ± 0,0631 | 0,6389 ± 0,1812 | 0,06 ± 0,00 s |
| 8 | 0,7072 ± 0,0885 | 0,7664 ± 0,0485 | 0,6549 ± 0,2571 | 0,7167 ± 0,0953 | 0,08 ± 0,02 s |
| 9 | 0,7252 ± 0,0127 | 0,7891 ± 0,0132 | 0,8105 ± 0,0314 | 0,7389 ± 0,0314 | 0,06 ± 0,00 s |
| 10 | 0,7297 ± 0,0292 | 0,7729 ± 0,0140 | 0,8010 ± 0,0644 | 0,7111 ± 0,0515 | 0,06 ± 0,00 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7162 ± 0,0833 | 0,7826 ± 0,0464 | 0,7660 ± 0,1450 | 0,7213 ± 0,1071 | 0,32 ± 0,12 s |
| 2 | 0,6667 ± 0,0566 | 0,7069 ± 0,0632 | 0,7192 ± 0,1923 | 0,6995 ± 0,1082 | 0,08 ± 0,01 s |
| 3 | 0,6622 ± 0,0552 | 0,7198 ± 0,0442 | 0,6759 ± 0,2066 | 0,6721 ± 0,0613 | 0,08 ± 0,01 s |
| 4 | 0,5991 ± 0,1566 | 0,7061 ± 0,1017 | 0,5621 ± 0,2690 | 0,5902 ± 0,1144 | 0,07 ± 0,00 s |
| 5 | 0,6441 ± 0,0608 | 0,7396 ± 0,0071 | 0,7305 ± 0,1076 | 0,6557 ± 0,0483 | 0,07 ± 0,00 s |
| 6 | 0,7432 ± 0,0292 | 0,7932 ± 0,0051 | 0,8049 ± 0,0444 | 0,7377 ± 0,0354 | 0,07 ± 0,01 s |
| 7 | 0,7523 ± 0,0627 | 0,7691 ± 0,0718 | 0,8324 ± 0,0311 | 0,7104 ± 0,0507 | 0,09 ± 0,02 s |
| 8 | 0,6216 ± 0,0943 | 0,6927 ± 0,0771 | 0,6935 ± 0,1540 | 0,6120 ± 0,1695 | 0,08 ± 0,01 s |
| 9 | 0,7387 ± 0,0665 | 0,7980 ± 0,0467 | 0,8040 ± 0,0263 | 0,7104 ± 0,0470 | 0,07 ± 0,01 s |
| 10 | 0,7703 ± 0,0292 | 0,8026 ± 0,0287 | 0,8469 ± 0,0296 | 0,7650 ± 0,0077 | 0,08 ± 0,02 s |

## Comportamento do método

- `eta` é a taxa de aprendizado do servidor — parâmetro que o FedAvg não possui, por não dar passo algum no servidor. Seu valor foi calibrado por varredura na partição IID; o registro está em `calibration_eta_fedadam.json`. Com `eta = 1,0` o treinamento divergiu, colapsando para 50% de acurácia nos últimos rounds.
- `eta_l` declara a taxa do cliente à estratégia, mas não controla o treino local nesta implementação: o cliente usa a `learning-rate` recebida via `train_config`. `tau` é o termo de estabilidade numérica, análogo ao epsilon do Adam convencional.
- Neste dataset o método converge mais devagar que o FedAvg, e nenhum `eta` testado alcançou 99% de acurácia em 10 rounds. O resultado é coerente com o problema: sendo linearmente separável, a média simples dos pesos já aponta direto para o ótimo, e o momento acumulado no servidor só adiciona inércia. A vantagem dos otimizadores adaptativos aparece quando a agregação simples é ruim — gradientes de magnitude muito variável entre clientes ou heterogeneidade severa — e não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" dataset="social_media" num-clients=5 seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" dataset="social_media" num-clients=5 seed=44'
```
