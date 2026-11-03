# FedAdam

> Trata a diferença entre os pesos agregados e os pesos globais anteriores como se fosse um gradiente, e aplica sobre ela um passo do otimizador Adam no servidor — com momento de primeira e segunda ordem acumulados entre rounds. Enquanto o FedAvg sempre dá um passo do tamanho da média recebida, o FedAdam adapta o tamanho do passo por parâmetro, o que normalmente reduz o número de rounds até a convergência. O preço é um conjunto maior de hiperparâmetros para ajustar.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7142 ± 0,0008 | 0,7015 ± 0,0021 | 0,7881 ± 0,0004 | 1,0 | 4,7 ± 1,2 | 479,18 ± 6,38 s | 0,0132 ± 0,0014 | 0,6871 ± 0,0058 |
| natural (por país) | 3 | 0,7029 ± 0,0051 | 0,6843 ± 0,0131 | 0,7751 ± 0,0033 | 1,3 ± 0,5 | não atingido | 380,50 ± 4,59 s | 0,0614 ± 0,0054 | 0,5638 ± 0,0303 |

Dataset `pisa` · 48 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,7190. Alvos de convergência: 95% do teto = 0,6830 · 99% do teto = 0,7118.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6941 ± 0,0021 | 0,7103 ± 0,0077 | 0,7628 ± 0,0004 | 0,6942 ± 0,0026 | 46,41 ± 1,63 s |
| 2 | 0,7085 ± 0,0011 | 0,7193 ± 0,0020 | 0,7800 ± 0,0006 | 0,7080 ± 0,0018 | 48,82 ± 0,80 s |
| 3 | 0,6986 ± 0,0171 | 0,7259 ± 0,0064 | 0,7839 ± 0,0015 | 0,6985 ± 0,0184 | 48,30 ± 1,19 s |
| 4 | 0,6379 ± 0,0976 | 0,7094 ± 0,0302 | 0,7337 ± 0,0731 | 0,6374 ± 0,0973 | 47,01 ± 1,00 s |
| 5 | 0,7052 ± 0,0135 | 0,7193 ± 0,0077 | 0,7852 ± 0,0041 | 0,7042 ± 0,0152 | 49,22 ± 0,89 s |
| 6 | 0,7143 ± 0,0002 | 0,7085 ± 0,0015 | 0,7883 ± 0,0006 | 0,7159 ± 0,0003 | 47,81 ± 1,42 s |
| 7 | 0,7156 ± 0,0002 | 0,7165 ± 0,0009 | 0,7888 ± 0,0006 | 0,7161 ± 0,0009 | 47,84 ± 2,03 s |
| 8 | 0,7161 ± 0,0010 | 0,7139 ± 0,0067 | 0,7893 ± 0,0009 | 0,7156 ± 0,0007 | 48,22 ± 0,41 s |
| 9 | 0,7154 ± 0,0009 | 0,7063 ± 0,0023 | 0,7893 ± 0,0009 | 0,7153 ± 0,0003 | 47,73 ± 0,73 s |
| 10 | 0,7142 ± 0,0008 | 0,7015 ± 0,0021 | 0,7881 ± 0,0004 | 0,7147 ± 0,0006 | 47,82 ± 0,61 s |

### Curva por round — partição natural (por país)

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6808 ± 0,0050 | 0,6823 ± 0,0136 | 0,7451 ± 0,0069 | 0,6816 ± 0,0052 | 37,68 ± 0,64 s |
| 2 | 0,6970 ± 0,0042 | 0,6988 ± 0,0059 | 0,7689 ± 0,0029 | 0,6982 ± 0,0028 | 37,73 ± 0,45 s |
| 3 | 0,7035 ± 0,0059 | 0,7129 ± 0,0083 | 0,7770 ± 0,0020 | 0,7038 ± 0,0039 | 38,15 ± 0,68 s |
| 4 | 0,6857 ± 0,0161 | 0,7190 ± 0,0113 | 0,7768 ± 0,0024 | 0,6863 ± 0,0171 | 37,68 ± 0,43 s |
| 5 | 0,7051 ± 0,0009 | 0,7027 ± 0,0113 | 0,7764 ± 0,0018 | 0,7058 ± 0,0017 | 39,09 ± 1,71 s |
| 6 | 0,7035 ± 0,0013 | 0,6888 ± 0,0048 | 0,7752 ± 0,0003 | 0,7049 ± 0,0013 | 37,86 ± 0,12 s |
| 7 | 0,7034 ± 0,0022 | 0,6900 ± 0,0097 | 0,7758 ± 0,0010 | 0,7054 ± 0,0037 | 38,22 ± 1,52 s |
| 8 | 0,7063 ± 0,0004 | 0,6975 ± 0,0070 | 0,7774 ± 0,0009 | 0,7066 ± 0,0020 | 37,10 ± 1,54 s |
| 9 | 0,7043 ± 0,0043 | 0,6920 ± 0,0156 | 0,7772 ± 0,0024 | 0,7056 ± 0,0053 | 37,68 ± 0,50 s |
| 10 | 0,7029 ± 0,0051 | 0,6843 ± 0,0131 | 0,7751 ± 0,0033 | 0,7042 ± 0,0048 | 39,32 ± 0,74 s |

## Comportamento do método

- `eta` é a taxa de aprendizado do servidor — parâmetro que o FedAvg não possui, por não dar passo algum no servidor. Seu valor foi calibrado por varredura na partição IID; o registro está em `calibration_eta_fedadam.json`. Com `eta = 1,0` o treinamento divergiu, colapsando para 50% de acurácia nos últimos rounds.
- `eta_l` declara a taxa do cliente à estratégia, mas não controla o treino local nesta implementação: o cliente usa a `learning-rate` recebida via `train_config`. `tau` é o termo de estabilidade numérica, análogo ao epsilon do Adam convencional.
- Neste dataset o método converge mais devagar que o FedAvg, e nenhum `eta` testado alcançou 99% de acurácia em 10 rounds. O resultado é coerente com o problema: sendo linearmente separável, a média simples dos pesos já aponta direto para o ótimo, e o momento acumulado no servidor só adiciona inércia. A vantagem dos otimizadores adaptativos aparece quando a agregação simples é ruim — gradientes de magnitude muito variável entre clientes ou heterogeneidade severa — e não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="iid" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="iid" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="iid" dataset="pisa" num-clients=48 seed=44'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="natural" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="natural" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedadam" partition="natural" dataset="pisa" num-clients=48 seed=44'
```
