# FedProx

> Ataca o *client drift* na origem: em vez de mudar a agregação, restringe o quanto cada cliente pode se afastar do modelo global. O cliente soma à sua perda o termo proximal `(mu/2) * ||w - w_global||^2`, que penaliza o distanciamento dos pesos recebidos no início do round. Com `mu = 0` o método se reduz exatamente ao FedAvg; quanto maior o `mu`, mais conservador o treino local. É a estratégia projetada para dados heterogêneos (non-IID).
>
> **Referência:** Li et al., 2018 — arxiv.org/abs/1812.06127
>
> **Hiperparâmetros próprios:** `proximal_mu = 0.01`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7161 ± 0,0006 | 0,7167 ± 0,0016 | 0,7891 ± 0,0005 | 1,0 | 5,0 | 696,58 ± 10,40 s | 0,0143 ± 0,0011 | 0,6848 ± 0,0017 |
| natural (por país) | 3 | 0,7063 ± 0,0004 | 0,7022 ± 0,0017 | 0,7790 ± 0,0004 | 1,0 | não atingido | 559,60 ± 13,16 s | 0,0589 ± 0,0017 | 0,5777 ± 0,0091 |

Dataset `pisa` · 48 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,7190. Alvos de convergência: 95% do teto = 0,6830 · 99% do teto = 0,7118.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7025 ± 0,0006 | 0,7094 ± 0,0007 | 0,7726 ± 0,0008 | 0,7030 ± 0,0016 | 69,74 ± 0,87 s |
| 2 | 0,7060 ± 0,0002 | 0,7142 ± 0,0010 | 0,7778 ± 0,0002 | 0,7058 ± 0,0014 | 69,08 ± 1,61 s |
| 3 | 0,7092 ± 0,0009 | 0,7148 ± 0,0022 | 0,7808 ± 0,0001 | 0,7092 ± 0,0016 | 68,26 ± 1,28 s |
| 4 | 0,7106 ± 0,0008 | 0,7159 ± 0,0025 | 0,7831 ± 0,0003 | 0,7109 ± 0,0015 | 70,56 ± 2,57 s |
| 5 | 0,7123 ± 0,0003 | 0,7153 ± 0,0016 | 0,7848 ± 0,0004 | 0,7130 ± 0,0010 | 70,99 ± 1,72 s |
| 6 | 0,7139 ± 0,0005 | 0,7158 ± 0,0016 | 0,7863 ± 0,0003 | 0,7135 ± 0,0012 | 69,67 ± 1,10 s |
| 7 | 0,7146 ± 0,0003 | 0,7164 ± 0,0003 | 0,7873 ± 0,0004 | 0,7144 ± 0,0008 | 69,24 ± 1,04 s |
| 8 | 0,7153 ± 0,0005 | 0,7161 ± 0,0007 | 0,7880 ± 0,0005 | 0,7151 ± 0,0005 | 71,82 ± 0,52 s |
| 9 | 0,7159 ± 0,0004 | 0,7171 ± 0,0015 | 0,7886 ± 0,0005 | 0,7157 ± 0,0005 | 68,95 ± 1,82 s |
| 10 | 0,7161 ± 0,0006 | 0,7167 ± 0,0016 | 0,7891 ± 0,0005 | 0,7163 ± 0,0009 | 68,28 ± 0,88 s |

### Curva por round — partição natural (por país)

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6939 ± 0,0005 | 0,6873 ± 0,0098 | 0,7630 ± 0,0013 | 0,6968 ± 0,0014 | 56,37 ± 1,17 s |
| 2 | 0,6986 ± 0,0010 | 0,6992 ± 0,0024 | 0,7691 ± 0,0006 | 0,7007 ± 0,0017 | 56,29 ± 1,31 s |
| 3 | 0,6992 ± 0,0008 | 0,6983 ± 0,0008 | 0,7705 ± 0,0005 | 0,7019 ± 0,0017 | 56,35 ± 3,18 s |
| 4 | 0,7000 ± 0,0006 | 0,6982 ± 0,0015 | 0,7718 ± 0,0004 | 0,7027 ± 0,0019 | 54,16 ± 0,29 s |
| 5 | 0,7015 ± 0,0000 | 0,6980 ± 0,0010 | 0,7733 ± 0,0004 | 0,7039 ± 0,0020 | 55,10 ± 2,11 s |
| 6 | 0,7021 ± 0,0005 | 0,6991 ± 0,0033 | 0,7744 ± 0,0001 | 0,7051 ± 0,0018 | 55,80 ± 1,97 s |
| 7 | 0,7033 ± 0,0006 | 0,6974 ± 0,0014 | 0,7760 ± 0,0001 | 0,7057 ± 0,0012 | 55,48 ± 2,60 s |
| 8 | 0,7044 ± 0,0006 | 0,6976 ± 0,0038 | 0,7773 ± 0,0004 | 0,7067 ± 0,0017 | 58,53 ± 5,55 s |
| 9 | 0,7052 ± 0,0005 | 0,7023 ± 0,0040 | 0,7781 ± 0,0004 | 0,7080 ± 0,0015 | 55,45 ± 0,48 s |
| 10 | 0,7063 ± 0,0004 | 0,7022 ± 0,0017 | 0,7790 ± 0,0004 | 0,7089 ± 0,0010 | 56,07 ± 1,21 s |

## Comportamento do método

- A agregação no servidor é idêntica à do FedAvg. O que distingue o FedProx acontece no cliente: a estratégia apenas envia `proximal-mu` na configuração do round, e cabe ao cliente aplicar a penalidade na função de perda. Sem essa aplicação, selecionar FedProx produz resultados indistinguíveis de FedAvg.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="iid" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="iid" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="iid" dataset="pisa" num-clients=48 seed=44'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="natural" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="natural" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedprox" partition="natural" dataset="pisa" num-clients=48 seed=44'
```
