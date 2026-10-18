# FedProx

> Ataca o *client drift* na origem: em vez de mudar a agregação, restringe o quanto cada cliente pode se afastar do modelo global. O cliente soma à sua perda o termo proximal `(mu/2) * ||w - w_global||^2`, que penaliza o distanciamento dos pesos recebidos no início do round. Com `mu = 0` o método se reduz exatamente ao FedAvg; quanto maior o `mu`, mais conservador o treino local. É a estratégia projetada para dados heterogêneos (non-IID).
>
> **Referência:** Li et al., 2018 — arxiv.org/abs/1812.06127
>
> **Hiperparâmetros próprios:** `proximal_mu = 0.01`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,9941 ± 0,0008 | 0,9941 ± 0,0008 | 0,9999 ± 0,0000 | 1,0 | 2,3 ± 0,5 | 34,38 ± 0,60 s | 0,0023 ± 0,0003 | 0,9911 ± 0,0019 |
| non-IID | 3 | 0,9925 ± 0,0007 | 0,9925 ± 0,0007 | 0,9999 ± 0,0000 | 1,0 | 3,3 ± 1,2 | 32,65 ± 4,52 s | 0,0057 ± 0,0022 | 0,9831 ± 0,0083 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,9820 ± 0,0008 | 0,9822 ± 0,0008 | 0,9992 ± 0,0002 | 0,9827 ± 0,0008 | 3,26 ± 0,10 s |
| 2 | 0,9907 ± 0,0015 | 0,9908 ± 0,0014 | 0,9997 ± 0,0001 | 0,9900 ± 0,0012 | 3,47 ± 0,11 s |
| 3 | 0,9917 ± 0,0012 | 0,9917 ± 0,0013 | 0,9998 ± 0,0001 | 0,9905 ± 0,0013 | 3,44 ± 0,06 s |
| 4 | 0,9902 ± 0,0006 | 0,9902 ± 0,0006 | 0,9998 ± 0,0000 | 0,9905 ± 0,0023 | 3,59 ± 0,28 s |
| 5 | 0,9939 ± 0,0003 | 0,9939 ± 0,0003 | 0,9999 ± 0,0000 | 0,9914 ± 0,0013 | 3,44 ± 0,12 s |
| 6 | 0,9925 ± 0,0009 | 0,9925 ± 0,0009 | 0,9998 ± 0,0001 | 0,9901 ± 0,0013 | 3,39 ± 0,06 s |
| 7 | 0,9936 ± 0,0004 | 0,9936 ± 0,0004 | 0,9999 ± 0,0000 | 0,9925 ± 0,0005 | 3,45 ± 0,04 s |
| 8 | 0,9916 ± 0,0005 | 0,9916 ± 0,0005 | 0,9998 ± 0,0000 | 0,9906 ± 0,0005 | 3,43 ± 0,02 s |
| 9 | 0,9925 ± 0,0011 | 0,9925 ± 0,0011 | 0,9999 ± 0,0000 | 0,9922 ± 0,0021 | 3,50 ± 0,13 s |
| 10 | 0,9941 ± 0,0008 | 0,9941 ± 0,0008 | 0,9999 ± 0,0000 | 0,9949 ± 0,0018 | 3,40 ± 0,01 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,9746 ± 0,0085 | 0,9743 ± 0,0090 | 0,9980 ± 0,0013 | 0,9727 ± 0,0127 | 3,36 ± 0,43 s |
| 2 | 0,9853 ± 0,0039 | 0,9852 ± 0,0039 | 0,9994 ± 0,0002 | 0,9842 ± 0,0047 | 3,45 ± 0,74 s |
| 3 | 0,9892 ± 0,0024 | 0,9892 ± 0,0024 | 0,9997 ± 0,0001 | 0,9916 ± 0,0016 | 3,04 ± 0,33 s |
| 4 | 0,9899 ± 0,0024 | 0,9899 ± 0,0024 | 0,9997 ± 0,0002 | 0,9883 ± 0,0037 | 3,22 ± 0,55 s |
| 5 | 0,9886 ± 0,0022 | 0,9887 ± 0,0021 | 0,9999 ± 0,0000 | 0,9891 ± 0,0007 | 2,99 ± 0,22 s |
| 6 | 0,9917 ± 0,0016 | 0,9916 ± 0,0017 | 0,9998 ± 0,0001 | 0,9907 ± 0,0011 | 3,30 ± 0,70 s |
| 7 | 0,9918 ± 0,0008 | 0,9918 ± 0,0008 | 0,9999 ± 0,0000 | 0,9910 ± 0,0034 | 3,18 ± 0,11 s |
| 8 | 0,9904 ± 0,0014 | 0,9904 ± 0,0014 | 0,9998 ± 0,0001 | 0,9914 ± 0,0034 | 3,32 ± 0,48 s |
| 9 | 0,9915 ± 0,0027 | 0,9915 ± 0,0026 | 0,9998 ± 0,0001 | 0,9921 ± 0,0012 | 3,37 ± 0,49 s |
| 10 | 0,9925 ± 0,0007 | 0,9925 ± 0,0007 | 0,9999 ± 0,0000 | 0,9926 ± 0,0012 | 3,42 ± 0,63 s |

## Comportamento do método

- A agregação no servidor é idêntica à do FedAvg. O que distingue o FedProx acontece no cliente: a estratégia apenas envia `proximal-mu` na configuração do round, e cabe ao cliente aplicar a penalidade na função de perda. Sem essa aplicação, selecionar FedProx produz resultados indistinguíveis de FedAvg.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=44'
```
