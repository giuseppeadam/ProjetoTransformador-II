# FedAdam

> Trata a diferença entre os pesos agregados e os pesos globais anteriores como se fosse um gradiente, e aplica sobre ela um passo do otimizador Adam no servidor — com momento de primeira e segunda ordem acumulados entre rounds. Enquanto o FedAvg sempre dá um passo do tamanho da média recebida, o FedAdam adapta o tamanho do passo por parâmetro, o que normalmente reduz o número de rounds até a convergência. O preço é um conjunto maior de hiperparâmetros para ajustar.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,9860 ± 0,0025 | 0,9861 ± 0,0025 | 0,9994 ± 0,0002 | 4,0 | não atingido | 28,10 ± 3,90 s | 0,0057 ± 0,0017 | 0,9750 ± 0,0034 |
| non-IID | 3 | 0,9820 ± 0,0030 | 0,9821 ± 0,0029 | 0,9991 ± 0,0003 | 4,3 ± 0,5 | não atingido | 23,47 ± 1,05 s | 0,0098 ± 0,0036 | 0,9602 ± 0,0116 |

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,8595 ± 0,0113 | 0,8590 ± 0,0081 | 0,9467 ± 0,0107 | 0,8570 ± 0,0105 | 2,57 ± 0,17 s |
| 2 | 0,8999 ± 0,0037 | 0,9002 ± 0,0012 | 0,9645 ± 0,0081 | 0,8999 ± 0,0050 | 2,86 ± 0,27 s |
| 3 | 0,9335 ± 0,0044 | 0,9338 ± 0,0041 | 0,9780 ± 0,0032 | 0,9339 ± 0,0031 | 2,62 ± 0,23 s |
| 4 | 0,9599 ± 0,0033 | 0,9602 ± 0,0030 | 0,9926 ± 0,0016 | 0,9606 ± 0,0051 | 2,57 ± 0,22 s |
| 5 | 0,9726 ± 0,0004 | 0,9729 ± 0,0002 | 0,9981 ± 0,0002 | 0,9760 ± 0,0017 | 3,00 ± 0,40 s |
| 6 | 0,9766 ± 0,0022 | 0,9768 ± 0,0022 | 0,9984 ± 0,0005 | 0,9779 ± 0,0028 | 2,76 ± 0,42 s |
| 7 | 0,9739 ± 0,0023 | 0,9739 ± 0,0025 | 0,9981 ± 0,0002 | 0,9739 ± 0,0031 | 2,56 ± 0,26 s |
| 8 | 0,9766 ± 0,0037 | 0,9764 ± 0,0038 | 0,9983 ± 0,0005 | 0,9762 ± 0,0038 | 3,03 ± 0,72 s |
| 9 | 0,9822 ± 0,0038 | 0,9822 ± 0,0039 | 0,9990 ± 0,0002 | 0,9797 ± 0,0018 | 3,14 ± 0,88 s |
| 10 | 0,9860 ± 0,0025 | 0,9861 ± 0,0025 | 0,9994 ± 0,0002 | 0,9847 ± 0,0023 | 2,98 ± 0,66 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,8605 ± 0,0106 | 0,8645 ± 0,0014 | 0,9498 ± 0,0074 | 0,8616 ± 0,0123 | 2,28 ± 0,22 s |
| 2 | 0,8925 ± 0,0059 | 0,8943 ± 0,0029 | 0,9580 ± 0,0083 | 0,8884 ± 0,0098 | 2,31 ± 0,20 s |
| 3 | 0,9217 ± 0,0049 | 0,9220 ± 0,0049 | 0,9728 ± 0,0038 | 0,9224 ± 0,0071 | 2,45 ± 0,36 s |
| 4 | 0,9488 ± 0,0051 | 0,9486 ± 0,0059 | 0,9908 ± 0,0006 | 0,9494 ± 0,0084 | 2,25 ± 0,04 s |
| 5 | 0,9667 ± 0,0056 | 0,9666 ± 0,0060 | 0,9970 ± 0,0006 | 0,9665 ± 0,0091 | 2,20 ± 0,10 s |
| 6 | 0,9781 ± 0,0065 | 0,9779 ± 0,0068 | 0,9986 ± 0,0007 | 0,9767 ± 0,0091 | 2,36 ± 0,06 s |
| 7 | 0,9784 ± 0,0047 | 0,9782 ± 0,0050 | 0,9987 ± 0,0005 | 0,9793 ± 0,0067 | 2,38 ± 0,10 s |
| 8 | 0,9776 ± 0,0015 | 0,9775 ± 0,0017 | 0,9987 ± 0,0002 | 0,9790 ± 0,0023 | 2,35 ± 0,18 s |
| 9 | 0,9791 ± 0,0017 | 0,9792 ± 0,0016 | 0,9989 ± 0,0002 | 0,9801 ± 0,0043 | 2,47 ± 0,30 s |
| 10 | 0,9820 ± 0,0030 | 0,9821 ± 0,0029 | 0,9991 ± 0,0003 | 0,9826 ± 0,0039 | 2,41 ± 0,11 s |

## Comportamento do método

- `eta` é a taxa de aprendizado do servidor — parâmetro que o FedAvg não possui, por não dar passo algum no servidor. Seu valor foi calibrado por varredura na partição IID; o registro está em `calibration_eta_fedadam.json`. Com `eta = 1,0` o treinamento divergiu, colapsando para 50% de acurácia nos últimos rounds.
- `eta_l` declara a taxa do cliente à estratégia, mas não controla o treino local nesta implementação: o cliente usa a `learning-rate` recebida via `train_config`. `tau` é o termo de estabilidade numérica, análogo ao epsilon do Adam convencional.
- Neste dataset o método converge mais devagar que o FedAvg, e nenhum `eta` testado alcançou 99% de acurácia em 10 rounds. O resultado é coerente com o problema: sendo linearmente separável, a média simples dos pesos já aponta direto para o ótimo, e o momento acumulado no servidor só adiciona inércia. A vantagem dos otimizadores adaptativos aparece quando a agregação simples é ruim — gradientes de magnitude muito variável entre clientes ou heterogeneidade severa — e não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="iid" seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam" partition="noniid" seed=44'
```
