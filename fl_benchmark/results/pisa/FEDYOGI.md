# FedYogi

> Mesma ideia do FedAdam — otimizador adaptativo no servidor — trocando Adam por Yogi. A diferença está em como o momento de segunda ordem é atualizado: o Adam o faz de forma multiplicativa, o que pode fazer o tamanho efetivo do passo crescer rápido demais; o Yogi usa atualização aditiva, controlando esse crescimento. Na prática tende a ser mais estável que o FedAdam quando os updates dos clientes variam muito de magnitude entre rounds — o que é justamente o caso sob partição non-IID.
>
> **Referência:** Reddi et al., 2020 — arxiv.org/abs/2003.00295
>
> **Hiperparâmetros próprios:** `eta = 0.3` · `eta_l = 0.1` · `beta_1 = 0.9` · `beta_2 = 0.99` · `tau = 1e-09`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,7135 ± 0,0033 | 0,7102 ± 0,0172 | 0,7870 ± 0,0011 | 1,0 | 6,0 ± 0,8 | 498,07 ± 16,06 s | 0,0140 ± 0,0012 | 0,6854 ± 0,0049 |
| natural (por país) | 3 | 0,7042 ± 0,0029 | 0,6886 ± 0,0128 | 0,7769 ± 0,0007 | 2,0 | não atingido | 377,72 ± 4,21 s | 0,0620 ± 0,0056 | 0,5756 ± 0,0269 |

Dataset `pisa` · 48 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,7190. Alvos de convergência: 95% do teto = 0,6830 · 99% do teto = 0,7118.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6916 ± 0,0020 | 0,7062 ± 0,0077 | 0,7587 ± 0,0016 | 0,6915 ± 0,0028 | 48,63 ± 1,53 s |
| 2 | 0,6956 ± 0,0103 | 0,7017 ± 0,0218 | 0,7662 ± 0,0111 | 0,6971 ± 0,0085 | 50,22 ± 2,62 s |
| 3 | 0,5805 ± 0,0739 | 0,6882 ± 0,0377 | 0,6277 ± 0,1661 | 0,5799 ± 0,0746 | 49,37 ± 1,83 s |
| 4 | 0,6819 ± 0,0280 | 0,7241 ± 0,0017 | 0,7795 ± 0,0037 | 0,6810 ± 0,0286 | 48,88 ± 2,32 s |
| 5 | 0,6855 ± 0,0376 | 0,6937 ± 0,0096 | 0,7598 ± 0,0394 | 0,6868 ± 0,0369 | 49,77 ± 1,40 s |
| 6 | 0,7140 ± 0,0029 | 0,7098 ± 0,0051 | 0,7851 ± 0,0037 | 0,7127 ± 0,0040 | 49,93 ± 2,02 s |
| 7 | 0,7116 ± 0,0020 | 0,7026 ± 0,0175 | 0,7850 ± 0,0017 | 0,7108 ± 0,0011 | 49,61 ± 1,78 s |
| 8 | 0,6715 ± 0,0400 | 0,6213 ± 0,0636 | 0,7496 ± 0,0445 | 0,6721 ± 0,0400 | 50,99 ± 1,29 s |
| 9 | 0,6892 ± 0,0214 | 0,6598 ± 0,0354 | 0,7496 ± 0,0489 | 0,6908 ± 0,0208 | 50,05 ± 1,47 s |
| 10 | 0,7135 ± 0,0033 | 0,7102 ± 0,0172 | 0,7870 ± 0,0011 | 0,7134 ± 0,0023 | 50,63 ± 1,65 s |

### Curva por round — partição natural (por país)

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,6742 ± 0,0084 | 0,6740 ± 0,0195 | 0,7361 ± 0,0108 | 0,6753 ± 0,0084 | 37,81 ± 0,50 s |
| 2 | 0,6962 ± 0,0014 | 0,6996 ± 0,0090 | 0,7658 ± 0,0027 | 0,6972 ± 0,0012 | 39,14 ± 0,29 s |
| 3 | 0,5596 ± 0,0843 | 0,6862 ± 0,0276 | 0,5141 ± 0,1716 | 0,5591 ± 0,0835 | 36,06 ± 0,23 s |
| 4 | 0,6224 ± 0,0830 | 0,6939 ± 0,0211 | 0,7011 ± 0,0956 | 0,6237 ± 0,0835 | 37,93 ± 0,87 s |
| 5 | 0,7031 ± 0,0013 | 0,6843 ± 0,0071 | 0,7749 ± 0,0010 | 0,7042 ± 0,0020 | 38,70 ± 0,94 s |
| 6 | 0,7062 ± 0,0014 | 0,7010 ± 0,0087 | 0,7759 ± 0,0009 | 0,7064 ± 0,0012 | 37,80 ± 1,35 s |
| 7 | 0,7015 ± 0,0047 | 0,6963 ± 0,0234 | 0,7741 ± 0,0020 | 0,7024 ± 0,0054 | 37,12 ± 0,56 s |
| 8 | 0,6929 ± 0,0068 | 0,6587 ± 0,0160 | 0,7691 ± 0,0044 | 0,6938 ± 0,0053 | 37,71 ± 0,80 s |
| 9 | 0,6793 ± 0,0225 | 0,6592 ± 0,0136 | 0,7483 ± 0,0297 | 0,6824 ± 0,0223 | 38,51 ± 0,91 s |
| 10 | 0,7042 ± 0,0029 | 0,6886 ± 0,0128 | 0,7769 ± 0,0007 | 0,7055 ± 0,0044 | 36,94 ± 0,44 s |

## Comportamento do método

- `eta`, a taxa de aprendizado do servidor, foi calibrada por varredura na partição IID; o registro está em `calibration_eta_fedyogi.json`.
- A comparação mais informativa é com o FedAdam, de quem difere apenas na forma de atualizar o momento de segunda ordem. Sob partição non-IID o FedYogi apresentou dispersão de acurácia entre clientes menor que a do FedAdam e atendeu melhor o cliente de pior desempenho, o que é consistente com a atualização aditiva: ela limita o crescimento do passo efetivo quando os updates dos clientes variam muito de magnitude, situação que a heterogeneidade dos dados provoca.
- Como o FedAdam, não alcança 99% de acurácia em 10 rounds neste dataset: a média simples do FedAvg já é adequada para um problema linearmente separável, e o passo adaptativo no servidor adiciona inércia sem benefício correspondente. A vantagem dessa família aparece quando a agregação simples é ruim, não num problema fácil.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="iid" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="iid" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="iid" dataset="pisa" num-clients=48 seed=44'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="natural" dataset="pisa" num-clients=48 seed=42'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="natural" dataset="pisa" num-clients=48 seed=43'
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedyogi" partition="natural" dataset="pisa" num-clients=48 seed=44'
```
