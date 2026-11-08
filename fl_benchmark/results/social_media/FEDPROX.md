# FedProx

> Ataca o *client drift* na origem: em vez de mudar a agregação, restringe o quanto cada cliente pode se afastar do modelo global. O cliente soma à sua perda o termo proximal `(mu/2) * ||w - w_global||^2`, que penaliza o distanciamento dos pesos recebidos no início do round. Com `mu = 0` o método se reduz exatamente ao FedAvg; quanto maior o `mu`, mais conservador o treino local. É a estratégia projetada para dados heterogêneos (non-IID).
>
> **Referência:** Li et al., 2018 — arxiv.org/abs/1812.06127
>
> **Hiperparâmetros próprios:** `proximal_mu = 0.01`

## Resultados

| Partição | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| IID | 3 | 0,8423 ± 0,0127 | 0,8644 ± 0,0152 | 0,9126 ± 0,0152 | 7,0 ± 1,4 | 8,0 (em 1/3) | 1,74 ± 0,47 s | 0,0679 ± 0,0050 | 0,7500 |
| non-IID | 3 | 0,8288 ± 0,0278 | 0,8585 ± 0,0211 | 0,8884 ± 0,0214 | 4,5 (em 2/3) | 8,0 (em 1/3) | 1,08 ± 0,12 s | 0,1155 ± 0,0382 | 0,6217 ± 0,0919 |

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · α de Dirichlet = 0,3 · sementes 42, 43, 44 · dispositivo `cpu`

Teto centralizado do dataset: 0,8649. Alvos de convergência: 95% do teto = 0,8216 · 99% do teto = 0,8562.

Os valores são a média entre 3 execuções independentes, seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit — a ordem de chegada das respostas dos clientes varia entre execuções. Diferenças entre estratégias menores que essa dispersão não são conclusivas.

### Curva por round — partição IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7613 ± 0,0169 | 0,7889 ± 0,0139 | 0,8807 ± 0,0166 | 0,8611 ± 0,0342 | 0,95 ± 0,45 s |
| 2 | 0,7658 ± 0,0064 | 0,7814 ± 0,0089 | 0,8876 ± 0,0083 | 0,8278 ± 0,0515 | 0,08 ± 0,00 s |
| 3 | 0,7883 ± 0,0127 | 0,8141 ± 0,0134 | 0,8904 ± 0,0082 | 0,8222 ± 0,0567 | 0,12 ± 0,04 s |
| 4 | 0,7973 | 0,8235 ± 0,0000 | 0,8953 ± 0,0085 | 0,8222 ± 0,0567 | 0,08 ± 0,00 s |
| 5 | 0,8018 ± 0,0064 | 0,8281 ± 0,0064 | 0,8994 ± 0,0092 | 0,8278 ± 0,0393 | 0,08 ± 0,00 s |
| 6 | 0,8198 ± 0,0255 | 0,8444 ± 0,0250 | 0,9032 ± 0,0103 | 0,8500 ± 0,0360 | 0,09 ± 0,01 s |
| 7 | 0,8288 ± 0,0230 | 0,8522 ± 0,0233 | 0,9080 ± 0,0125 | 0,8444 ± 0,0416 | 0,08 ± 0,00 s |
| 8 | 0,8423 ± 0,0230 | 0,8640 ± 0,0247 | 0,9101 ± 0,0124 | 0,8444 ± 0,0416 | 0,08 ± 0,00 s |
| 9 | 0,8468 ± 0,0169 | 0,8687 ± 0,0180 | 0,9126 ± 0,0125 | 0,8500 ± 0,0360 | 0,09 ± 0,00 s |
| 10 | 0,8423 ± 0,0127 | 0,8644 ± 0,0152 | 0,9126 ± 0,0152 | 0,8389 ± 0,0208 | 0,08 ± 0,00 s |

### Curva por round — partição non-IID

| Round | Acurácia central | F1 central | ROC-AUC central | Acurácia federada | Tempo de treino |
|---|---|---|---|---|---|
| 1 | 0,7207 ± 0,0230 | 0,7616 ± 0,0205 | 0,8692 ± 0,0305 | 0,7650 ± 0,0470 | 0,26 ± 0,14 s |
| 2 | 0,7883 ± 0,0255 | 0,8102 ± 0,0350 | 0,8774 ± 0,0183 | 0,7923 ± 0,0204 | 0,09 ± 0,01 s |
| 3 | 0,7838 ± 0,0110 | 0,8035 ± 0,0226 | 0,8782 ± 0,0142 | 0,7923 ± 0,0279 | 0,09 ± 0,01 s |
| 4 | 0,7883 ± 0,0064 | 0,8170 ± 0,0077 | 0,8784 ± 0,0113 | 0,8087 ± 0,0337 | 0,08 ± 0,00 s |
| 5 | 0,7883 ± 0,0064 | 0,8157 ± 0,0025 | 0,8828 ± 0,0144 | 0,8142 ± 0,0470 | 0,09 ± 0,01 s |
| 6 | 0,7838 ± 0,0221 | 0,8155 ± 0,0171 | 0,8843 ± 0,0173 | 0,8142 ± 0,0409 | 0,09 ± 0,01 s |
| 7 | 0,7883 ± 0,0278 | 0,8188 ± 0,0213 | 0,8851 ± 0,0201 | 0,7978 ± 0,0337 | 0,09 ± 0,01 s |
| 8 | 0,8153 ± 0,0355 | 0,8454 ± 0,0291 | 0,8868 ± 0,0218 | 0,7869 ± 0,0268 | 0,10 ± 0,01 s |
| 9 | 0,8198 ± 0,0230 | 0,8508 ± 0,0182 | 0,8868 ± 0,0216 | 0,7814 ± 0,0077 | 0,09 ± 0,02 s |
| 10 | 0,8288 ± 0,0278 | 0,8585 ± 0,0211 | 0,8884 ± 0,0214 | 0,7814 ± 0,0077 | 0,08 ± 0,00 s |

## Comportamento do método

- A agregação no servidor é idêntica à do FedAvg. O que distingue o FedProx acontece no cliente: a estratégia apenas envia `proximal-mu` na configuração do round, e cabe ao cliente aplicar a penalidade na função de perda. Sem essa aplicação, selecionar FedProx produz resultados indistinguíveis de FedAvg.

## Como reproduzir

```bash
cd fl_benchmark
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="iid" dataset="social_media" num-clients=5 seed=44'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" dataset="social_media" num-clients=5 seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" dataset="social_media" num-clients=5 seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" dataset="social_media" num-clients=5 seed=44'
```
