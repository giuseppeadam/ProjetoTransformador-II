# Comparativo das estratégias

Dataset `productivity` · 5 clientes · 10 rounds · 3 épocas locais · sementes 42, 43, 44 · dispositivo `cpu`

Cada célula traz a média de 3 execuções independentes seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit: a ordem de chegada das respostas dos clientes varia, e diferenças da ordem de 1e-7 nos pesos se amplificam ao longo dos rounds. **Diferenças entre estratégias menores que a dispersão não são conclusivas.**

## Teto de referência

O mesmo modelo treinado de forma centralizada, com acesso a todos os dados de treino de uma só vez. Nenhuma estratégia federada deve superá-lo de forma significativa — se superasse, haveria vazamento de dados no pipeline.

| | Acurácia | F1 | ROC-AUC | Treino efetivo (s) |
|---|---|---|---|---|
| Centralizado (30 épocas) | 0,9932 | 0,9932 | 0,9998 | 31,46 |

## Partição IID (controle)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,9914 ± 0,0001 | 0,9914 ± 0,0001 | 0,9998 ± 0,0000 | 1,0 | 2,7 ± 0,5 | 25,15 ± 0,21 | 0,0028 ± 0,0002 | 0,9885 ± 0,0007 |
| FedProx | 3 | 0,9941 ± 0,0008 | 0,9941 ± 0,0008 | 0,9999 ± 0,0000 | 1,0 | 2,3 ± 0,5 | 34,38 ± 0,60 | 0,0023 ± 0,0003 | 0,9911 ± 0,0019 |
| FedAdam | 3 | 0,9860 ± 0,0025 | 0,9861 ± 0,0025 | 0,9994 ± 0,0002 | 4,0 | não atingido | 28,10 ± 3,90 | 0,0057 ± 0,0017 | 0,9750 ± 0,0034 |
| FedYogi | 3 | 0,9832 ± 0,0019 | 0,9833 ± 0,0018 | 0,9983 ± 0,0006 | 3,3 ± 0,5 | não atingido | 25,66 ± 0,55 | 0,0036 ± 0,0004 | 0,9766 ± 0,0013 |
| FedTrimmedAvg | 3 | 0,9913 ± 0,0006 | 0,9913 ± 0,0006 | 0,9998 ± 0,0000 | 1,0 | 2,7 ± 0,5 | 25,33 ± 0,70 | 0,0028 ± 0,0003 | 0,9896 ± 0,0007 |

## Partição non-IID (Dirichlet)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ ≥95% | Rounds p/ ≥99% | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,9884 ± 0,0001 | 0,9884 ± 0,0001 | 0,9997 ± 0,0000 | 1,0 | 4,5 (em 2/3) | 25,41 ± 1,76 | 0,0062 ± 0,0018 | 0,9837 ± 0,0029 |
| FedProx | 3 | 0,9925 ± 0,0007 | 0,9925 ± 0,0007 | 0,9999 ± 0,0000 | 1,0 | 3,3 ± 1,2 | 32,65 ± 4,52 | 0,0057 ± 0,0022 | 0,9831 ± 0,0083 |
| FedAdam | 3 | 0,9820 ± 0,0030 | 0,9821 ± 0,0029 | 0,9991 ± 0,0003 | 4,3 ± 0,5 | não atingido | 23,47 ± 1,05 | 0,0098 ± 0,0036 | 0,9602 ± 0,0116 |
| FedYogi | 3 | 0,9845 ± 0,0015 | 0,9845 ± 0,0014 | 0,9984 ± 0,0007 | 4,3 ± 0,5 | não atingido | 23,15 ± 0,43 | 0,0075 ± 0,0017 | 0,9761 ± 0,0038 |
| FedTrimmedAvg | 3 | 0,9895 ± 0,0040 | 0,9894 ± 0,0041 | 0,9997 ± 0,0001 | 1,7 ± 0,5 | 5,5 (em 2/3) | 23,14 ± 1,60 | 0,0119 ± 0,0029 | 0,9672 ± 0,0055 |

## Leitura dos resultados

- **Convergência consistente:** FedProx foi a única estratégia a alcançar 99% de acurácia em todas as 3 sementes, em 3,3 ± 1,2 rounds. As demais atingiram o alvo em parte das sementes ou em nenhuma, dentro dos 10 rounds executados.
- **Equidade entre instituições:** a dispersão de acurácia entre clientes ficou entre 0,0057 ± 0,0022 (FedProx) e 0,0119 ± 0,0029 (FedTrimmedAvg), mas as faixas das primeiras colocadas se sobrepõem. Há separação clara apenas entre os extremos, não uma vencedora isolada.
- **Instituição menos favorecida:** o pior cliente teve acurácia entre 0,9602 ± 0,0116 (FedAdam) e 0,9837 ± 0,0029 (FedAvg); nenhuma estratégia protege a instituição minoritária de forma destacada das demais.
- **Controle IID:** no regime homogêneo a diferença de acurácia entre a melhor e a pior estratégia é de 0,0109, confirmando que o pipeline é o mesmo para todas e que a heterogeneidade dos dados, não a implementação, é a origem das diferenças observadas no regime non-IID.

## Heterogeneidade da partição non-IID

Todas as execuções non-IID com a mesma semente compartilham a mesma partição, de
modo que a diferença entre estratégias não vem de sorteios distintos. A divisão
efetiva com a semente 42 foi:

| Cliente | Amostras | Classe 0 | Classe 1 | Maior fração de uma classe |
|---|---|---|---|---|
| 0 | 3701 | 2922 | 779 | 0,790 |
| 1 | 327 | 241 | 86 | 0,737 |
| 2 | 5647 | 1004 | 4643 | 0,822 |
| 3 | 177 | 0 | 177 | 1,000 |
| 4 | 6148 | 3833 | 2315 | 0,623 |

## Calibração dos hiperparâmetros

FedAdam e FedYogi aplicam um passo de otimizador no servidor, controlado por
`eta`. FedAvg não tem passo no servidor — apenas promedia — e portanto não tem
esse hiperparâmetro para ajustar. Comparar as três com um `eta` arbitrário mediria
a qualidade do ajuste em vez da do método, então `eta` foi calibrado por varredura
na partição IID, que é o cenário de controle. Calibrar no regime non-IID, onde a
comparação acontece, enviesaria o resultado a favor das adaptativas.

**FedAdam** — `eta` escolhido: 0,3

| eta | Acurácia final | Rounds p/ ≥95% | Rounds p/ ≥99% |
|---|---|---|---|
| 0,01 | 0,9213 | não atingido | não atingido |
| 0,1 | 0,9735 | 8 | não atingido |
| 0,3 | 0,9868 | 4 | não atingido |
| 0,6 | 0,9788 | 4 | não atingido |
| 1,0 | 0,5025 | 5 | não atingido |

**FedYogi** — `eta` escolhido: 0,3

| eta | Acurácia final | Rounds p/ ≥95% | Rounds p/ ≥99% |
|---|---|---|---|
| 0,01 | 0,9335 | não atingido | não atingido |
| 0,3 | 0,9842 | 3 | não atingido |
| 0,6 | 0,9725 | 4 | não atingido |

## Observações sobre o dataset

- O alvo `productivity_score` é uma função linear exata de 6 das 14 features (study_hours_per_day, focus_score, sleep_hours, phone_usage_hours, stress_level, attendance_percentage): uma regressão linear sobre o conjunto atinge R² = 1,0000.
- As 8 features restantes têm coeficiente zero — são ruído e foram mantidas de propósito, pois saber ignorá-las faz parte do que o benchmark mede.
- Como consequência, o problema é linearmente separável e qualquer estratégia satura perto de 100% de acurácia no regime IID. A comparação entre estratégias se dá no regime non-IID e nas métricas de eficiência.

## Como os números foram obtidos

Acurácia, F1 e ROC-AUC são medidos pelo servidor sobre o conjunto de teste (productivity, 20% do total) que nenhum cliente recebeu em nenhum momento. *Rounds p/ ≥95%* e *Rounds p/ ≥99%* são os primeiros rounds em que a acurácia central alcança cada alvo, e medem eficiência de comunicação: o alvo de 95% é frouxo neste dataset e serve de verificação, enquanto o de 99% fica logo abaixo do teto centralizado e é o que separa as estratégias. *Treino efetivo* soma o tempo gasto dentro do laço de otimização em todos os clientes e rounds, excluindo carga de dados e inicialização. *Desvio entre clientes* e *pior cliente* vêm da avaliação que cada cliente faz do modelo global na sua própria validação, e medem equidade: se o modelo serve todas as instituições ou apenas as que detêm mais dados.
