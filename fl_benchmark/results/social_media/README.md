# Comparativo das estratégias — social_media

> Questionário sobre uso de redes sociais por estudantes (369 respondentes distintos, 25 features ordinais/one-hot). Alvo: impacto da rede social na produtividade moderado ou significativo.

Dataset `social_media` · 5 clientes · 10 rounds · 3 épocas locais · sementes 42, 43, 44 · dispositivo `cpu`

Cada célula traz a média de 3 execuções independentes seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit: a ordem de chegada das respostas dos clientes varia, e diferenças da ordem de 1e-7 nos pesos se amplificam ao longo dos rounds. **Diferenças entre estratégias menores que a dispersão não são conclusivas.**

## Teto de referência

O mesmo modelo treinado de forma centralizada, com acesso a todos os dados de treino de uma só vez. Nenhuma estratégia federada deve superá-lo de forma significativa — se superasse, haveria vazamento de dados no pipeline.

| | Acurácia | F1 | ROC-AUC | Treino efetivo (s) |
|---|---|---|---|---|
| Centralizado (30 épocas) | 0,8649 | 0,8810 | 0,9111 | 0,72 |

## Partição IID (controle)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,8559 ± 0,0064 | 0,8778 ± 0,0054 | 0,9134 ± 0,0143 | 7,0 ± 1,4 | 8,0 (em 1/3) | 1,63 ± 0,75 | 0,0754 ± 0,0155 | 0,7222 ± 0,0393 |
| FedProx | 3 | 0,8423 ± 0,0127 | 0,8644 ± 0,0152 | 0,9126 ± 0,0152 | 7,0 ± 1,4 | 8,0 (em 1/3) | 1,74 ± 0,47 | 0,0679 ± 0,0050 | 0,7500 |
| FedAdam | 3 | 0,7297 ± 0,0292 | 0,7729 ± 0,0140 | 0,8010 ± 0,0644 | 7,0 (em 1/3) | não atingido | 0,83 ± 0,03 | 0,1164 ± 0,0181 | 0,5556 ± 0,0786 |
| FedYogi | 3 | 0,7252 ± 0,1401 | 0,7203 ± 0,1952 | 0,7524 ± 0,1188 | 7,5 (em 2/3) | não atingido | 1,49 ± 0,39 | 0,0913 ± 0,0360 | 0,6111 ± 0,1416 |
| FedTrimmedAvg | 3 | 0,8649 ± 0,0110 | 0,8867 ± 0,0133 | 0,9175 ± 0,0179 | 6,0 ± 0,8 | 7,5 (em 2/3) | 1,37 ± 0,56 | 0,0875 ± 0,0295 | 0,6944 ± 0,0393 |

## Partição non-IID (Dirichlet simulado)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,8198 ± 0,0278 | 0,8521 ± 0,0206 | 0,8861 ± 0,0214 | 5,0 (em 2/3) | não atingido | 1,03 ± 0,22 | 0,1334 ± 0,0522 | 0,5646 ± 0,1316 |
| FedProx | 3 | 0,8288 ± 0,0278 | 0,8585 ± 0,0211 | 0,8884 ± 0,0214 | 4,5 (em 2/3) | 8,0 (em 1/3) | 1,08 ± 0,12 | 0,1155 ± 0,0382 | 0,6217 ± 0,0919 |
| FedAdam | 3 | 0,7703 ± 0,0292 | 0,8026 ± 0,0287 | 0,8469 ± 0,0296 | 7,0 (em 1/3) | não atingido | 1,00 ± 0,14 | 0,1026 ± 0,0626 | 0,5884 ± 0,1371 |
| FedYogi | 3 | 0,6937 ± 0,0718 | 0,7851 ± 0,0235 | 0,7185 ± 0,1159 | 5,0 (em 1/3) | não atingido | 1,50 ± 0,43 | 0,2433 ± 0,1169 | 0,2794 ± 0,2741 |
| FedTrimmedAvg | 3 | 0,8198 ± 0,0169 | 0,8583 ± 0,0114 | 0,8792 ± 0,0227 | 7,0 (em 2/3) | 5,0 (em 1/3) | 1,67 ± 0,08 | 0,1347 ± 0,0450 | 0,5698 ± 0,1205 |

## Partição natural (uma unidade por país)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | — | — | — | — | — | — | — | — | — |
| FedProx | — | — | — | — | — | — | — | — | — |
| FedAdam | — | — | — | — | — | — | — | — | — |
| FedYogi | — | — | — | — | — | — | — | — | — |
| FedTrimmedAvg | — | — | — | — | — | — | — | — | — |

## Leitura dos resultados

- **Alvo de 99% do teto não alcançado de forma consistente:** FedAvg, FedProx, FedAdam, FedYogi, FedTrimmedAvg não chegaram a 99% do teto em todas as sementes dentro dos rounds executados, e por isso ficaram fora da comparação de eficiência. A coluna *Rounds p/ 99% do teto* indica em quantas sementes o alvo foi atingido.
- **Equidade entre instituições:** a dispersão de acurácia entre clientes ficou entre 0,1026 ± 0,0626 (FedAdam) e 0,2433 ± 0,1169 (FedYogi), mas as faixas das primeiras colocadas se sobrepõem. Há separação clara apenas entre os extremos, não uma vencedora isolada.
- **Instituição menos favorecida:** o pior cliente teve acurácia entre 0,2794 ± 0,2741 (FedYogi) e 0,6217 ± 0,0919 (FedProx); nenhuma estratégia protege a instituição minoritária de forma destacada das demais.
- **Controle IID:** no regime homogêneo a diferença de acurácia entre a melhor e a pior estratégia é de 0,1396, confirmando que o pipeline é o mesmo para todas e que a heterogeneidade dos dados, não a implementação, é a origem das diferenças observadas na partição non-IID.

## Heterogeneidade da partição non-IID

Todas as execuções non-IID com a mesma semente compartilham a mesma partição, de
modo que a diferença entre estratégias não vem de sorteios distintos. A divisão
efetiva com a semente 42 foi:

| Cliente | Amostras | Classe 0 | Classe 1 | Maior fração de uma classe |
|---|---|---|---|---|
| 0 | 74 | 0 | 74 | 1,000 |
| 1 | 66 | 17 | 49 | 0,742 |
| 2 | 70 | 67 | 3 | 0,957 |
| 3 | 41 | 7 | 34 | 0,829 |
| 4 | 44 | 26 | 18 | 0,591 |

## Execuções pendentes

As combinações a seguir ainda não foram executadas e aparecem como linha vazia nas tabelas acima:

- `fedavg` / `natural`
- `fedprox` / `natural`
- `fedadam` / `natural`
- `fedyogi` / `natural`
- `fedtrimmedavg` / `natural`

## Observações sobre o dataset

- O CSV original tem 1.501 linhas, mas 1.130 são duplicatas exatas e 9 são linhas inválidas (texto da pergunta no lugar da resposta). Após a limpeza restam 369 amostras — cerca de 59 por cliente com 5 clientes.
- `AcademicImpact` e `ProductivityWithoutSM` ficaram fora das features: são autoavaliações de consequência, da mesma natureza do alvo, e funcionariam como vazamento parcial da resposta.
- Diferente do `productivity`, o alvo não é derivado das features: uma regressão linear explica R² ≈ 0,56. O teto de acurácia fica bem abaixo de 100%, e os alvos de convergência devem ser lidos relativos ao baseline.
- No regime natural cada faixa etária (<18, 18–24, 25–34) é um cliente: partições de tamanho bem diferente, heterogeneidade real.

## Sensibilidade ao alvo de convergência

A coluna *rounds até o alvo* depende de qual alvo se escolhe, e a ordem das estratégias pode mudar com ele. Uma estratégia que converge rápido para um patamar mais baixo cruza um alvo frouxo antes; outra que converge devagar mas termina mais alto é a única a cruzar um alvo exigente. Nenhuma das duas leituras é errada — elas respondem a perguntas diferentes, e é por isso que dois alvos são reportados em vez de um. Ao citar eficiência de comunicação, o alvo precisa ser citado junto.

## Como os números foram obtidos

Acurácia, F1 e ROC-AUC são medidos pelo servidor sobre o conjunto de teste (social_media, 20% do total) que nenhum cliente recebeu em nenhum momento. *Rounds p/ ≥95%* e *Rounds p/ ≥99%* são os primeiros rounds em que a acurácia central alcança cada alvo, e medem eficiência de comunicação: o alvo de 95% é frouxo neste dataset e serve de verificação, enquanto o de 99% fica logo abaixo do teto centralizado e é o que separa as estratégias. *Treino efetivo* soma o tempo gasto dentro do laço de otimização em todos os clientes e rounds, excluindo carga de dados e inicialização. *Desvio entre clientes* e *pior cliente* vêm da avaliação que cada cliente faz do modelo global na sua própria validação, e medem equidade: se o modelo serve todas as instituições ou apenas as que detêm mais dados.
