# Comparativo das estratégias — pisa

> PISA 2025, questionário do estudante: uso de recursos digitais, autocontrole digital, contexto socioeconômico e desempenho em matemática. Alvo: nota acima da mediana global.

Dataset `pisa` · 48 clientes · 10 rounds · 3 épocas locais · sementes 42, 43, 44 · dispositivo `cpu`

Cada célula traz a média de 3 execuções independentes seguida da dispersão entre elas. A dispersão existe porque a agregação dos pesos no servidor não é reproduzível bit a bit: a ordem de chegada das respostas dos clientes varia, e diferenças da ordem de 1e-7 nos pesos se amplificam ao longo dos rounds. **Diferenças entre estratégias menores que a dispersão não são conclusivas.**

## Partição IID (controle)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,7184 ± 0,0006 | 0,7226 ± 0,0011 | 0,7922 ± 0,0004 | 1,0 | 2,3 ± 0,5 | 466,47 ± 41,85 | 0,0143 ± 0,0002 | 0,6907 ± 0,0020 |
| FedProx | 3 | 0,7161 ± 0,0006 | 0,7167 ± 0,0016 | 0,7891 ± 0,0005 | 1,0 | 5,0 | 696,58 ± 10,40 | 0,0143 ± 0,0011 | 0,6848 ± 0,0017 |
| FedAdam | 3 | 0,7142 ± 0,0008 | 0,7015 ± 0,0021 | 0,7881 ± 0,0004 | 1,0 | 4,7 ± 1,2 | 479,18 ± 6,38 | 0,0132 ± 0,0014 | 0,6871 ± 0,0058 |
| FedYogi | 3 | 0,7135 ± 0,0033 | 0,7102 ± 0,0172 | 0,7870 ± 0,0011 | 1,0 | 6,0 ± 0,8 | 498,07 ± 16,06 | 0,0140 ± 0,0012 | 0,6854 ± 0,0049 |
| FedTrimmedAvg | 3 | 0,7184 ± 0,0006 | 0,7227 ± 0,0023 | 0,7923 ± 0,0005 | 1,0 | 2,0 | 498,40 ± 13,93 | 0,0139 ± 0,0005 | 0,6927 ± 0,0017 |

## Partição non-IID (Dirichlet simulado)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | — | — | — | — | — | — | — | — | — |
| FedProx | — | — | — | — | — | — | — | — | — |
| FedAdam | — | — | — | — | — | — | — | — | — |
| FedYogi | — | — | — | — | — | — | — | — | — |
| FedTrimmedAvg | — | — | — | — | — | — | — | — | — |

## Partição natural (uma unidade por país)

| Estratégia | Sementes | Acurácia | F1 | ROC-AUC | Rounds p/ 95% do teto | Rounds p/ 99% do teto | Treino efetivo (s) | Desvio entre clientes | Pior cliente |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 3 | 0,7096 ± 0,0005 | 0,7058 ± 0,0028 | 0,7830 ± 0,0003 | 1,0 | não atingido | 383,39 ± 7,92 | 0,0564 ± 0,0013 | 0,5847 ± 0,0090 |
| FedProx | 3 | 0,7063 ± 0,0004 | 0,7022 ± 0,0017 | 0,7790 ± 0,0004 | 1,0 | não atingido | 559,60 ± 13,16 | 0,0589 ± 0,0017 | 0,5777 ± 0,0091 |
| FedAdam | 3 | 0,7029 ± 0,0051 | 0,6843 ± 0,0131 | 0,7751 ± 0,0033 | 1,3 ± 0,5 | não atingido | 380,50 ± 4,59 | 0,0614 ± 0,0054 | 0,5638 ± 0,0303 |
| FedYogi | 3 | 0,7042 ± 0,0029 | 0,6886 ± 0,0128 | 0,7769 ± 0,0007 | 2,0 | não atingido | 377,72 ± 4,21 | 0,0620 ± 0,0056 | 0,5756 ± 0,0269 |
| FedTrimmedAvg | 3 | 0,7068 ± 0,0009 | 0,6905 ± 0,0034 | 0,7815 ± 0,0009 | 1,0 | não atingido | 381,31 ± 1,04 | 0,0629 ± 0,0014 | 0,5694 ± 0,0084 |

## Leitura dos resultados

- **Alvo de 99% do teto não alcançado de forma consistente:** FedAvg, FedProx, FedAdam, FedYogi, FedTrimmedAvg não chegaram a 99% do teto em todas as sementes dentro dos rounds executados, e por isso ficaram fora da comparação de eficiência. A coluna *Rounds p/ 99% do teto* indica em quantas sementes o alvo foi atingido.
- **Equidade entre instituições:** a dispersão de acurácia entre clientes ficou entre 0,0564 ± 0,0013 (FedAvg) e 0,0629 ± 0,0014 (FedTrimmedAvg), mas as faixas das primeiras colocadas se sobrepõem. Há separação clara apenas entre os extremos, não uma vencedora isolada.
- **Instituição menos favorecida:** o pior cliente teve acurácia entre 0,5638 ± 0,0303 (FedAdam) e 0,5847 ± 0,0090 (FedAvg); nenhuma estratégia protege a instituição minoritária de forma destacada das demais.
- **Controle IID:** no regime homogêneo a diferença de acurácia entre a melhor e a pior estratégia é de 0,0049, confirmando que o pipeline é o mesmo para todas e que a heterogeneidade dos dados, não a implementação, é a origem das diferenças observadas na partição natural por país.

## Execuções pendentes

As combinações a seguir ainda não foram executadas e aparecem como linha vazia nas tabelas acima:

- `fedavg` / `noniid`
- `fedprox` / `noniid`
- `fedadam` / `noniid`
- `fedyogi` / `noniid`
- `fedtrimmedavg` / `noniid`

## Observações sobre o dataset

- Os dados de origem foram extraídos com `user_missing=True`, que preserva os códigos de ausência do SPSS em vez de convertê-los em nulo. Valores como 97 (não aplicável) e 99 (sem resposta) constavam como medições, e `isna()` reportava zero nulos. Sem a limpeza, a correlação de ESCS com o desempenho aparece como -0,150 quando na verdade é +0,404 — sinal invertido.
- A amostra efetiva é de 303.681 dos 755.721 registros, por exigir as 17 features e a nota presentes. A perda não é aleatória: países que não aplicaram partes do questionário saem, o que altera a composição da amostra.
- O alvo é a média das dez plausible values de matemática. Elas não são dez notas, mas dez imputações da mesma proficiência latente; usar a média subestima o erro de medida, o que o procedimento da OCDE trata com análise múltipla.
- O peso amostral `W_FSTUWT` não é utilizado. Os resultados comparam estratégias de agregação federada e não constituem estimativas populacionais nacionais.
- A normalização é global, ajustada no conjunto de treino. Em federated learning estrito, estatísticas globais exigiriam um round prévio de agregação.

## Sensibilidade ao alvo de convergência

A coluna *rounds até o alvo* depende de qual alvo se escolhe, e a ordem das estratégias pode mudar com ele. Uma estratégia que converge rápido para um patamar mais baixo cruza um alvo frouxo antes; outra que converge devagar mas termina mais alto é a única a cruzar um alvo exigente. Nenhuma das duas leituras é errada — elas respondem a perguntas diferentes, e é por isso que dois alvos são reportados em vez de um. Ao citar eficiência de comunicação, o alvo precisa ser citado junto.

## Como os números foram obtidos

Acurácia, F1 e ROC-AUC são medidos pelo servidor sobre o conjunto de teste (pisa, 20% do total) que nenhum cliente recebeu em nenhum momento. *Rounds p/ ≥95%* e *Rounds p/ ≥99%* são os primeiros rounds em que a acurácia central alcança cada alvo, e medem eficiência de comunicação: o alvo de 95% é frouxo neste dataset e serve de verificação, enquanto o de 99% fica logo abaixo do teto centralizado e é o que separa as estratégias. *Treino efetivo* soma o tempo gasto dentro do laço de otimização em todos os clientes e rounds, excluindo carga de dados e inicialização. *Desvio entre clientes* e *pior cliente* vêm da avaliação que cada cliente faz do modelo global na sua própria validação, e medem equidade: se o modelo serve todas as instituições ou apenas as que detêm mais dados.
