# Estratégia para o PISA 2025

Documento de planejamento da etapa 4: levar o benchmark de estratégias federadas
do dataset de produtividade para o PISA 2025, e comparar os dois de
forma defensável.

---

## 1. O que o PISA muda

O benchmark atual (etapa 3) roda sobre um dataset cuja limitação é estrutural —
não nos dados, que são observações reais de estudantes, mas no alvo: o
`productivity_score` é uma coluna derivada por fórmula de 6 das 14 features. Uma
regressão linear atinge R² = 1,0000 e toda estratégia satura acima de 98% de
acurácia. A comparação teve de se apoiar em convergência e equidade, porque a
acurácia não discriminava.

O PISA remove essa limitação e acrescenta algo que não se podia simular:

| | Produtividade (etapa 3) | PISA 2025 (etapa 4) |
|---|---|---|
| Linhas | 20.000 | 303.681 (após limpeza e filtro) |
| R² de regressão linear | 1,0000 (alvo derivado) | 0,2621 |
| ROC-AUC de regressão logística | 1,000 | 0,765 |
| Unidades federadas | 5, inventadas por Dirichlet | **48 países reais** |
| Heterogeneidade de rótulo | simulada (α configurável) | **natural: 0,09 a 0,97** |
| Heterogeneidade de volume | partes iguais, ou Dirichlet | **natural: 1.156 a 25.890** |

A última linha é a mais importante. A fração de alunos acima da mediana mundial
de matemática vai de **0,09 na República Dominicana a 0,97 em QCI**
(Pequim-Xangai-Jiangsu-Zhejiang). Isso é *label skew* severo que não precisa ser
fabricado — e num domínio onde a restrição que justifica Federated Learning é
real: microdados educacionais nacionais não são centralizáveis livremente.

### Decisões tomadas

| Decisão | Escolha |
|---|---|
| Unidade federada | **País** (48 clientes com dados válidos) |
| Features | **Núcleo ST + WLE** — 17 features, 303.681 linhas, 48 países |
| Alvo | **Classificação binária** pela mediana global de matemática |
| Plausible values | **Média das 10**, documentada como aproximação |

---

## 2. Três defeitos nos dados que precisam ser corrigidos antes de qualquer treino

### 2.1 Os códigos de missing não foram convertidos

O `dataset_apenas_colunas_relevantes.py` usa `pyreadstat.read_sav(...,
user_missing=True)`. O parâmetro faz o oposto do que o comentário do script
pretendia: ele **preserva** os códigos de usuário do SPSS em vez de convertê-los
em `NaN`. O resultado é que `97` ("não aplicável"), `98` ("sem resposta") e `99`
("ausente") estão no dataset como se fossem medições.

`df.isna().sum()` reporta zero nulos — e é falso.

O efeito é grave e silencioso. Correlação com a nota de matemática:

| Variável | Sem limpeza | Com limpeza |
|---|---|---|
| ESCS (nível socioeconômico) | **−0,150** | **+0,401** |
| HOMEPOS (bens no domicílio) | −0,079 | +0,348 |
| ICTRES (recursos digitais) | +0,177 | +0,302 |
| PERSEV (perseverança) | −0,166 | — |

Os sinais invertidos contradizem toda a literatura do PISA. A causa é o volume de
códigos: `ICTRES` tem 275.099 registros (36%) com valor 97; `PERSEV`, 212.764;
`GOALSET`, 164.175.

**Correção adotada:** limpeza por **range válido declarado por variável**, não por
lista de códigos. A distinção importa: `ST326*` usa `9` como categoria legítima
("mais de 8 horas por dia"), então remover o código `9` cegamente destruiria
dados reais. Os ranges estão declarados em `datasets.py`.

### 2.2 Dois questionários são opcionais

As colunas `Option_ICTQ` e `Option_PQ` indicam se o país aplicou cada
questionário complementar:

- **Questionário de ICT** (`IC177*`, `IC178*`, `IC400*`): 363.472 alunos (48%)
- **Questionário de pais** (`PA201*`): 171.090 alunos (23%)

Cobertura resultante, exigindo as features completas mais a nota:

| Cenário | Linhas | % do total |
|---|---|---|
| Núcleo ST + WLE + nota | 303.681 | 40,2% |
| + questionário de ICT | ~223.000 | 29,5% |
| + questionário de pais | ~65.000 | 8,6% |

As features de lazer digital (videogames, redes sociais, streaming) estão no
bloco de ICT — justamente as mais comparáveis com o dataset anterior. O custo de
incluí-las é 26% da amostra e menos países, e por isso ficaram fora do núcleo.
Entram, se necessário, numa análise secundária (seção 5).

### 2.3 As dez colunas `PV*MATH` não são dez notas

São *plausible values*: dez imputações da mesma proficiência latente, geradas pelo
modelo de resposta ao item da OCDE. O manual prescreve analisar as dez e combinar
os resultados pela regra de Rubin, o que propaga corretamente o erro de medida.

Usar a média das dez como alvo único — a decisão adotada — é aceitável em ML
aplicado, mas **subestima o erro de medida da proficiência**. Fica registrado
como limitação explícita; rodar o benchmark nas dez multiplicaria por dez o custo
de execução (300 execuções), o que não se justifica aqui.

O dataset também traz `W_FSTUWT`, o peso amostral final. O PISA é uma amostra
estratificada complexa, e estimativas populacionais exigem o peso. **O benchmark
não o usa**: o objetivo é comparar estratégias de agregação federada sob
heterogeneidade, não produzir estimativas nacionais. Usar o peso mudaria a
pergunta. Isso também fica registrado.

---

## 3. Arquitetura: o que muda no código

O registro de datasets em `fl_benchmark/fl_benchmark/datasets.py` foi desenhado
para isso. O PISA entra como uma segunda `DatasetSpec`, e **nenhuma estratégia,
métrica ou relatório muda**. Três pontos exigem extensão real:

**a) Carga de Parquet e limpeza por range.** A `DatasetSpec` atual assume CSV com
colunas numéricas limpas. Ganha três campos: formato do arquivo, dicionário de
ranges válidos por coluna, e política de descarte de linhas incompletas.

**b) Partição por coluna, não por sorteio.** O `partition.py` hoje tem dois
regimes: `iid` (sorteio) e `noniid` (Dirichlet). Entra um terceiro, `natural`,
que agrupa por uma coluna do próprio dataset — `CNT` no caso do PISA. É a
mudança conceitual central: a partição deixa de ser construída e passa a ser
**lida dos dados**.

**c) Número de clientes dirigido pelos dados.** Hoje `num-clients` é configuração
(5). Com partição natural, o número de clientes é o número de países (48), e a
federação precisa de `num-supernodes=48`. Isso muda o custo de execução — ver
seção 6.

```
datasets.py     + DatasetSpec do PISA: Parquet, ranges válidos, alvo derivado
partition.py    + regime "natural": agrupa por coluna (CNT)
pyproject.toml  + dataset="pisa", partition="natural", num-clients=48
model.py        inalterado (17 features em vez de 14 — já é parâmetro)
engine.py       inalterado
metrics.py      inalterado
strategies/*    inalterado
report.py       inalterado
```

---

## 4. Como comparar os dois datasets de forma defensável

Esta é a parte que exige mais cuidado, porque há uma forma errada e tentadora de
fazer: harmonizar features à força e comparar acurácias. Não funciona, e o motivo
é simples — **0,99 no dataset de produtividade e 0,70 no PISA não são
comparáveis**, já que o primeiro tem alvo derivado por fórmula e o segundo mede
proficiência aferida por prova. A
diferença nos números diria respeito aos datasets, não às estratégias.

A comparação se faz em duas camadas, com pesos bem diferentes.

### Camada 1 — Comparação de protocolo (principal)

Rodar **o mesmo protocolo** nos dois datasets e comparar **o comportamento
relativo das estratégias**, não os valores absolutos:

- As cinco estratégias, as mesmas métricas, três sementes, mesmo modelo.
- A pergunta é: **o ranking das estratégias se mantém quando o problema deixa de
  ser trivial e a heterogeneidade passa a ser real?**

O resultado da etapa 3 foi que FedProx foi a única a atingir 99% de acurácia em
todas as sementes sob partição non-IID. A hipótese a testar: *FedProx mantém a
vantagem sob heterogeneidade natural, ou ela era artefato da partição Dirichlet?*

Essa pergunta tem resposta válida e é publicável independentemente de qual seja o
resultado. É esta a comparação que sustenta o trabalho.

### Camada 2 — Harmonização de features (secundária)

Existe sobreposição conceitual real entre os dois datasets, em **quatro eixos**:

| Eixo | Produtividade | PISA 2025 |
|---|---|---|
| Uso digital para lazer | `phone_usage_hours`, `social_media_hours`, `youtube_hours` | `ST326Q04/05/06JA` (lazer na escola, fora, fim de semana) |
| Tempo de estudo | `study_hours_per_day` | `ST296Q04JA` (tempo total de lição de casa) |
| Distração / foco | `focus_score` | `ST097Q06DA` (distração por dispositivos em aula, invertido) |
| Estresse / ansiedade | `stress_level` | `ST322Q07JA` (ansiedade sem o dispositivo) |
| *(alvo)* | `productivity_score` | média dos `PV*MATH` |

**Harmonização por quantis**: cada eixo é convertido em seu percentil *dentro do
próprio dataset*, descartando a escala absoluta. É o único jeito honesto de pôr
"horas contínuas" e "categoria ordinal de 1 a 9" no mesmo eixo.

O que isso permite: um benchmark de 4 features nos dois datasets, com o mesmo
protocolo e escalas comparáveis.

**Três ressalvas que precisam acompanhar esse resultado:**

1. O alinhamento `focus_score` ↔ `ST097Q06DA` é fraco. O primeiro é um escore
   individual de foco; o segundo é a percepção do aluno sobre **a turma** se
   distrair. Medem coisas relacionadas, não a mesma coisa.
2. `stress_level` é estresse geral; `ST322Q07JA` é ansiedade **específica de
   privação do dispositivo**. Conceitos vizinhos, não equivalentes.
3. O PISA não tem equivalente para `sleep_hours`, `exercise_minutes`,
   `attendance_percentage`, `coffee_intake_mg`, `breaks_per_day`,
   `assignments_completed` e `age`; e o dataset de produtividade não tem nada
   correspondente a `ESCS`, `HOMEPOS`, `ICTRES`, `BELONG` e `BULLIED` — que no
   PISA são os preditores mais fortes. **A harmonização descarta justamente o
   que mais prediz.**

Por isso a camada 2 é um **teste de robustez conceitual**, não o resultado
principal. Se ela for apresentada como comparação direta de desempenho, está
sendo apresentada errado.

---

## 5. Análises secundárias que o PISA habilita

Itens que não eram possíveis antes e que ficam disponíveis a baixo custo:

- **Partição por escola** (13.587 unidades): aproxima o cenário de federação de
  muitos clientes pequenos, oposto ao de 48 clientes grandes. Permite testar como
  as estratégias se comportam quando cada cliente tem ~20 amostras.
- **Grupos de países** (ex: por região ou faixa de ESCS): federação de poucos
  clientes muito desiguais, onde o `FedTrimmedAvg` deve mostrar seu pior
  comportamento — o corte descartaria países inteiros.
- **Questionário de ICT** como recorte à parte (223.000 linhas): habilita as
  features de videogames e redes sociais, que tornam a camada 2 bem mais forte.
- **Países como teste de generalização**: treinar federado em N−1 países e
  avaliar no país retido. Mede se o modelo global serve uma instituição que não
  participou do treino — pergunta central em FL aplicado e que o dataset de
  produtividade não permitia formular, por não ter unidades reais.

---

## 6. Custo de execução e o que ele impõe

A etapa 3 levou cerca de 3 minutos por execução, com 5 clientes e 16.000
amostras de treino. O PISA tem **48 clientes e 243.000 amostras de treino** —
quase dez vezes mais dados e dez vezes mais clientes por round.

Isso precisa ser medido antes de comprometer o plano: o protocolo completo da
etapa 3 (5 estratégias × 2 regimes × 3 sementes = 30 execuções) pode passar de
30 horas se cada execução subir para 30 minutos.

**Sequência proposta:**

1. Implementar a ingestão e validar a limpeza (correlações com os sinais certos,
   48 países, cobertura esperada).
2. Rodar **uma** execução de FedAvg para medir o custo real por round.
3. Decidir o escopo com o número na mão. Três reduções possíveis, em ordem de
   preferência:
   - amostrar um subconjunto estratificado de países (ex: 20 em vez de 48,
     cobrindo a amplitude de ESCS e desempenho);
   - reduzir de 10 para 5 rounds, já que com mais dados por cliente a
     convergência tende a ser mais rápida;
   - amostrar alunos dentro de cada país, preservando a proporção.

Qualquer redução precisa ser registrada no relatório — um benchmark que amostra
em silêncio sugere cobertura que não tem.

---

## 7. Ordem de execução

| Passo | Entrega | Risco |
|---|---|---|
| 1 | `DatasetSpec` do PISA: Parquet, ranges válidos, alvo pela mediana | baixo — já validado nesta análise |
| 2 | Regime de partição `natural` por coluna | baixo |
| 3 | Verificação: correlações corretas, 48 países, skew natural documentado | — |
| 4 | Baseline centralizado no PISA | baixo |
| 5 | Uma execução de FedAvg para medir custo | **decide o escopo** |
| 6 | Benchmark completo, escopo definido no passo 5 | custo |
| 7 | Comparativo entre datasets (camada 1) | — |
| 8 | Harmonização por quantis (camada 2), se houver tempo | baixo valor relativo |

Os passos 1 a 4 não dependem de nenhuma decisão pendente e podem ser executados
em sequência. O passo 5 é um ponto de parada deliberado: o escopo do benchmark
completo só deve ser fixado depois de conhecer o custo real por execução.

---

## 8. Limitações a declarar no relatório final

1. O alvo é a média das dez *plausible values*, o que subestima o erro de medida
   da proficiência. O procedimento da OCDE prescreve análise múltipla.
2. O peso amostral `W_FSTUWT` não é usado. Os resultados não são estimativas
   populacionais nacionais, e sim comparações entre estratégias de agregação.
3. A amostra efetiva (303.681 de 755.721) resulta de exigir as 17 features e a
   nota presentes. A perda não é aleatória: países que não aplicaram partes do
   questionário saem, o que enviesa a composição.
4. A comparação com o dataset de produtividade é de **protocolo**, não de
   desempenho absoluto. Os números de acurácia dos dois não são comparáveis entre
   si.
