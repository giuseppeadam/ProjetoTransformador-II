# ProjetoTransformador II

Classificação de desempenho e hábitos digitais de estudantes, desenvolvida em
quatro etapas que compartilham o mesmo problema e mudam a forma de resolvê-lo.

| Etapa | O que faz | Onde está |
|---|---|---|
| 1 — Modelos clássicos | Bagging, Gradient Boosting, Decision Tree, Random Forest, SVM, KNN e Voting sobre o dataset de vício digital | `dataset.ipynb` |
| 2 — Federated Learning | Primeira implementação federada com Flower, uma estratégia selecionável por configuração | `fl_addiction/` |
| 3 — Benchmark de estratégias | Cinco estratégias federadas comparadas em dois regimes de partição, com métricas e relatórios por estratégia | `fl_benchmark/` |
| 4 — Heterogeneidade real | O mesmo benchmark sobre o PISA 2025: 303.681 alunos e **48 países como clientes reais**, em vez de partições sorteadas | `fl_benchmark/` |
| 5 — Questionário de redes sociais | O mesmo benchmark sobre um questionário de impacto das redes sociais nos estudos: **369 respondentes**, alvo não derivado das features | `fl_benchmark/` |

Os comparativos ficam em [`fl_benchmark/results/`](fl_benchmark/results/), um por
dataset, cada um com o relatório detalhado das cinco estratégias ao lado.

---

## Overview dos resultados

Noventa execuções: cinco estratégias × dois regimes de partição × três sementes,
em três datasets.

### Os três datasets

| | Produtividade (etapa 3) | PISA 2025 (etapa 4) | Redes sociais (etapa 5) |
|---|---|---|---|
| Linhas | 20.000 | 303.681 | 369 (de 1.501) |
| Features | 14 | 17 | 25 |
| Clientes | 5, por sorteio ou Dirichlet | **48 países reais** | 5, por sorteio ou Dirichlet |
| Teto centralizado | 0,9932 | **0,7190** | 0,8649 |
| Heterogeneidade de rótulo | simulada por Dirichlet | **natural: 0,09 a 0,97** | simulada por Dirichlet |
| Tamanho dos clientes | partes iguais | **925 a 20.712** | ~59 (IID) / 41 a 90 (non-IID) |

O teto de 0,9932 no primeiro dataset não é mérito do modelo: o
`productivity_score` é uma coluna derivada por fórmula de 6 das 14 features, o
que torna o problema linearmente separável. No PISA o alvo é proficiência em
matemática aferida por prova, e o teto cai para 0,7190 — um problema de
dificuldade genuína.

### Pré-processamento do PISA

O PISA passa por um passo obrigatório — `preparar_pisa.py`, executado uma vez —
que faz três coisas, cada uma por um motivo concreto:

| O que | Por quê |
|---|---|
| **Converte os códigos de ausência em nulo** e descarta as linhas incompletas (755.721 → 303.681) | O arquivo de origem foi extraído com `user_missing=True`, que **preserva** os códigos do SPSS em vez de convertê-los. Valores como 97 e 99 constavam como medições e `isna()` reportava zero nulos. Sem isso, a correlação de `ESCS` com desempenho aparece como **−0,150** em vez de **+0,404** — sinal invertido. |
| **Deriva o alvo e normaliza** (média das dez *plausible values*, binarizada pela mediana global) | Cálculo caro, feito uma vez em vez de 48 vezes por round. |
| **Grava um Parquet por país** | Na simulação cada cliente é um processo com memória própria. Sem isso, os 48 clientes carregariam os 755 mil registros brutos e repetiriam a limpeza a cada round. A carga cai de ~4 s para **0,31 s**, e 90 MB viram 11 MB. |

A limpeza é por **range válido declarado por variável**, não por lista de
códigos: em `ST326*` o valor 9 é uma categoria legítima ("mais de 8 horas por
dia"), e descartá-lo cegamente destruiria dados reais. Detalhes em
[Sobre o dataset da etapa 4](#sobre-o-dataset-da-etapa-4-pisa-2025).

### O que a heterogeneidade real faz

Comparando o regime heterogêneo contra o controle IID **dentro do PISA**:

| Métrica | IID (controle) | Natural (48 países) | |
|---|---|---|---|
| Acurácia | 0,7135–0,7184 | 0,7029–0,7096 | −1 ponto |
| **Desvio entre clientes** | 0,0132–0,0143 | **0,0564–0,0629** | **4× pior** |
| **Pior cliente** | 0,6848–0,6927 | **0,5638–0,5847** | **−11 pontos** |
| Atingiram 99% do teto | **5 de 5** | **0 de 5** | — |

A heterogeneidade natural quase não custa acurácia média, mas **destrói a
equidade**: o país menos favorecido cai para 0,58 — perto do chute em um problema
binário — e nenhuma estratégia alcança 99% do teto em 10 rounds.

### Ranking no PISA, partição natural

| # | Acurácia | ROC-AUC | Pior cliente | Desvio entre clientes |
|---|---|---|---|---|
| 1 | **FedAvg** 0,7096 | **FedAvg** 0,7830 | FedAvg 0,5847 | FedAvg 0,0564 |
| 2 | FedTrimmedAvg 0,7068 | FedTrimmedAvg 0,7815 | FedProx 0,5777 | FedProx 0,0589 |
| 3 | FedProx 0,7063 | FedProx 0,7790 | FedYogi 0,5756 | FedAdam 0,0614 |
| 4 | FedYogi 0,7042 | FedYogi 0,7769 | FedTrimmedAvg 0,5694 | FedYogi 0,0620 |
| 5 | FedAdam 0,7029 | FedAdam 0,7751 | FedAdam 0,5638 | FedTrimmedAvg 0,0629 |
| | *separação clara* | *separação clara* | *faixas se sobrepõem* | *faixas se sobrepõem* |

No dataset de produtividade, sob Dirichlet, o FedProx liderava. No PISA ele cai
para terceiro, e **o FedAvg — média ponderada pura, sem nenhum mecanismo contra
heterogeneidade — lidera as quatro métricas**.

### Quais features importam

Peso relativo dos coeficientes padronizados de uma regressão logística:

| Eixo | Produtividade | PISA | |
|---|---|---|---|
| Uso digital para lazer | 15,2% | **24,0%** | transferiu |
| Estresse / ansiedade | 8,6% | 9,8% | transferiu |
| Distração / foco | **18,5%** | 2,4% | colapsou |
| Tempo de estudo | **33,6%** | 1,4% | colapsou |

E as dimensões que só o PISA tem somam **62,3% do poder preditivo**: contexto
socioeconômico (38,4%, com o `ESCS` sozinho em 25%), autocontrole digital
(19,7%) e ambiente escolar (3,0%).

A feature em segundo lugar é a mais interessante: `ST322Q01JA` — *"desligo as
notificações do celular durante a aula"*, com 15,9%. Não é **quanto** o aluno usa
o dispositivo, é se ele consegue **controlar** o uso.

### Redes sociais: um dataset pequeno

| Estratégia | Acurácia IID | Acurácia non-IID | Pior cliente non-IID |
|---|---|---|---|
| FedAvg | 0,8559 | 0,8198 | 0,5646 |
| FedProx | 0,8423 | **0,8288** | **0,6217** |
| FedAdam | 0,7297 | 0,7703 | 0,5884 |
| FedYogi | 0,7252 | 0,6937 | 0,2794 |
| FedTrimmedAvg | **0,8649** | 0,8198 | 0,5698 |

FedAvg, FedProx e FedTrimmedAvg chegam perto do teto no IID e perdem de 1 a 5
pontos sob Dirichlet, com faixas que se sobrepõem. **FedAdam e FedYogi ficam
cerca de 13 pontos abaixo já no controle IID**: o `eta = 0,3` foi calibrado no dataset de
produtividade e não transfere — ver [Calibração do eta](#calibração-do-eta).

O teste central tem 74 amostras, então cada acerto vale 1,35 ponto. Diferenças
dessa ordem entre estratégias não são conclusivas.

### Resumo

O PISA converteu um benchmark que media eficiência de comunicação em um que mede
capacidade real de aprendizado sob heterogeneidade. Três descobertas sustentam o
trabalho. Primeira: a vantagem do FedProx era artefato da partição Dirichlet —
sob heterogeneidade natural ele perde para o FedAvg, e nenhuma das quatro
estratégias sofisticadas supera a média ponderada simples. Segunda: a
heterogeneidade real custa pouca acurácia média, mas quadruplica a desigualdade
entre instituições e derruba o pior cliente a 0,58, o que expõe equidade — não
acurácia — como o problema aberto de federated learning aplicado. Terceira: o que
prediz desempenho é o contexto socioeconômico e a autorregulação digital, não o
tempo declarado de estudo, cujo peso colapsou de 33,6% para 1,4% ao sair de um
alvo derivado por fórmula para uma proficiência aferida por prova.

---

## Glossário

- **Cliente (ClientApp)**: representa uma instituição. Recebe os pesos globais,
  treina localmente por algumas épocas na sua própria partição de dados, e
  devolve os pesos atualizados, sem nunca enviar os dados.
- **Servidor (ServerApp)**: coordena os rounds e agrega os pesos recebidos de
  todos os clientes num novo modelo global, usando uma **estratégia**.
- **Round**: um ciclo completo de enviar pesos → treinar localmente → agregar.
  É a unidade de comunicação do Federated Learning, e o recurso caro: cada round
  exige uma troca de pesos com todas as instituições.
- **Estratégia**: a regra de agregação aplicada pelo servidor sobre os pesos
  recebidos. É o que este benchmark compara — ver
  [As cinco estratégias](#as-cinco-estratégias).
- **Partição IID / non-IID**: como os dados são divididos entre as instituições.
  No regime IID todas recebem distribuições semelhantes; no non-IID algumas
  concentram certas classes, o que faz os modelos locais divergirem entre si —
  o fenômeno chamado *client drift*.

## Estrutura do repositório

```
ProjetoTransformador-II/
├── archive/
│   ├── student_productivity_distraction_dataset_20000.csv   dataset da etapa 3
│   └── Social_Media_Impact_OnStudents_numeric.csv           dataset da etapa 5
├── dataset_novo_indice.csv - ....csv                        dataset das etapas 1 e 2
├── fl_addiction/                                            etapa 2
└── fl_benchmark/                                            etapa 3
    ├── pyproject.toml          configuração do app Flower e hiperparâmetros
    ├── baseline.py             teto de referência centralizado
    ├── run_all.py              executa todas as combinações em sequência
    ├── make_summary.py         consolida os resultados no comparativo
    ├── fl_benchmark/
    │   ├── datasets.py         registro de datasets, carga e normalização
    │   ├── partition.py        partição IID e non-IID (Dirichlet)
    │   ├── model.py            a rede neural, idêntica em todas as execuções
    │   ├── engine.py           treino local e avaliação
    │   ├── metrics.py          agregação das métricas dos clientes
    │   ├── report.py           gravação dos relatórios
    │   ├── client_app.py       a instituição simulada
    │   ├── server_app.py       coordenação dos rounds e instrumentação
    │   └── strategies/         um arquivo por estratégia federada
    │       ├── fedavg.py
    │       ├── fedprox.py
    │       ├── fedadam.py
    │       ├── fedyogi.py
    │       └── fedtrimmedavg.py
    ├── calibrate_eta.py         varredura do eta das estratégias adaptativas
    └── results/
        ├── README.md            comparativo das cinco estratégias
        ├── FEDAVG.md            um relatório por estratégia
        ├── FEDPROX.md
        ├── FEDADAM.md
        ├── FEDYOGI.md
        ├── FEDTRIMMEDAVG.md
        ├── baseline.json        teto de referência centralizado
        ├── calibration_eta_*.json       varreduras de hiperparâmetro
        ├── <estrategia>_<particao>_seed<N>.json
        └── logs/
            ├── calibracao/              execuções da varredura de eta
            ├── verificacoes/            execuções dos testes de reprodutibilidade
            └── antes-correcao-determinismo/
```

O código de cada estratégia vive em seu próprio arquivo sob
`fl_benchmark/fl_benchmark/strategies/`, contendo apenas a construção do método
e a explicação de como ele funciona. Todo o resto — dados, partição, modelo,
treino, métricas — é compartilhado, e é justamente isso que garante que a
comparação meça a forma de agregar os pesos e não uma diferença acidental de
configuração entre uma execução e outra.

---

## Como executar

### Requisitos

```bash
pip install "flwr[simulation]>=1.28.0" "torch>=2.5.0" "pandas>=2.0.0" "scikit-learn>=1.5.0"
```

No Windows, defina a codificação de saída antes de executar. O Flower emite
caracteres que o console não cobre por padrão, e a execução falharia na primeira
linha de log:

```bash
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
```

O backend de simulação depende do `ray`, que não instala no Python 3.13 do
Windows. Use um ambiente com Python 3.12:

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\python -m pip install "flwr[simulation]" torch pandas scikit-learn pyarrow
```

O ambiente pode ficar dentro de `fl_benchmark/`: o `.gitignore` da pasta impede
que o Flower o empacote junto com a aplicação. Se um `flwr run` anterior foi
feito com outro Python, encerre os processos `flower-superlink` e
`flower-superexec` que ficaram em segundo plano — senão a simulação continua
usando o interpretador antigo. Nas versões recentes do Flower a federação local
se chama `local`, e o `run_all.py` recebe `--federation local`.

### Teto de referência

Antes de comparar estratégias, vale estabelecer o teto: o mesmo modelo treinado
de forma centralizada, com acesso a todos os dados de uma vez.

```bash
cd fl_benchmark
python baseline.py --dataset productivity
python baseline.py --dataset pisa --epochs 30
python baseline.py --dataset social_media --epochs 30
```

O teto é gravado em `results/<dataset>/baseline.json` e serve de referência para
os alvos de convergência, que são expressos como **fração do teto** e não como
acurácia absoluta. O motivo: 0,99 é um alvo exigente onde o teto é 0,9932 e
inalcançável onde o teto é 0,7190.

O PISA exige um passo de preparação antes do primeiro treino, executado uma vez.
Ele converte os códigos de ausência do SPSS em nulo — sem isso o sinal das
features inverte — e grava um Parquet por país, para que cada cliente leia apenas
os seus dados:

```bash
python preparar_pisa.py
```

Roda em ~4 s e produz `archive/pisa_preparado/`. Ver
[Pré-processamento do PISA](#pré-processamento-do-pisa).

### Uma estratégia por vez

Cada estratégia é executada individualmente, trocando apenas o valor de
`strategy`. O regime de partição é escolhido em `partition`.

```bash
cd fl_benchmark

# --- Dataset de produtividade: 5 clientes ---
# Partição IID — o cenário de controle
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg"        partition="iid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox"       partition="iid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam"       partition="iid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi"       partition="iid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="iid"'

# Partição non-IID — onde as estratégias se diferenciam
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedavg"        partition="noniid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox"       partition="noniid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedadam"       partition="noniid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedyogi"       partition="noniid"'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedtrimmedavg" partition="noniid"'

# --- PISA: 48 paises como clientes ---
# `num-clients` precisa acompanhar `num-supernodes`: o primeiro monta a partição
# no servidor, o segundo dimensiona a federação. Passar só um faz a partição
# discordar do número real de clientes, e a execução falha.
flwr run . --federation-config num-supernodes=48 --run-config 'strategy="fedavg" partition="natural" dataset="pisa" num-clients=48'
```

Para o lote completo, `run_all.py` cuida desses parâmetros:

```bash
python run_all.py --dataset pisa --num-clients 48 --partitions natural iid
python run_all.py --dataset social_media --partitions iid noniid
```

E `run_noturno.py` executa dentro de um limite de tempo, medindo as primeiras
execuções para estimar quantas cabem e parando antes de ser interrompido no meio:

```bash
python run_noturno.py --limite-horas 7 --dataset pisa --num-clients 48
```

Cada execução grava dois artefatos em `results/`: um JSON por semente, com a
curva completa por round, e um Markdown com o resumo da estratégia. Repetir uma
execução com a mesma semente atualiza os mesmos arquivos, em vez de duplicá-los.

### Repetições com sementes diferentes

Os comandos acima usam a semente padrão (42). Para que os resultados sejam
comparáveis entre estratégias, cada configuração é executada três vezes:

```bash
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=42'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=43'
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox" partition="noniid" seed=44'
```

O motivo está em **Reprodutibilidade**, mais abaixo: a agregação dos pesos no
servidor não é reproduzível bit a bit, e diferenças entre estratégias menores que
a variação entre execuções não seriam conclusivas. Os relatórios reportam média
e dispersão entre as sementes.

### Todas em sequência

```bash
cd fl_benchmark
python run_all.py
```

Executa as 5 estratégias × 2 partições × 3 sementes — 30 execuções, cerca de 90
minutos nesta máquina. São feitas uma por vez, nunca em paralelo: *tempo de
treino* é uma das métricas comparadas, e processos concorrentes disputariam CPU
entre si, contaminando a medição.

O script aceita recortes:

```bash
python run_all.py --strategies fedavg fedprox --partitions noniid
python run_all.py --seeds 42                      # uma semente só, para um teste rápido
python run_all.py --rounds 20
```

O sucesso de cada execução é julgado pelo artefato gravado, não pelo código de
saída do processo: o `flwr run` pode encerrar com código 0 mesmo quando o backend
de simulação falha na inicialização, e confiar no código de saída faria uma
execução perdida ser contada como concluída. O artefato é procurado em
`results/<dataset>/`, onde o servidor grava.

Os logs vão para `results/logs/<estrategia>_<particao>_seed<N>.log`, sem o nome
do dataset. Rodar um dataset novo sobrescreve os logs do anterior; os do
`social_media` foram movidos para `results/logs/social_media/`.

### Consolidar o comparativo

```bash
cd fl_benchmark
python make_summary.py
```

Gera `results/README.md` com as cinco estratégias lado a lado nos dois regimes.
Combinações ainda não executadas aparecem como linha vazia, em vez de
desaparecer da tabela.

### Alterar hiperparâmetros sem editar arquivos

Qualquer chave de `[tool.flwr.app.config]` no `pyproject.toml` pode ser
sobrescrita na linha de comando:

```bash
flwr run . --federation-config num-supernodes=5 \
  --run-config 'strategy="fedprox" partition="noniid" proximal-mu=0.1 num-server-rounds=20'

# alpha menor concentra mais as classes e agrava a heterogeneidade
flwr run . --federation-config num-supernodes=5 \
  --run-config 'strategy="fedavg" partition="noniid" dirichlet-alpha=0.1'
```

---

## O que é medido

| Métrica | O que mede | Por que está aqui |
|---|---|---|
| **Acurácia** | Fração de acertos no conjunto de teste centralizado | Comparável com os modelos clássicos da etapa 1 |
| **F1-score** | Média harmônica entre precisão e recall | A partição non-IID desbalanceia as classes dentro de cada cliente; a acurácia sozinha esconde o caso em que o modelo acerta a classe majoritária e erra a minoritária |
| **ROC-AUC** | Qualidade da ordenação das probabilidades, independente do limiar de 0,5 | Continua discriminando modelos quando a acurácia satura perto de 100%, o que acontece neste dataset |
| **Rounds até ≥95% e ≥99%** | Primeiro round em que a acurácia central alcança cada alvo | Eficiência de comunicação. Em Federated Learning real cada round é uma troca de pesos com todas as instituições, e é esse o recurso caro — duas estratégias que terminam na mesma acurácia não são equivalentes se uma precisou de dois rounds e a outra de nove. São dois alvos porque, neste dataset, 95% é alcançado no primeiro round por todas as estratégias e só serve de verificação; o alvo de 99% fica logo abaixo do teto centralizado e é o que de fato separa uma estratégia da outra |
| **Tempo de treino efetivo** | Soma do tempo gasto dentro do laço de otimização, em todos os clientes e rounds | Custo computacional do aprendizado, sem contar carga de dados nem inicialização da simulação |
| **Desvio entre clientes** | Dispersão da acurácia que cada cliente mede na própria validação | Equidade: indica se o modelo global serve bem todas as instituições ou apenas as que detêm mais dados. Sob partição non-IID é frequentemente a diferença mais informativa entre estratégias |

Acurácia, F1 e ROC-AUC são medidos pelo servidor sobre os 20% do dataset
reservados antes de qualquer partição, que nenhum cliente recebe em momento
algum.

Todas as métricas são reportadas como **média ± dispersão entre as três
sementes**. A dispersão não é decoração: é o que permite separar diferença de
método de variação de execução. Uma diferença entre estratégias menor que a
dispersão não sustenta conclusão, e a seção *Leitura dos resultados* do
comparativo aplica esse teste automaticamente — quando as faixas das primeiras
colocadas se sobrepõem, ela diz isso em vez de apontar uma vencedora.

### Reprodutibilidade

Com a mesma semente, a partição dos dados, a inicialização do modelo e a ordem
dos batches em cada cliente são idênticas. Ainda assim, duas execuções da mesma
configuração não produzem métricas exatamente iguais.

A causa é a agregação dos pesos no servidor: a ordem de chegada das respostas dos
clientes varia entre execuções, e a soma de ponto flutuante não é associativa.
As diferenças resultantes são da ordem de 1e-7 por round, mas se amplificam ao
longo dos 10 rounds — e num problema cuja fronteira de decisão é apertada, basta
reclassificar cerca de uma dúzia das 4.000 amostras de teste para mover a
acurácia em 0,003. Isso é interno ao framework e não se resolve do lado da
aplicação; a saída é repetir com várias sementes, que é o que o benchmark faz.

Uma segunda fonte, esta sim corrigida: a ordem dos batches no treino local. O
`DataLoader` do PyTorch consome o RNG global do processo, e numa simulação cada
cliente roda em um processo próprio que não herda a semente definida no servidor.
O resultado era uma ordem de batches diferente a cada execução. A correção foi
derivar uma semente de `(semente, cliente, round)` — o número do round vem
injetado pela estratégia em cada mensagem de treino — e passá-la a um
`torch.Generator` explícito. Isso reduziu a dispersão da acurácia de ±0,003 para
±0,0001.

Uma terceira, também corrigida, na partição non-IID. Quando o sorteio de
Dirichlet deixa um cliente com menos de 40 amostras, ele é refeito — e a nova
tentativa usava `seed + tentativa`. Sementes vizinhas colidiam: a 42 na
tentativa 1 usava o mesmo gerador que a 43 na tentativa 0. **No dataset de
produtividade, as sementes 42 e 43 do non-IID compartilham a mesma partição**; no
de redes sociais, que exige dezenas de tentativas, as três coincidiam. A
correção usa `default_rng([seed, tentativa])` a partir da segunda tentativa,
preservando as partições que já saíam na primeira.

### Treino efetivo e wall-clock

Os relatórios registram os dois tempos, e eles são bem diferentes. O
*wall-clock* de uma execução inclui cerca de dois minutos e meio de custo fixo
de inicialização do backend de simulação — construção do pacote da aplicação,
criação do ambiente e subida dos processos dos clientes — que não tem relação
com o método federado sendo avaliado. Para comparar estratégias entre si, a
métrica válida é o **treino efetivo**.

---

## Os três regimes de partição

As instituições simuladas recebem fatias diferentes do conjunto de treino, e a
forma de fatiar muda completamente a dificuldade do problema federado.

**`iid`** — cada cliente recebe uma amostra aleatória de tamanho igual. A
distribuição de classes é praticamente idêntica entre clientes, então o modelo
treinado localmente por cada um já é próximo do modelo global ideal. É o cenário
de controle: se as cinco estratégias não empatam aqui, há erro de implementação
em alguma delas.

**`noniid`** — o desbalanceamento de rótulos é governado por uma distribuição de
Dirichlet. Para cada classe, a proporção que vai a cada cliente é sorteada de
`Dirichlet(alpha)`: com `alpha` alto o resultado tende ao IID, com `alpha` baixo
alguns clientes ficam quase com uma única classe. Com `alpha = 0,3` e cinco
clientes, a partição efetiva vai de 177 a 6.148 amostras por cliente, e um deles
recebe apenas uma das duas classes.

**`natural`** — a partição é **lida dos dados**, não construída. Cada valor
distinto de uma coluna do dataset define um cliente; no PISA é o país (`CNT`),
e são 48. A heterogeneidade resultante não é parametrizada por `alpha`: ela
simplesmente é o que os dados trazem. A fração de alunos acima da mediana
mundial vai de 0,09 na República Dominicana a 0,97 em QCI
(Pequim-Xangai-Jiangsu-Zhejiang), e o tamanho dos clientes de 925 a 20.712
alunos — *label skew* e *quantity skew* severos, sem precisar fabricar nenhum
dos dois.

É o regime mais fiel ao caso de uso, porque a impossibilidade de centralizar
microdados educacionais nacionais é justamente o que justifica a federação.

Os regimes heterogêneos são os que separam as estratégias. FedProx, FedAdam e
FedYogi existem precisamente para lidar com o *client drift* — a divergência
entre os modelos locais quando as distribuições diferem — e não têm vantagem
alguma a demonstrar num cenário homogêneo.

A diferença entre `noniid` e `natural` acabou sendo o achado central do projeto:
o FedProx lidera sob Dirichlet e perde para o FedAvg sob heterogeneidade real.
Partição simulada e partição real não são intercambiáveis.

A semente é fixa em todos os regimes, de modo que todas as estratégias são
avaliadas exatamente sobre a mesma divisão de dados. No regime `natural` a
semente não afeta a partição — ela vem dos dados — apenas a ordem dos batches e
a divisão interna de validação de cada cliente.

---

## Sobre o dataset da etapa 3

`archive/student_productivity_distraction_dataset_20000.csv` tem 20.000 linhas,
18 colunas e nenhum valor ausente. O alvo é `productivity_score`, binarizado
pela mediana — o que dá classes exatamente equilibradas, 10.000 para cada lado,
sem escolher um limiar arbitrário.

Uma característica precisa ser registrada para que os resultados não sejam mal
interpretados. As features são observações de estudantes, mas o alvo não é uma
medição: **`productivity_score` é uma coluna derivada por fórmula a partir de 6
das 14 features**. Uma regressão linear sobre o conjunto atinge R² = 1,0000, com
os seguintes coeficientes padronizados:

```
study_hours_per_day  +11,84    focus_score            +6,55
sleep_hours           +5,49    phone_usage_hours      -5,35
stress_level          -3,09    attendance_percentage  +2,82
```

As outras 8 features não entram na fórmula do alvo e têm coeficiente exatamente
zero no modelo. Foram mantidas de propósito, porque aprender a ignorá-las faz
parte do que o benchmark mede.

A consequência prática é que o problema é linearmente separável e qualquer
modelo razoável passa de 98% de acurácia. **A acurácia alta é propriedade de como
o alvo foi construído, não mérito do modelo nem da estratégia federada.** É por isso que o
benchmark reporta rounds até a convergência e equidade entre clientes: num
problema em que todas as estratégias saturam, são essas as métricas que ainda
distinguem uma da outra.

O efeito é visível nos resultados: as cinco estratégias terminam com acurácia
praticamente idêntica nos dois regimes, enquanto o desvio entre clientes na
partição non-IID chega a ser uma ordem de grandeza maior que no IID. A pergunta
que o benchmark responde neste dataset não é *qual estratégia aprende melhor*,
mas *qual distribui o aprendizado de forma mais equilibrada entre as
instituições e com menos rounds de comunicação*.

Essa limitação foi superada na etapa 4: o PISA 2025 tem alvo aferido por prova,
e as métricas de acurácia voltam a discriminar por si mesmas.

## Sobre o dataset da etapa 4 — PISA 2025

`archive/pisa_preparado/` é produzido por `fl_benchmark/preparar_pisa.py` a
partir dos microdados do estudante do PISA 2025: 755.721 alunos e 90 países na
origem, 303.681 alunos e 48 países após a limpeza. O alvo é a nota de matemática
acima da mediana global.

O script roda **uma vez** e materializa um arquivo Parquet por país. Isso não é
conveniência: na simulação cada cliente é um processo próprio com memória
própria, e sem a preparação os 48 clientes carregariam os 755 mil registros
brutos e repetiriam toda a limpeza a cada round. Com a preparação, a carga cai de
~4 s para 0,31 s e o conjunto ocupa 11 MB em vez de 90 MB. O layout também
espelha a realidade do federated learning: cada instituição possui um arquivo com
os seus dados e apenas os seus.

Três tratamentos acontecem ali, e o primeiro é indispensável.

**Os códigos de ausência do SPSS não vinham convertidos.** O arquivo de origem
foi extraído com `pyreadstat.read_sav(..., user_missing=True)`, parâmetro que
**preserva** os códigos em vez de convertê-los em nulo. Valores como `97` (não
aplicável) e `99` (sem resposta) estavam gravados como se fossem medições, e
`isna()` reportava zero nulos — falso. O efeito é grave e silencioso:

| Variável | Sem limpeza | Com limpeza |
|---|---|---|
| `ESCS` (nível socioeconômico) | **−0,150** | **+0,404** |
| `HOMEPOS` (bens no domicílio) | −0,079 | +0,348 |
| `ICTRES` (recursos digitais) | +0,177 | +0,302 |

Os sinais invertidos contradizem a literatura do PISA. A causa é o volume: o
`ICTRES` tinha 275.099 registros (36%) com valor 97. A limpeza é feita por
**range válido declarado por variável**, não por lista de códigos — a distinção
importa, porque em `ST326*` o valor 9 é uma categoria legítima ("mais de 8 horas
por dia"), e descartá-lo cegamente destruiria dados reais.

**Dois questionários são opcionais.** As colunas `Option_ICTQ` e `Option_PQ`
indicam se o país os aplicou: o de ICT alcançou 48% dos alunos e o de pais 23%.
Exigir as features do bloco de ICT custaria 26% da amostra, e por isso ele ficou
fora do núcleo de 17 features — junto com as perguntas sobre videogames e redes
sociais, que eram as mais comparáveis com o dataset da etapa 3.

**As dez colunas `PV*MATH` não são dez notas**, mas dez imputações da mesma
proficiência latente. O alvo usa a média, o que subestima o erro de medida; o
procedimento da OCDE prescreve análise múltipla pela regra de Rubin. O peso
amostral `W_FSTUWT` não é utilizado, porque o objetivo é comparar estratégias de
agregação e não produzir estimativas populacionais nacionais.

## Sobre o dataset da etapa 5 — redes sociais

`archive/Social_Media_Impact_OnStudents_numeric.csv` é um questionário sobre o
uso de redes sociais durante os estudos, já codificado em números: escalas
Likert como ordinais e gênero e plataforma em one-hot. A limpeza é feita na
carga, pela função `preprocess` declarada no registro:

| O que | Por quê |
|---|---|
| **Remove 1.130 linhas duplicadas** (1.501 → 369) | São cópias exatas de outras respostas. Sem isso, a mesma resposta cai no treino e no teste, e a acurácia passa a medir memorização. |
| **Remove 9 linhas inválidas** | Em 6 delas o campo `UsageLimitation` traz o texto da própria pergunta ("Q15. How often…"); nas outras 3 está vazio. O valor numérico ali era imputado. |

O dataset não tem uma coluna de vício. O alvo é `ProductivityImpact ≥ 2` — o
estudante declara que as redes sociais prejudicam sua produtividade de forma
moderada ou significativa —, o que dá 60% de positivos. O limiar é explícito
(`positive_threshold`), e não pela mediana, porque numa escala de quatro níveis
o corte "acima da mediana" deixaria 27% de positivos.

`AcademicImpact` e `ProductivityWithoutSM` ficaram fora das 25 features: são
autoavaliações de consequência, da mesma natureza do alvo, e funcionariam como
vazamento parcial da resposta. Diferente do dataset de produtividade, **o alvo
não é derivado das features** — uma regressão linear sobre as 25 explica
R² = 0,54.

A base é pequena: com 5 clientes, cada um treina com cerca de 47 amostras e
valida com 12. Mais rounds não ajudam — com 30 ou 50 o non-IID tende a piorar
por overfitting —, e por isso o protocolo de 10 rounds foi mantido.

O registro declara `AgeGroup` como `group_column`, o que habilita o regime
`natural` com as três faixas etárias como clientes (50, 174 e 71 amostras). Ele
ainda não foi executado:

```bash
python run_all.py --dataset social_media --partitions natural --num-clients 3
```

### Adicionar um novo dataset

O registro em `fl_benchmark/fl_benchmark/datasets.py` existe para isso. Coloque
o CSV em `archive/` e acrescente uma entrada:

```python
NOVO = DatasetSpec(
    name="nome-curto",
    csv_filename="arquivo.csv",
    features=["coluna_a", "coluna_b"],
    target_column="alvo",
    description="de onde vem e o que representa",
)

DATASETS = {PRODUCTIVITY.name: PRODUCTIVITY, NOVO.name: NOVO}
```

Depois, execute com `--run-config 'dataset="nome-curto"'`. Nenhum código de
estratégia, partição, treino ou métrica precisa mudar.

Três campos opcionais cobrem os casos menos simples: `preprocess`, uma função
de limpeza aplicada ao DataFrame antes do split; `positive_threshold`, um corte
explícito para alvos ordinais; e `group_column`, para o regime `natural`. A
entrada `SOCIAL_MEDIA` usa os três.

Se os dados estiverem fora do repositório, aponte a variável de ambiente
`FL_BENCHMARK_DATA_DIR` para a pasta que os contém.

---

## CPU e GPU

O benchmark roda em CPU por padrão. Isso é uma decisão medida, não um descuido:
a rede tem cerca de mil parâmetros, cada batch leva microssegundos de
computação, e o custo de lançar kernels CUDA e transferir dados para a placa
supera o ganho de paralelismo.

Medição nesta máquina — 3 épocas sobre 3.200 amostras, melhor de 3 execuções,
GPU NVIDIA GeForce RTX 3050 6GB Laptop:

| Batch size | CPU | GPU |
|---|---|---|
| 32 | 0,44 s | 0,92 s |
| 128 | 0,19 s | 0,24 s |

A GPU é mais lenta nos dois casos, e a desvantagem diminui conforme o batch
cresce — exatamente o comportamento esperado quando o gargalo é overhead de
lançamento, e não capacidade de cálculo. A tendência se inverteria com uma rede
substancialmente maior ou batches muito grandes.

Para executar na GPU de todo modo:

```bash
cd fl_benchmark
flwr run . local-gpu --federation-config num-supernodes=5 \
  --run-config 'strategy="fedavg" partition="noniid" device="cuda"'
```

A federação `local-gpu`, definida no `pyproject.toml`, reserva fração de GPU para
cada cliente da simulação. O `device="cuda"` é necessário em conjunto, porque a
alocação de recursos pelo backend e o dispositivo usado pelo PyTorch são
configurações independentes. O dispositivo de cada execução fica registrado nos
relatórios, já que afeta diretamente a métrica de tempo.

---

## As cinco estratégias

Cada uma tem seu próprio relatório em `results/`, com os números das duas
partições, a curva por round e a explicação do método.

| Estratégia | Ideia central | Artigo |
|---|---|---|
| **FedAvg** | Média dos pesos ponderada pelo número de exemplos de cada cliente | [arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629) |
| **FedProx** | FedAvg mais um termo proximal que penaliza, no cliente, o afastamento do modelo global | [arxiv.org/abs/1812.06127](https://arxiv.org/abs/1812.06127) |
| **FedAdam** | Trata a diferença agregada como gradiente e aplica um passo de Adam no servidor | [arxiv.org/abs/2003.00295](https://arxiv.org/abs/2003.00295) |
| **FedYogi** | Mesma ideia do FedAdam com atualização aditiva do momento de segunda ordem, mais estável quando os updates variam de magnitude | [arxiv.org/abs/2003.00295](https://arxiv.org/abs/2003.00295) |
| **FedTrimmedAvg** | Descarta uma fração dos valores extremos de cada ponta antes de calcular a média, por robustez a clientes atípicos | [arxiv.org/abs/1803.01498](https://arxiv.org/abs/1803.01498) |

Trocar de estratégia é trocar uma string na linha de comando. Acrescentar uma
sexta é criar um arquivo em `strategies/` com a mesma interface — `NAME`,
`TITLE`, `REFERENCE`, `SUMMARY`, `hyperparams()` e `build()` — e incluí-lo no
registro em `strategies/__init__.py`.

### Calibração do eta

FedAdam e FedYogi aplicam um passo de otimizador no servidor, controlado por
`eta`. FedAvg não tem passo no servidor — apenas promedia — e portanto não tem
esse hiperparâmetro. Comparar as três com um `eta` arbitrário mediria a qualidade
do ajuste, não a do método.

O `eta` foi calibrado por varredura, na partição IID — o cenário de controle.
Calibrar no regime non-IID, onde a comparação de fato acontece, enviesaria o
resultado a favor das adaptativas.

```bash
cd fl_benchmark
python calibrate_eta.py --strategy fedadam --etas 0.01 0.1 0.3 0.6 1.0
```

Resultado para o FedAdam, acurácia final na partição IID:

| eta | 0,01 | 0,1 | **0,3** | 0,6 | 1,0 |
|---|---|---|---|---|---|
| Acurácia | 0,9213 | 0,9735 | **0,9868** | 0,9788 | 0,5025 |

Com `eta = 1,0` o treinamento divergiu, colapsando para 50% nos últimos rounds.
O valor escolhido, 0,3, é o padrão no `pyproject.toml`; as varreduras completas
estão em `results/calibration_eta_*.json` e reproduzidas no comparativo.

A calibração vale para o dataset em que foi feita. No de redes sociais, com o
mesmo `eta = 0,3`, FedAdam e FedYogi terminam o controle IID em 0,73 contra
0,86 das demais — sinal de que o passo do servidor precisa ser recalibrado ali
antes de qualquer conclusão sobre as adaptativas.

Vale notar o que o valor original revela: `eta = 0,01` deixa o passo do servidor
dez vezes menor que o default da implementação de referência, e o FedAdam
termina em 0,92 — não por limitação do método, mas por subparametrização. É o
erro mais fácil de cometer com essa família de estratégias, porque ela tem
**duas** taxas de aprendizado e a do servidor passa despercebida.

### Duas notas de implementação

**O FedProx depende do cliente.** A estratégia, do lado do servidor, agrega
exatamente como o FedAvg; o que distingue o método é o termo proximal
`(mu/2) · ||w - w_global||²` somado à perda **dentro do cliente**. O Flower
apenas envia `proximal-mu` na configuração do round, e cabe à implementação do
cliente aplicá-lo. Sem isso, selecionar FedProx produz resultados
indistinguíveis de FedAvg.

**O corte do FedTrimmedAvg é alheio ao motivo.** Com cinco clientes e
`trim-beta = 0,2`, a estratégia remove um cliente de cada extremo e agrega
apenas os três centrais — 40% dos participantes do round são descartados. Sob
partição non-IID, o cliente cujos dados são mais atípicos é legítimo e é
exatamente o que tende a ser cortado, o que pode piorar o resultado em vez de
protegê-lo. O comportamento é esperado e está registrado em
`results/FEDTRIMMEDAVG.md`.

### Scaffold

Corrige o *client drift* causado por dados heterogêneos, usando *control
variates* — uma estimativa, mantida por cliente, da direção em que seu gradiente
local se desvia do global.

Citado em aula e **não implementado** aqui. Não é uma estratégia nativa do
Flower, e adicioná-la não se resume a escrever um arquivo em `strategies/`: as
*control variates* exigem estado por cliente persistido entre rounds e mudanças
na lógica do cliente, não apenas na agregação do servidor. É um trabalho de
natureza diferente de selecionar as cinco acima, e fica como próximo passo.

---

## Etapa 2 — primeira implementação federada

`fl_addiction/` contém a primeira versão federada do projeto, sobre o dataset de
vício digital. Roda com uma estratégia selecionável por configuração:

```bash
cd fl_addiction
flwr run . --federation-config num-supernodes=5
flwr run . --federation-config num-supernodes=5 --run-config 'strategy="fedprox"'
```

A troca do modelo clássico da etapa 1 por uma rede neural aconteceu aqui, e a
razão vale ser registrada: Federated Learning agrega os clientes operando sobre
os **pesos** do modelo a cada round, e Bagging, Gradient Boosting e Decision
Tree não expõem um vetor de pesos contínuo que possa ser promediado — são
conjuntos de árvores. Uma rede neural, cujas camadas lineares oferecem
exatamente isso, tornou a agregação possível.
