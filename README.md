# ProjetoTransformador II

Classificação de hábitos digitais de estudantes, desenvolvida em três etapas que
compartilham o mesmo problema e mudam a forma de resolvê-lo.

| Etapa | O que faz | Onde está |
|---|---|---|
| 1 — Modelos clássicos | Bagging, Gradient Boosting, Decision Tree, Random Forest, SVM, KNN e Voting sobre o dataset de vício digital | `dataset.ipynb` |
| 2 — Federated Learning | Primeira implementação federada com Flower, uma estratégia selecionável por configuração | `fl_addiction/` |
| 3 — Benchmark de estratégias | Cinco estratégias federadas comparadas em dois regimes de partição, com métricas e relatórios por estratégia | `fl_benchmark/` |

O comparativo consolidado da etapa 3 está em
[`fl_benchmark/results/README.md`](fl_benchmark/results/README.md), e cada
estratégia tem seu próprio relatório ao lado dele. O resultado principal: sob
dados heterogêneos, **FedProx foi a única das cinco a alcançar 99% de acurácia em
todas as sementes**, em 3,3 ± 1,2 rounds. As demais atingiram o alvo em parte das
sementes ou em nenhuma, dentro dos 10 rounds executados.

A etapa 2 continua no repositório como registro da primeira implementação, e roda
sobre o dataset de vício digital (`dataset_novo_indice.csv`). A etapa 3 é
independente dela e usa o dataset de produtividade em `archive/`.

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
│   └── student_productivity_distraction_dataset_20000.csv   dataset da etapa 3
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

### Teto de referência

Antes de comparar estratégias, vale estabelecer o teto: o mesmo modelo treinado
de forma centralizada, com acesso a todos os dados de uma vez.

```bash
cd fl_benchmark
python baseline.py
```

### Uma estratégia por vez

Cada estratégia é executada individualmente, trocando apenas o valor de
`strategy`. O regime de partição é escolhido em `partition`.

```bash
cd fl_benchmark

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
execução perdida ser contada como concluída.

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

### Treino efetivo e wall-clock

Os relatórios registram os dois tempos, e eles são bem diferentes. O
*wall-clock* de uma execução inclui cerca de dois minutos e meio de custo fixo
de inicialização do backend de simulação — construção do pacote da aplicação,
criação do ambiente e subida dos processos dos clientes — que não tem relação
com o método federado sendo avaliado. Para comparar estratégias entre si, a
métrica válida é o **treino efetivo**.

---

## Os dois regimes de partição

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

É esse segundo regime que separa as estratégias. FedProx, FedAdam e FedYogi
existem precisamente para lidar com o *client drift* — a divergência entre os
modelos locais quando as distribuições diferem — e não têm vantagem alguma a
demonstrar num cenário homogêneo.

A semente é fixa nos dois regimes, de modo que todas as estratégias são
avaliadas exatamente sobre a mesma divisão de dados.

---

## Sobre o dataset da etapa 3

`archive/student_productivity_distraction_dataset_20000.csv` tem 20.000 linhas,
18 colunas e nenhum valor ausente. O alvo é `productivity_score`, binarizado
pela mediana — o que dá classes exatamente equilibradas, 10.000 para cada lado,
sem escolher um limiar arbitrário.

Uma característica do dataset precisa ser registrada para que os resultados não
sejam mal interpretados: **`productivity_score` é uma função linear exata de 6
das 14 features utilizadas**. Uma regressão linear sobre o conjunto atinge
R² = 1,0000, com os seguintes coeficientes padronizados:

```
study_hours_per_day  +11,84    focus_score            +6,55
sleep_hours           +5,49    phone_usage_hours      -5,35
stress_level          -3,09    attendance_percentage  +2,82
```

As outras 8 features têm coeficiente exatamente zero — são ruído. Foram mantidas
de propósito, porque aprender a ignorá-las faz parte do que o benchmark mede.

A consequência prática é que o problema é linearmente separável e qualquer
modelo razoável passa de 98% de acurácia. **A acurácia alta é propriedade do
dataset, não mérito do modelo nem da estratégia federada.** É por isso que o
benchmark reporta rounds até a convergência e equidade entre clientes: num
problema em que todas as estratégias saturam, são essas as métricas que ainda
distinguem uma da outra.

O efeito é visível nos resultados: as cinco estratégias terminam com acurácia
praticamente idêntica nos dois regimes, enquanto o desvio entre clientes na
partição non-IID chega a ser uma ordem de grandeza maior que no IID. A pergunta
que o benchmark responde neste dataset não é *qual estratégia aprende melhor*,
mas *qual distribui o aprendizado de forma mais equilibrada entre as
instituições e com menos rounds de comunicação*.

Quando o dataset da instituição for liberado, essa limitação provavelmente
desaparece, e as métricas de acurácia voltam a discriminar por si mesmas.

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
