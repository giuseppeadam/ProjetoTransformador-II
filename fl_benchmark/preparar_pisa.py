"""Prepara o PISA 2025 para o benchmark federado.

Executado **uma vez**, materializa em disco o dataset já limpo, normalizado e
dividido por país. A razão é de custo: na simulação cada cliente roda como um
processo separado, com memória própria, e sem esta etapa os 48 clientes
carregariam os 755.721 registros brutos e repetiriam toda a limpeza a cada
round.

O layout resultante também espelha a realidade do federated learning — cada
instituição possui um arquivo com os seus dados e apenas os seus.

Três tratamentos acontecem aqui e são essenciais:

1. **Limpeza dos códigos de missing do SPSS.** O arquivo de origem foi extraído
   com `user_missing=True`, que *preserva* os códigos em vez de convertê-los em
   nulo. Valores como 97 ("não aplicável") e 99 ("sem resposta") estão gravados
   como se fossem medições, e `isna()` reporta zero nulos — falso. Sem este
   tratamento a correlação de ESCS com desempenho aparece como -0,150, quando
   na verdade é +0,401.

2. **Derivação do alvo.** As dez colunas `PV*MATH` não são dez notas: são dez
   imputações da mesma proficiência latente. Usamos a média, binarizada pela
   mediana global.

3. **Normalização global.** Média e desvio são calculados sobre o conjunto de
   treino inteiro e aplicados a todos os países. Isso é uma simplificação
   deliberada, discutida em `ESTRATEGIA_PISA.md`: em federated learning estrito,
   obter estatísticas globais exigiria um round prévio de agregação. Como o
   objeto de estudo aqui é a agregação de pesos, e não o protocolo de
   pré-processamento, normaliza-se de forma centralizada e registra-se a
   ressalva.

Uso:
    python preparar_pisa.py
    python preparar_pisa.py --min-alunos 500 --test-size 0.2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent

ORIGEM = REPO / "dataset_PISA_2025" / "CY09_MS_STU_PUF" / "pisa_reduzido.parquet"
DESTINO = REPO / "archive" / "pisa_preparado"

COLUNA_PAIS = "CNT"
COLUNA_ESCOLA = "CNTSCHID"
ALVO = "acima_mediana"

# Range válido de cada feature. A limpeza é por range, e não por lista de
# códigos, porque os códigos de missing variam com a escala da pergunta: em
# `ST326*` o valor 9 é uma categoria legítima ("mais de 8 horas por dia"),
# enquanto em `ST097Q06DA`, cuja escala vai até 4, o 9 é ausência de resposta.
RANGES: dict[str, tuple[float, float]] = {}
for _i in range(1, 7):
    RANGES[f"ST326Q0{_i}JA"] = (1, 9)
for _c in ("ST322Q01JA", "ST322Q02JA", "ST322Q06JA", "ST322Q07JA", "ST296Q04JA"):
    RANGES[_c] = (1, 6)
RANGES["ST097Q06DA"] = (1, 4)
# As escalas WLE do PISA são padronizadas na população internacional; valores
# fora de [-12, 12] são códigos, não medidas.
for _c in ("ESCS", "HOMEPOS", "ICTRES", "BELONG", "BULLIED"):
    RANGES[_c] = (-12.0, 12.0)

FEATURES = list(RANGES)
PLAUSIBLE_VALUES = [f"PV{i}MATH" for i in range(1, 11)]


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui por nulo todo valor fora do range válido de sua coluna."""
    limpo = df.copy()
    for coluna, (minimo, maximo) in RANGES.items():
        serie = limpo[coluna]
        limpo[coluna] = serie.where((serie >= minimo) & (serie <= maximo))
    return limpo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origem", default=str(ORIGEM))
    parser.add_argument("--destino", default=str(DESTINO))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-alunos",
        type=int,
        default=200,
        help=(
            "países com menos alunos que isto são descartados: um cliente com "
            "pouquíssimos dados não sustenta treino e validação locais"
        ),
    )
    args = parser.parse_args()

    origem = Path(args.origem)
    destino = Path(args.destino)
    if not origem.is_file():
        print(f"arquivo de origem não encontrado: {origem}")
        return 1

    inicio = time.perf_counter()
    print(f"lendo {origem.name}...", flush=True)
    bruto = pd.read_parquet(origem)
    print(f"  {len(bruto):,} linhas x {bruto.shape[1]} colunas", flush=True)

    faltando = [c for c in FEATURES + PLAUSIBLE_VALUES + [COLUNA_PAIS] if c not in bruto.columns]
    if faltando:
        print(f"colunas ausentes na origem: {faltando}")
        return 1

    # --- limpeza e alvo ----------------------------------------------------
    print("limpando códigos de missing por range válido...", flush=True)
    dados = limpar(bruto[FEATURES + PLAUSIBLE_VALUES + [COLUNA_PAIS, COLUNA_ESCOLA]])

    invalidos_por_feature = {
        c: float(round(dados[c].isna().mean() * 100, 2)) for c in FEATURES
    }
    nota = dados[PLAUSIBLE_VALUES].mean(axis=1)
    dados = dados.assign(nota_math=nota)

    antes = len(dados)
    dados = dados.dropna(subset=FEATURES + ["nota_math"])
    print(f"  linhas completas: {len(dados):,} de {antes:,} ({len(dados)/antes*100:.1f}%)", flush=True)

    mediana = float(dados.nota_math.median())
    dados[ALVO] = (dados.nota_math > mediana).astype(np.float32)
    print(f"  mediana global de matemática: {mediana:.1f}", flush=True)

    # --- descarte de países pequenos ---------------------------------------
    contagem = dados[COLUNA_PAIS].value_counts()
    pequenos = sorted(contagem[contagem < args.min_alunos].index)
    if pequenos:
        print(f"  descartando {len(pequenos)} países com < {args.min_alunos} alunos: {pequenos}", flush=True)
        dados = dados[~dados[COLUNA_PAIS].isin(pequenos)]
    paises = sorted(dados[COLUNA_PAIS].unique())
    print(f"  {len(paises)} países mantidos, {len(dados):,} alunos", flush=True)

    # --- divisão treino/teste, estratificada dentro de cada país -----------
    # Estratificar por país e por classe garante que o conjunto de teste
    # central represente todos os países, e que todo país mantenha dados de
    # treino próprios — as duas coisas são necessárias: o teste central é a
    # métrica honesta do modelo global, e cada país é um cliente.
    print("dividindo treino/teste (estratificado por país e classe)...", flush=True)
    estrato = dados[COLUNA_PAIS].astype(str) + "_" + dados[ALVO].astype(int).astype(str)
    treino, teste = train_test_split(
        dados,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=estrato,
    )
    print(f"  treino: {len(treino):,} | teste central: {len(teste):,}", flush=True)

    # --- normalização global, com a ressalva registrada --------------------
    media = treino[FEATURES].mean()
    escala = treino[FEATURES].std().replace(0.0, 1.0)

    def normalizar(frame: pd.DataFrame) -> pd.DataFrame:
        saida = frame.copy()
        saida[FEATURES] = ((frame[FEATURES] - media) / escala).astype(np.float32)
        return saida

    treino = normalizar(treino)
    teste = normalizar(teste)

    # --- gravação ----------------------------------------------------------
    pasta_clientes = destino / "clientes"
    pasta_clientes.mkdir(parents=True, exist_ok=True)

    colunas_saida = FEATURES + [ALVO, COLUNA_PAIS, COLUNA_ESCOLA, "nota_math"]
    teste[colunas_saida].to_parquet(destino / "teste_central.parquet", index=False)

    print(f"gravando um arquivo por país em {pasta_clientes.name}/...", flush=True)
    resumo_clientes = []
    for pais in paises:
        fatia = treino[treino[COLUNA_PAIS] == pais]
        fatia[colunas_saida].to_parquet(pasta_clientes / f"{pais}.parquet", index=False)
        resumo_clientes.append(
            {
                "pais": pais,
                "alunos": int(len(fatia)),
                "fracao_acima_mediana": float(round(fatia[ALVO].mean(), 4)),
                "nota_media": float(round(fatia.nota_math.mean(), 1)),
                "escolas": int(fatia[COLUNA_ESCOLA].nunique()),
            }
        )

    resumo_clientes.sort(key=lambda r: r["fracao_acima_mediana"])
    metadados = {
        "origem": origem.name,
        "features": FEATURES,
        "target": ALVO,
        "target_definicao": (
            "media das 10 plausible values de matematica, binarizada pela mediana global"
        ),
        "mediana_global": round(mediana, 2),
        "normalizacao": {
            "tipo": "global, ajustada no conjunto de treino",
            "ressalva": (
                "em federated learning estrito, estatisticas globais exigiriam um round "
                "previo de agregacao; simplificacao documentada em ESTRATEGIA_PISA.md"
            ),
            "media": {c: float(media[c]) for c in FEATURES},
            "escala": {c: float(escala[c]) for c in FEATURES},
        },
        "linhas": {
            "origem": int(len(bruto)),
            "completas": int(antes and len(dados)),
            "treino": int(len(treino)),
            "teste_central": int(len(teste)),
        },
        "seed": args.seed,
        "test_size": args.test_size,
        "min_alunos": args.min_alunos,
        "paises_descartados": pequenos,
        "invalidos_por_feature_pct": invalidos_por_feature,
        "clientes": resumo_clientes,
    }
    with (destino / "metadados.json").open("w", encoding="utf-8") as arquivo:
        json.dump(metadados, arquivo, indent=2, ensure_ascii=False)

    duracao = time.perf_counter() - inicio
    skew = [r["fracao_acima_mediana"] for r in resumo_clientes]
    tamanhos = [r["alunos"] for r in resumo_clientes]
    print()
    print(f"concluido em {duracao:.1f}s")
    print(f"  {len(paises)} clientes em {pasta_clientes}")
    print(f"  heterogeneidade de rotulo: {min(skew):.2f} a {max(skew):.2f}")
    print(f"  tamanho por cliente: {min(tamanhos):,} a {max(tamanhos):,}")
    print(f"  metadados em {destino / 'metadados.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
