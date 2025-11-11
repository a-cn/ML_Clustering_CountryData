# app.py
# ------------------------------------------------------------
# Clustering com PyCaret + Streamlit (PyCaret 3.x)
# Base: Iris (ou CSV) | Métricas + Interpretação
# Dendrograma (scipy) + Heatmap de médias (plotly)
# Seleção de nº de clusters (k) p/ kmeans/hclust/birch/spectral
# Parâmetros p/ DBSCAN e OPTICS
# Compatível com Python 3.9+
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional
import os

from pycaret.datasets import get_data
from pycaret.clustering import (
    setup, create_model, assign_model, plot_model, save_model
)

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Gráficos extras
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Clustering com PyCaret", layout="wide")
st.title("Clusterização Automática com PyCaret (v3.x)")

# Âncora e botão flutuante "Topo" disponível em todas as abas
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("""
<style>
.back-to-top {
  position: fixed;
  right: 24px;
  bottom: 24px;
  z-index: 9999;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-color, #2c7be5);
  color: #ffffff !important;
  border: none;
  text-decoration: none !important;
  font-weight: 700;
  font-size: 20px;
  line-height: 1;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.back-to-top:visited,
.back-to-top:active,
.back-to-top:focus { color: #ffffff !important; }
.back-to-top:hover { transform: translateY(-2px); opacity: .95; }
@media (prefers-reduced-motion: reduce) { .back-to-top:hover { transform: none; } }
@media (max-width: 600px) { .back-to-top { right: 12px; bottom: 12px; width: 40px; height: 40px; font-size: 18px; } }
</style>
<a href="#top" class="back-to-top" title="Voltar ao topo" aria-label="Voltar ao topo">&#8593;</a>
""", unsafe_allow_html=True)

# Abas principais do corpo do app
tab_orient, tab_eda, tab_treino, tab_relatorio = st.tabs([
    "Orientações",
    "EDA (profiling)",
    "Treino & Métricas",
    "Relatório",
])

# Containers para facilitar roteamento de conteúdo
orient_container = tab_orient.container()
eda_container = tab_eda.container()
treino_container = tab_treino.container()
relatorio_container = tab_relatorio.container()

# ============================================================
# Aba Orientações
# ============================================================
with orient_container:
    st.subheader("Orientações de Uso — Clustering com PyCaret")

    st.markdown("""
        ### **1. Objetivo do Aplicativo**

        Este aplicativo tem como objetivo **descobrir padrões ocultos em bases de dados numéricos** através de **algoritmos de aprendizado não supervisionado (clustering)**.  
        Ele utiliza a biblioteca **PyCaret**, permitindo explorar, treinar e interpretar modelos sem necessidade de programação manual.

        O app foi projetado para:
        - Facilitar **análises exploratórias (EDA)**;  
        - Realizar **pré-processamento automático** (normalização, PCA, seleção de colunas);  
        - Treinar e comparar **múltiplos algoritmos de clusterização**;  
        - Exibir **métricas e visualizações** de qualidade dos clusters;  
        - Permitir **interpretação dos resultados** e **download** do modelo e dos dados rotulados.
        
        ---

        ### **2. Escolha e Carregamento do Dataset**

        Na **barra lateral (Sidebar)**, você encontrará a seção **“Fonte de Dados”** com duas opções:

        #### **Opção 1 — Iris (Exemplo)**
        - Dataset clássico de classificação com 3 classes (`setosa`, `versicolor`, `virginica`).
        - Serve para **testes rápidos** e validação do funcionamento do app.
        - É carregado automaticamente pelo PyCaret (`get_data("iris")`).

        #### **Opção 2 — Upload CSV (Seu Dataset)**
        - Utilize esta opção para enviar seu próprio arquivo `.csv`.
        - No contexto deste projeto, use o arquivo **`Country-data.csv`**, que contém dados socioeconômicos de 167 países.

        > ⚠️ **Importante:**  
        > - O dataset `Country-data.csv` inclui uma coluna chamada `country`, que **identifica os países**.  
        > - Essa coluna **não deve ser usada no treinamento**, pois não é numérica.  
        > - Na barra lateral, em **Configuração das Features**, use o campo **“Selecione features para ignorar”** para marcar `country` (pré-selecionada por padrão).  
        > - As colunas escolhidas nesse campo serão ignoradas pelo `setup` do PyCaret.
        
        ---

        ### **3. Configuração das Features**

        Após carregar o dataset:
        - Primeiro, use o campo **“Selecione features para ignorar”** para excluir colunas do treinamento (ex.: `country`). Por padrão, `country` já vem pré-selecionada.  
        - As colunas numéricas são detectadas automaticamente.  
        - Você pode selecionar manualmente quais features usar (menu **“Selecione features numéricas”**).  

        Para o `Country-data.csv`, recomenda-se incluir:
        child_mort, exports, health, imports, income, inflation, life_expec, total_fer, gdpp

        Essas variáveis descrevem **aspectos econômicos e sociais** de cada país e servirão de base para a descoberta dos grupos.
        
        ---

        ### **4. Pré-Processamento**

        A seção **“Pré-processamento”** permite ajustar o comportamento dos dados antes do clustering:

        | Opção | Função | Recomendação |
        |-------|--------|---------------|
        | **Normalizar** | Padroniza todas as variáveis (média=0, desvio=1). | ✅ **Ativar sempre** (algoritmos de clustering são sensíveis à escala). |
        | **Aplicar PCA** | Reduz dimensionalidade via Análise de Componentes Principais. | ❌ **Desativar** para `Country-data.csv` (mantém interpretabilidade). |
        | **Componentes PCA** | Define o número de componentes a reter. | Usado apenas se PCA estiver ativado. |

        > 🔎 *Dica:* A normalização garante que variáveis como “income” (valores grandes) e “health” (% do PIB) contribuam igualmente no agrupamento.
        
        ---

        ### **5. Parâmetros de Clusterização**

        A próxima seção da barra lateral define **parâmetros específicos dos algoritmos**:

        | Parâmetro | Descrição | Recomendações |
        |------------|------------|---------------|
        | **Número de clusters (k)** | Usado em *K-Means*, *Hierarchical Clustering*, *Birch*, *Spectral*. | Se `0`, o app escolherá automaticamente (usando o gráfico de cotovelo). |
        | **DBSCAN – eps** | Distância máxima entre pontos para formar um cluster. | Para `Country-data.csv`, experimente valores entre `0.5` e `1.5`. |
        | **DBSCAN – min_samples** | Número mínimo de pontos por cluster. | Valores entre `3` e `10` geralmente funcionam bem. |
        | **OPTICS – min_samples** | Parâmetro análogo ao DBSCAN. | Pode deixar o padrão (`5`). |
        
        ---

        ### **6. Execução do Treinamento**

        Na aba **“Treino & Métricas”**, clique em **“Rodar Clusterização”**.  
        O app irá:

        1. Aplicar o pré-processamento configurado (normalização e PCA se marcado);  
        2. Treinar todos os algoritmos selecionados na barra lateral;  
        3. Calcular automaticamente as métricas de qualidade:
        - **Silhouette Score** → quanto maior, melhor separação;
        - **Calinski–Harabasz Index** → quanto maior, melhor;
        - **Davies–Bouldin Index** → quanto menor, melhor;
        4. Exibir uma **tabela comparativa** com os resultados e uma **interpretação automática** das métricas.
        
        ---

        ### **7. Interpretação e Visualização dos Clusters**

        Após o treinamento:
        - Escolha um modelo específico (ex.: *kmeans*, *hclust*, *dbscan*) para análise detalhada.  
        - São exibidos:
        - Tabela com **estatísticas médias por cluster** (perfil socioeconômico de cada grupo);
        - **Gráficos automáticos do PyCaret**:
            - *Elbow Plot*: sugere o número ideal de clusters;
            - *Silhouette Plot*: avalia separação dos grupos;
            - *t-SNE Plot*: representação 2D dos clusters;
        - **Dendrograma** (para *Hierarchical Clustering*);
        - **Heatmap interativo** mostrando médias normalizadas por cluster.
        
        ---

        ### **8. Exportação dos Resultados**

        Na parte inferior da aba de treino:
        - Faça o **download do CSV** com os países e seus respectivos clusters atribuídos;  
        - Baixe o **modelo treinado (.pkl)**, que pode ser reutilizado para novas predições ou deploy.

        A aba **“Exportar Relatório”** apenas centraliza as opções de download disponíveis após o treino.
        
        ---

        ### **9. Interpretação no Contexto do Country-data.csv**

        Os grupos (clusters) gerados representam **conjuntos de países com características socioeconômicas semelhantes**.  
        Exemplo de possíveis interpretações:

        - **Cluster 0:** países com alta mortalidade infantil, baixa renda e baixa expectativa de vida;  
        - **Cluster 1:** países de alta renda e boa expectativa de vida;  
        - **Cluster 2:** países de desenvolvimento intermediário.  

        Esses insights podem ser usados para **análises comparativas**, **planejamento de políticas públicas** ou **estudos de desenvolvimento econômico**.
        
        ---

        ### **10. Dicas Finais**

        - Use o botão **“Gerar relatório (profiling)”** na aba *EDA* para obter um resumo completo das variáveis.  
        - Ajuste o valor de **k** e repita o treinamento para observar mudanças nos agrupamentos.  
        - Compare diferentes algoritmos — o **K-Means** costuma gerar resultados mais estáveis para o dataset `Country-data.csv`.  
        - Evite ativar PCA se o foco for **interpretação das variáveis originais**.
    """)

# ============================================================
# Utilidades
# ============================================================
def safe_numeric_df(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    """Seleciona apenas colunas numéricas (ou as informadas) e remove constantes."""
    if cols:
        num_df = df[cols].copy()
    else:
        num_df = df.select_dtypes(include="number").copy()
    nunique = num_df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    return num_df[keep]

def compute_metrics(X: pd.DataFrame, labels: pd.Series) -> tuple:
    """Retorna (silhouette, calinski-harabasz, davies-bouldin) ou NaN se não houver >1 cluster."""
    if len(np.unique(labels)) <= 1:
        return (np.nan, np.nan, np.nan)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return (sil, ch, db)

def interpret_row(model_name: str, sil: float, ch: float, db: float) -> str:
    if np.isnan(sil):
        return f"{model_name}: não conseguiu formar múltiplos clusters ou falhou na avaliação."
    parts = []
    if sil > 0.5:
        parts.append(f"Silhouette = {sil:.2f} (boa separação)")
    elif sil > 0.25:
        parts.append(f"Silhouette = {sil:.2f} (moderada, pode melhorar)")
    else:
        parts.append(f"Silhouette = {sil:.2f} (clusters fracos/ruído)")
    parts.append(f"CH = {ch:.1f} (quanto maior, melhor)")
    if db < 0.5:
        parts.append(f"DB = {db:.2f} (excelente compactação)")
    elif db < 1.0:
        parts.append(f"DB = {db:.2f} (bom resultado)")
    else:
        parts.append(f"DB = {db:.2f} (sobreposição entre clusters)")
    return f"{model_name} → " + "; ".join(parts) + "."

def cluster_profiles_table(labeled: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Estatísticas descritivas por cluster (mean/median/std/min/max) das variáveis numéricas."""
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        return pd.DataFrame()
    prof = labeled.groupby(cluster_col)[num_cols].agg(["mean", "median", "std", "min", "max"])
    return prof

def best_cluster_label_mapping(labeled: pd.DataFrame, cluster_col="Cluster", truth_col="species") -> tuple:
    """Mapeia cluster -> rótulo mais frequente e calcula pureza (acertos por modo / total)."""
    mapping = {}
    total = len(labeled)
    correct = 0
    for c, group in labeled.groupby(cluster_col):
        mode_label = group[truth_col].mode(dropna=False)
        if len(mode_label) > 0:
            chosen = mode_label.iloc[0]
            mapping[c] = chosen
            correct += (group[truth_col] == chosen).sum()
        else:
            mapping[c] = None
    purity = correct / total if total > 0 else np.nan
    return mapping, purity

def make_dendrogram(X: pd.DataFrame, sample_cap: int = 250, method: str = "ward"):
    """Dendrograma usando scipy; amostra linhas para desempenho."""
    if len(X) > sample_cap:
        X_plot = X.sample(n=sample_cap, random_state=42)
        st.caption(f"Dendrograma com amostra de {sample_cap} linhas (de {len(X)}) para desempenho.")
    else:
        X_plot = X
    Z = linkage(X_plot.values, method=method)
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(f"Dendrograma (método: {method})")
    ax.set_ylabel("Distância de ligação")
    st.pyplot(fig)

def plot_cluster_means_heatmap(labeled: pd.DataFrame, cluster_col: str = "Cluster",
                               zscore: bool = True, top_n_features: Optional[int] = None):
    """Heatmap das médias por cluster; pode aplicar z-score e limitar às top-N features."""
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        st.info("Sem colunas numéricas para heatmap.")
        return

    means = labeled.groupby(cluster_col)[num_cols].mean()

    # Selecionar features mais discriminativas
    if top_n_features is not None and top_n_features > 0 and top_n_features < len(num_cols):
        var_between = means.var(axis=0)
        selected = var_between.sort_values(ascending=False).head(top_n_features).index
        means = means[selected]

    data_plot = means.copy()
    title_suffix = ""
    if zscore:
        data_plot = (means - means.mean(axis=0)) / (means.std(axis=0).replace(0, np.nan))
        title_suffix = " (z-score)"
        data_plot = data_plot.fillna(0.0)

    # Rótulos do eixo Y (evita tentar int() em strings)
    def _fmt_cluster_label(v):
        if isinstance(v, (int, np.integer)):
            return f"Cluster {int(v)}"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return f"Cluster {int(v)}"
        s = str(v)
        return s if s.lower().startswith("cluster") else f"{s}"

    y_labels = [_fmt_cluster_label(i) for i in data_plot.index]

    fig = px.imshow(
        data_plot,
        x=[str(c) for c in data_plot.columns],
        y=y_labels,
        color_continuous_midpoint=0.0 if zscore else None,
        aspect="auto",
        labels=dict(color="intensidade")
    )
    fig.update_layout(title=f"Heatmap das médias por cluster{title_suffix}", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 1) Fonte de Dados
# ============================================================
st.sidebar.header("Fonte de Dados")
fonte = st.sidebar.radio("Selecione a fonte", ["Iris (Exemplo)", "Upload CSV"])

if fonte == "Iris (Exemplo)":
    df = get_data("iris")
    with treino_container:
        st.write("Usando base de exemplo Iris:", df.shape)
        st.dataframe(df.head())
else:
    file = st.sidebar.file_uploader("Carregue seu CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        with treino_container:
            st.subheader("Prévia dos dados carregados")
            st.write("Dimensões:", df.shape)
            st.dataframe(df.head())
    else:
        st.stop()

# ------------------------------------------------------------
# Rótulo opcional (clusters × rótulos) — patch robusto
# ------------------------------------------------------------
raw_cols = list(df.columns)
display_cols = [str(c) for c in raw_cols]
options_display = ["<nenhum>"] + display_cols

candidates_lower = {"species", "label", "target", "classe", "class"}
guess_idx = None
for i, c in enumerate(raw_cols):
    if str(c).lower() in candidates_lower:
        guess_idx = i
        break
default_index = 0 if guess_idx is None else int(1 + guess_idx)

sel_display = st.sidebar.selectbox(
    "Coluna de rótulo (opcional, para comparação)",
    options_display,
    index=default_index,
    help="Opcional: selecione a coluna com rótulos verdadeiros para comparar com os clusters. Não é usada no treinamento."
)
label_col = None if sel_display == "<nenhum>" else raw_cols[display_cols.index(sel_display)]

# ============================================================
# 2) Seleção de Colunas
# ============================================================
st.sidebar.header("Configuração das Features")
cols_ignore_default = ['country'] if 'country' in df.columns else []
cols_ignore = st.sidebar.multiselect(
    "Selecione features para ignorar",
    df.columns.tolist(),
    default=cols_ignore_default,
    help="Colunas que serão explicitamente ignoradas no setup do PyCaret (ex.: 'country')."
)
cols_num = st.sidebar.multiselect(
    "Selecione features numéricas",
    df.columns.tolist(),
    help="Escolha as variáveis numéricas usadas no clustering. Se nenhuma for escolhida, todas as numéricas não constantes serão utilizadas."
)
if not cols_num:
    cols_num = df.select_dtypes(include="number").columns.tolist()
    st.sidebar.info("Usando automaticamente colunas numéricas não constantes.")

# ============================================================
# 3) Pré-processamento
# ============================================================
st.sidebar.header("Pré-processamento")
normalize = st.sidebar.checkbox("Normalizar", value=True)
pca = st.sidebar.checkbox("Aplicar PCA", value=False)
pca_comp = st.sidebar.slider(
    "Componentes PCA",
    2,
    10,
    3,
    help="Número de componentes principais usados para reduzir dimensionalidade antes do clustering. Só é aplicado se 'Aplicar PCA' estiver marcado.",
    disabled=not pca
)

# ============================================================
# Parâmetros de modelos (K, eps, etc.)
# ============================================================
st.sidebar.header("Parâmetros de cluster")
k_clusters = st.sidebar.number_input(
    "Número de clusters (k) para k-means/hclust/birch/spectral (0 = auto)",
    min_value=0, value=0, step=1
)
dbscan_eps = st.sidebar.slider(
    "DBSCAN eps",
    min_value=0.05,
    max_value=5.0,
    value=1.0,
    step=0.05,
    help="Raio de vizinhança do DBSCAN. Aumente para formar menos clusters (mais aglomeração); diminua para separar mais."
)
dbscan_min_samples = st.sidebar.number_input(
    "DBSCAN min_samples",
    min_value=1,
    value=5,
    step=1,
    help="Número mínimo de pontos dentro de 'eps' para um ponto ser núcleo. Aumente para clusters mais conservadores (mais ruído)."
)
optics_min_samples = st.sidebar.number_input(
    "OPTICS min_samples",
    min_value=1,
    value=5,
    step=1,
    help="Número mínimo de pontos para considerar um núcleo no OPTICS. Controla a densidade mínima dos clusters."
)

# Desempenho
st.sidebar.header("Desempenho")
limit_rows = st.sidebar.number_input(
    "Limitar amostras (0 = sem limite)",
    min_value=0,
    value=0,
    step=100,
    help="Subamostra a base para testes rápidos. 0 usa todos os registros; >0 sorteia exatamente esse número de linhas."
)
models_to_try = st.sidebar.multiselect(
    "Algoritmos a testar",
    ["kmeans", "hclust", "dbscan", "optics", "birch", "spectral"],
    default=["kmeans", "hclust", "dbscan"]
)

# Extras visuais
st.sidebar.header("Exibição avançada")
show_dendro_anyway = st.sidebar.checkbox("Mostrar dendrograma mesmo se não for hclust", value=False)
heatmap_zscore = st.sidebar.checkbox("Heatmap com z-score (recomendado)", value=True)
heatmap_topn = st.sidebar.number_input("Heatmap: limitar às top-N variáveis (0 = todas)", min_value=0, value=0, step=1)

# ============================================================
# Aba EDA
# ============================================================
with eda_container:
    st.subheader("Visão geral dos dados")
    st.write("Dimensões:", df.shape)
    st.dataframe(df.head())
    if (
        fonte != "Iris (Exemplo)"
        and "file" in locals()
        and file is not None
        and str(getattr(file, "name", "")).lower() == "country-data.csv"
    ):
        st.subheader("Entendimento do dataset")
        st.markdown(
            """
            | Coluna       | Descrição                                           | Tipo       | Observação                             |
            | ------------ | --------------------------------------------------- | ---------- | ---------------------------------------|
            | `country`    | Nome do país                                        | Categórica | Identificador (ignorado no clustering) |
            | `child_mort` | Taxa de mortalidade infantil (por 1000 nascimentos) | Numérica   | Importante indicador social            |
            | `exports`    | Exportações (% do PIB)                              | Numérica   | Econômico                              |
            | `health`     | Gastos com saúde (% do PIB)                         | Numérica   | Econômico/Social                       |
            | `imports`    | Importações (% do PIB)                              | Numérica   | Econômico                              |
            | `income`     | Renda média per capita                              | Numérica   | Econômico                              |
            | `inflation`  | Taxa de inflação (%)                                | Numérica   | Econômico                              |
            | `life_expec` | Expectativa de vida                                 | Numérica   | Social                                 |
            | `total_fer`  | Taxa de fertilidade                                 | Numérica   | Social                                 |
            | `gdpp`       | PIB per capita                                      | Numérica   | Econômico                              |
            """
        )
    if st.button("🔍 Gerar relatório (profiling)", use_container_width=False):
        Profile = None
        try:
            from ydata_profiling import ProfileReport as Profile  # type: ignore
        except Exception:
            try:
                from pandas_profiling import ProfileReport as Profile  # type: ignore
            except Exception:
                Profile = None
        if Profile is None:
            st.info("Biblioteca de profiling não instalada. Instale 'ydata-profiling' ou 'pandas-profiling' para ativar.")
        else:
            try:
                with st.spinner("Gerando relatório..."):
                    profile = Profile(df, title="Profiling — Dataset", minimal=False)
                    html_str = profile.to_html()
                st.session_state.eda_profile_html = html_str
                st.success("Relatório gerado!")
            except Exception as e:
                st.error(f"Falha ao gerar o profiling: {e}")

    if st.session_state.get("eda_profile_html"):
        st.components.v1.html(st.session_state.eda_profile_html, height=900, scrolling=True)
        st.download_button(
            "💾 Baixar relatório (HTML)",
            data=st.session_state.eda_profile_html.encode("utf-8"),
            file_name="profiling_relatorio.html",
            mime="text/html",
            use_container_width=False
        )
        if st.button("🗑️ Limpar Relatório", use_container_width=False, key="btn_clear_eda_report"):
            st.session_state.pop("eda_profile_html", None)
            st.rerun()

# ============================================================
# 4) Rodar Pipeline - Aba Treino & Métricas
# ============================================================
with treino_container:
    if st.button("Rodar Clusterização"):
        data_full = safe_numeric_df(df, cols_num)
        if limit_rows and limit_rows > 0 and limit_rows < len(data_full):
            data = data_full.sample(n=limit_rows, random_state=42).reset_index(drop=True)
        else:
            data = data_full.copy()

        # Aplicar features a ignorar selecionadas pelo usuário, considerando apenas as presentes nos dados do setup
        ignore_cols_effective = [c for c in cols_ignore if c in data.columns]

        setup(
            data=data,
            session_id=42,
            normalize=normalize,
            pca=pca,
            pca_components=pca_comp if pca else None,
            ignore_features=ignore_cols_effective,
            verbose=False,
            html=False,
        )

        st.success("Setup concluído")
        st.caption("Observação: colunas constantes foram removidas automaticamente antes do setup.")

        # Testar modelos
        resultados, objetos = [], {}
        k_models = {"kmeans", "hclust", "birch", "spectral"}

        for m in models_to_try:
            try:
                params = {}
                # aplica k se informado (>0) e se o modelo aceitar k
                if (k_clusters is not None) and (int(k_clusters) > 0) and (m in k_models):
                    params["num_clusters"] = int(k_clusters)
                # parâmetros específicos
                if m == "dbscan":
                    params["eps"] = float(dbscan_eps)
                    params["min_samples"] = int(dbscan_min_samples)
                if m == "optics":
                    params["min_samples"] = int(optics_min_samples)

                model = create_model(m, **params)
                labeled = assign_model(model, transformation=True)
                X = labeled.drop(columns=["Cluster"])
                y = labeled["Cluster"]

                sil, ch, db = compute_metrics(X, y)
                resultados.append([m, sil, ch, db])
                objetos[m] = (model, labeled)
            except Exception as e:
                resultados.append([m, np.nan, np.nan, np.nan])
                objetos[m] = (str(e), None)

        res_df = pd.DataFrame(resultados, columns=["Modelo", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
        # Persistir resultados para manter após reruns
        st.session_state.cluster_results_df = res_df
        st.session_state.cluster_objects = objetos
        st.session_state.cluster_data_full = data_full
        st.session_state.cluster_data_sample = data
        st.session_state.cluster_df = df
        st.session_state.cluster_label_col = label_col
        st.session_state.cluster_setup_params = {
            "normalize": normalize,
            "pca": pca,
            "pca_components": pca_comp if pca else None,
            "ignore_features_user": cols_ignore,
            "ignore_features_effective": ignore_cols_effective,
        }

        st.success("Clusterização concluída")
        st.rerun()

    # Renderização persistente após executar clusterização
    if st.session_state.get("cluster_results_df") is not None:
        res_df = st.session_state.cluster_results_df
        objetos = st.session_state.cluster_objects
        data_full = st.session_state.cluster_data_full
        df = st.session_state.cluster_df
        label_col = st.session_state.cluster_label_col

        st.subheader("Comparação de modelos")
        st.dataframe(res_df)

        # Interpretação automática
        st.subheader("Interpretação automática das métricas")
        for _, row in res_df.iterrows():
            st.markdown(interpret_row(row["Modelo"], row["Silhouette"], row["Calinski-Harabasz"], row["Davies-Bouldin"]))

        # Escolher modelo
        st.subheader("Análise detalhada")
        escolha = st.selectbox("Modelo:", res_df["Modelo"].tolist(), key="cluster_model_select")
        obj, labeled_final = objetos.get(escolha, (None, None))

        if isinstance(obj, str) or labeled_final is None:
            st.warning(f"Não foi possível analisar {escolha}")
            st.stop()

        st.write("Amostra com clusters atribuídos:")
        st.dataframe(labeled_final.head())
        st.caption("Esta é uma amostra para conferência rápida onde cada linha mostra um registro do dataset e a coluna ‘Cluster’ indica o grupo ao qual ele foi atribuído.\n"
                   "Os demais números são valores padronizados:\n" 
                   "- 0 ≈ média do conjunto;\n"
                   "- Valores positivos = acima da média;\n"
                   "- Valores negativos = abaixo da média.")

        # Perfis por cluster (tabela)
        st.subheader("Perfis dos clusters (estatísticas)")
        prof = cluster_profiles_table(labeled_final, cluster_col="Cluster")
        if not prof.empty:
            st.dataframe(prof)
            st.caption("Resumo por grupo: para cada variável é exibida a média, a mediana, a variação (desvio‑padrão) e os menores/maiores valores (mín/máx) dentro do cluster. "
                       "Use para entender o que caracteriza cada grupo (ex.: ‘Cluster 2 tem renda acima da média’). "
                       "Os valores estão padronizados, então compare sinais e magnitudes, não unidades reais.")

        # Recriar contexto do PyCaret para permitir plot_model após rerun
        try:
            _cfg = st.session_state.get("cluster_setup_params", {})
            _data_for_setup = st.session_state.get(
                "cluster_data_sample",
                labeled_final.drop(columns=["Cluster"], errors="ignore")
            )
            _user_ignore = _cfg.get("ignore_features_user", [])
            _ignore_features = [c for c in _user_ignore if hasattr(_data_for_setup, "columns") and c in _data_for_setup.columns]

            setup(
                data=_data_for_setup,
                session_id=42,
                normalize=_cfg.get("normalize", True),
                pca=_cfg.get("pca", False),
                pca_components=_cfg.get("pca_components"),
                ignore_features=_ignore_features,
                verbose=False,
                html=False,
            )
        except Exception:
            pass

        # Visualizações do modelo escolhido (PyCaret)
        st.subheader("Visualizações do modelo (PyCaret)")
        captions_map = {
            "elbow": "**Como ler:**\n"
                     "- Procure o ‘cotovelo’: o ponto onde a curva deixa de cair rápido;\n"
                     "- O k nesse ponto costuma equilibrar simplicidade e qualidade do agrupamento;\n"
                     "- Se não houver cotovelo claro, teste k próximos e confirme com outras métricas.",
            "silhouette": "**Como ler (vai de −1 a 1):**\n"
                          "- Quanto maior, melhor a separação (≈ 0.5 ou mais é bom);\n"
                          "- Próximo de 0 indica mistura entre grupos (sobreposição);\n"
                          "- Negativo sugere pontos mal alocados;\n"
                          "- Barras longas acima de 0 para cada cluster indicam grupos coesos.",
            "tsne": "**Como ler:**\n"
                    "- Projeção em 3D para visualizar proximidade entre amostras; os eixos não têm significado direto.\n"
                    "- Grupos bem separados no gráfico sugerem clusters distintos e coesos; misturas/overlaps indicam fronteiras difusas.\n"
                    "- Pequenas variações entre execuções são normais (o t‑SNE é estocástico)."
        }
        for plot_type in ["elbow", "silhouette", "tsne"]:
            try:
                st.markdown(f"Plot: {plot_type}")
                plot_model(obj, plot=plot_type, display_format="streamlit")
                if plot_type in captions_map:
                    st.caption(captions_map[plot_type])
            except Exception as e:
                st.info(f"{plot_type} não disponível para {escolha}: {e}")

        # Dendrograma
        st.subheader("Dendrograma (hierárquico)")
        if escolha == "hclust" or show_dendro_anyway:
            X_for_dendro = labeled_final.drop(columns=["Cluster"])
            make_dendrogram(X_for_dendro, sample_cap=250, method="ward")
            st.caption("**Como ler:**\n"
                       "- Cada ‘união’ representa mesclagem de grupos parecidos;\n"
                       "- A altura da união mostra o quão diferentes eles eram (quanto maior, mais diferentes);\n"
                       "- ‘Saltos’ grandes de altura sugerem fronteiras naturais entre clusters (corte horizontal pouco antes do salto).")
        else:
            st.info("Dendrograma é mais apropriado para hclust. Ative a opção na barra lateral para forçar exibição.")

        # Heatmap das médias
        st.subheader("Heatmap das médias por cluster")
        top_n = int(heatmap_topn) if heatmap_topn and heatmap_topn > 0 else None
        plot_cluster_means_heatmap(labeled_final, cluster_col="Cluster",
                                   zscore=heatmap_zscore, top_n_features=top_n)
        st.caption("**Como ler:**\n"
                   "- Cada célula mostra a média da variável no cluster (z‑score se ativado);\n"
                   "- Cores positivas = valores acima da média global;\n"
                   "- Cores negativas = valores abaixo da média global.\n"
                   "- Compare colunas para ver quais variáveis diferenciam os clusters e linhas para entender o ‘perfil’ de cada grupo.")

        # Comparação com rótulos verdadeiros (opcional)
        if ('label_col' in locals()) and label_col and label_col in df.columns:
            st.subheader("Comparação clusters × rótulos")
            if limit_rows and limit_rows > 0 and limit_rows < len(data_full):
                sampled_idx = data_full.sample(n=limit_rows, random_state=42).index
                truth_series = df.loc[sampled_idx, label_col].reset_index(drop=True)
            else:
                truth_series = df[label_col].reset_index(drop=True)

            if len(truth_series) == len(labeled_final):
                labeled_cmp = labeled_final.copy()
                labeled_cmp[label_col] = truth_series
                ctab = pd.crosstab(labeled_cmp["Cluster"], labeled_cmp[label_col])
                st.dataframe(ctab)
                mapping, purity = best_cluster_label_mapping(labeled_cmp, cluster_col="Cluster", truth_col=label_col)
                st.write("Mapeamento cluster → rótulo mais frequente:")
                st.json(mapping)
                st.write(f"Pureza global: {purity:.3f}")
            else:
                st.info("Não foi possível alinhar rótulos com a amostra usada no clustering.")

        # Downloads
        st.subheader("Downloads")
        st.download_button("Baixar clusters (CSV)", labeled_final.to_csv(index=False).encode("utf-8"), "clusters.csv")
        os.makedirs(os.path.join("results", "models"), exist_ok=True)
        save_model(obj, os.path.join("results", "models", "modelo_cluster"))
        with open(os.path.join("results", "models", "modelo_cluster.pkl"), "rb") as f:
            st.download_button("Baixar modelo (PKL)", f, "modelo_cluster.pkl")

# ============================================================
# Aba Relatório
# ============================================================
with relatorio_container:
    # ============================
    # RELATÓRIO DE RESULTADOS
    # ============================

    st.header("Relatório de Resultados da Clusterização")

    st.markdown("""
    Após a execução do processo de clusterização, o modelo **K-Means** foi identificado como o mais adequado
    para o dataset `Country-data.csv`, com base nas métricas internas obtidas:

    - **Silhouette ≈ 0.29** → separação moderada entre clusters (estrutura existente, mas com sobreposição leve);
    - **Calinski–Harabasz ≈ 54.4** → boa compactação e separação interna;
    - **Davies–Bouldin ≈ 1.0** → separação aceitável entre grupos.

    Esses valores indicam que o K-Means conseguiu capturar **padrões socioeconômicos distintos entre os países**,
    apesar de transições graduais entre alguns grupos — o que é esperado em dados de desenvolvimento humano e econômico.
    """)

    # Recuperar dataset rotulado
    if "labeled_final" in locals() or "labeled_final" in globals():
        labeled = labeled_final.copy()
    else:
        st.warning("Nenhum modelo executado ainda. Execute a clusterização antes de gerar o relatório.")
        st.stop()

    # ---------------------------
    # Interpretação geral
    # ---------------------------
    st.subheader("Interpretação Geral dos Clusters")

    st.markdown("""
    O modelo K-Means formou **4 clusters principais**, representando diferentes estágios de desenvolvimento econômico e social:

    | Cluster  | Descrição  | Características predominantes |
    |----------|------------|-------------------------------|
    | **0** | Países em desenvolvimento intermediário | PIB e renda modestos, mortalidade infantil ainda alta, fertilidade elevada, expectativa de vida média. |
    | **1** | Países desenvolvidos                    | Alta renda e PIB per capita, elevada expectativa de vida, baixa mortalidade e fertilidade, inflação controlada, forte abertura comercial. |
    | **2** | Países de baixo desenvolvimento         | Renda e PIB baixos, mortalidade e fertilidade altas, expectativa de vida reduzida, inflação relativamente alta. |
    | **3** | País isolado de vulnerabilidade extrema (Nigéria) | Mortalidade e fertilidade extremamente altas, renda e expectativa de vida muito baixas, inflação muito elevada. |

    Esses agrupamentos refletem uma **progressão socioeconômica coerente**, indo de países de **vulnerabilidade extrema (Cluster 3)**
    até **economias desenvolvidas (Cluster 1)**, com **estágios intermediários (Clusters 0 e 2)**.
    """)

    # Preparar dataset exibível
    labeled_display = labeled.copy()
    if 'country' not in labeled_display.columns:
        df_full = st.session_state.get("cluster_df")
        if df_full is not None and 'country' in df_full.columns:
            labeled_display['country'] = df_full['country']
    # Métricas de tamanho/outliers para uso posterior
    cluster_counts = labeled_display['Cluster'].value_counts()
    isolated_clusters = cluster_counts[cluster_counts == 1].index.tolist()

    # ---------------------------
    # Mapa 2D via PCA
    # ---------------------------
    st.subheader("Visualização 2D dos Clusters (PCA)")

    st.markdown("""
    O Principal Component Analysis (PCA) é utilizado para resumir todas as variáveis em dois eixos (componentes PC1 e PC2) para visualização, sendo que, neste caso, cada ponto na imagem representa um país.  
    - Pontos **próximos** indicam países com perfis socioeconômicos semelhantes.  
    - Pontos **distantes** indicam perfis distintos.  
    - As **cores** correspondem aos clusters identificados.
    """)

    from sklearn.decomposition import PCA
    import plotly.express as px

    cols_num = [c for c in labeled_display.columns if c not in ['country', 'Cluster']]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(labeled_display[cols_num])
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labeled_display['Cluster']
    df_pca['country'] = labeled_display['country']

    # Ordem fixa dos clusters para manter consistência de cores
    cluster_order = sorted(df_pca['Cluster'].unique())

    fig_pca = px.scatter(
        df_pca, x='PC1', y='PC2',
        color='Cluster', hover_name='country',
        title='Mapa Socioeconômico 2D (PCA)',
        category_orders={'Cluster': cluster_order},
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    st.caption("""
    **Leitura do gráfico:**
    - *Cluster 1* (desenvolvidos) forma um grupo compacto e distante, com indicadores homogêneos e altos níveis de renda e qualidade de vida.  
    - *Cluster 0* aparece como faixa intermediária de desenvolvimento.  
    - *Cluster 2* agrupa países de baixo desenvolvimento, com indicadores desfavoráveis.  
    - *Cluster 3* (Nigéria) surge isolado — reflexo de um perfil socioeconômico extremamente desfavorável.
    """)

    # ---------------------------
    # Mapa Mundial
    # ---------------------------
    st.subheader("Distribuição Geográfica dos Clusters")

    st.markdown("""
    O mapa abaixo mostra a distribuição dos clusters por país.  
    Cores semelhantes indicam países com características socioeconômicas próximas.
    """)

    # Preparar mapa de cores do PCA para reutilizar em outras seções
    pca_color_map = {}
    try:
        for tr in getattr(fig_pca, 'data', []):
            cat_name = str(tr.name)
            tr_color = getattr(tr.marker, 'color', None)
            if isinstance(tr_color, (list, tuple)) and tr_color:
                tr_color = tr_color[0]
            pca_color_map[cat_name] = tr_color

        df_map = labeled_display.copy()
        df_map['ClusterLabel'] = df_map['Cluster'].apply(lambda c: str(c))
        fig_map = px.choropleth(
            df_map,
            locations="country",
            locationmode="country names",
            color="ClusterLabel",
            title="Mapa Mundial por Cluster (Country-data)",
            category_orders={'ClusterLabel': [str(c) for c in cluster_order]},
            color_discrete_map=pca_color_map,
            labels={'ClusterLabel': 'Cluster'},
        )
        fig_map.update_layout(legend_title_text='Cluster')
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning("Não foi possível gerar o mapa geográfico. Verifique se os nomes dos países estão padronizados.")

    # ---------------------------
    # Tamanho dos clusters e outliers
    # ---------------------------
    st.subheader("Tamanho dos Clusters e Outliers")
    st.markdown("""
    O gráfico (doughnut) abaixo apresenta a quantidade de países distribuídos por cluster.  
    Cada setor representa a proporção de países em cada grupo.
    """)
    try:
        counts_df = cluster_counts.sort_index().rename_axis('Cluster').reset_index(name='Quantidade')
        counts_df['ClusterStr'] = counts_df['Cluster'].astype(str)
        fig_counts = px.pie(
            counts_df,
            names='ClusterStr',
            values='Quantidade',
            color='ClusterStr',
            category_orders={'ClusterStr': [str(c) for c in cluster_order]},
            color_discrete_map=pca_color_map,
            title='Participação de Países por Cluster',
            hole=0.5
        )
        fig_counts.update_layout(legend_title_text='Cluster')
        st.plotly_chart(fig_counts, use_container_width=True)
    except Exception:
        st.table(cluster_counts.sort_index())

    if isolated_clusters:
        st.markdown("### ⚠️ Cluster(s) Isolado(s) Detectado(s)")
        for c in isolated_clusters:
            isolated_country = labeled_display.loc[labeled_display['Cluster'] == c, 'country'].values[0]
            label = str(c)
            if not label.lower().startswith("cluster"):
                label = f"Cluster {label}"
            st.info(f"**{label}** contém apenas **{isolated_country}**, indicando um possível outlier com perfil socioeconômico extremo.")
        st.caption("""
        **Interpretação:**  
        O modelo detectou país(es) significativamente diferentes dos demais.  
        No caso atual, a **Nigéria** apresenta mortalidade infantil, fertilidade e inflação muito elevadas,
        além de baixa renda e expectativa de vida, justificando seu isolamento.
        """)

    # ---------------------------
    # Perfil socioeconômico médio
    # ---------------------------
    st.subheader("Perfil Socioeconômico Médio por Cluster")

    st.markdown("""
    O gráfico abaixo apresenta as médias normalizadas das variáveis socioeconômicas dentro de cada grupo.
    Valores positivos indicam médias **acima da média global** e negativos, **abaixo da média global**.
    """)

    import plotly.graph_objects as go

    mean_by_cluster = labeled_display.groupby('Cluster')[cols_num].mean()

    fig_profile = go.Figure()
    for c in mean_by_cluster.index:
        fig_profile.add_trace(go.Bar(
            x=mean_by_cluster.columns,
            y=mean_by_cluster.loc[c],
            name=str(c),
        ))
    fig_profile.update_layout(
        barmode='group',
        title='Perfil Socioeconômico Médio (valores normalizados)',
        xaxis_title='Variáveis',
        yaxis_title='Média normalizada',
        height=500,
        legend_title_text='Cluster',
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    st.caption("""
    **Interpretação:**
    - *Cluster 1* (**desenvolvidos**): maiores valores em **income**, **gdpp**, **health**, **life_expec**, **exports** e **imports**; menores em **child_mort**, **total_fer** e **inflation**.  
    - *Cluster 0*: indicadores médios, representando países em **desenvolvimento intermediário**.  
    - *Cluster 2* (**baixo desenvolvimento**): renda e PIB baixos e mortalidade/fertilidade elevadas.  
    - *Cluster 3* (**Nigéria**): perfil extremo — inflação e mortalidade muito altas, renda e expectativa de vida muito baixas.
    """)

    # ---------------------------
    # Distribuição por variável (oculta)
    # ---------------------------
    with st.expander("Distribuição por variável"):
        st.markdown("""
        Nesta seção são exibidos gráficos do tipo **boxplot** para cada variável selecionada, separados por cluster.
        O objetivo é comparar a **distribuição** das variáveis entre os grupos.
        
        Como interpretar:
        - **Linha central**: a mediana dos valores do cluster;
        - **Caixa**: intervalo interquartil (Q1–Q3), onde se concentram 50% dos valores;
        - **Bigodes**: variação típica (sem outliers) dos dados;
        - **Pontos fora da caixa**: possíveis outliers.
        
        Selecione abaixo as variáveis para explorar suas distribuições por cluster.
        """)
        candidate_vars = [v for v in cols_num if v in ['income','gdpp','life_expec','child_mort','total_fer','inflation','health','imports','exports']]
        default_vars = [v for v in ['health','gdpp','life_expec','child_mort'] if v in candidate_vars]
        selected_vars = st.multiselect("Selecione as variáveis para visualizar:", options=candidate_vars, default=default_vars)
        for var in selected_vars:
            df_box = labeled_display[['Cluster', var]].copy()
            df_box['ClusterStr'] = df_box['Cluster'].astype(str)
            fig_box = px.box(
                df_box, x='ClusterStr', y=var, color='ClusterStr',
                category_orders={'ClusterStr': [str(c) for c in cluster_order]},
                color_discrete_map=pca_color_map,
                points='outliers',
                title=f'Distribuição de {var} por Cluster'
            )
            fig_box.update_layout(xaxis_title='Cluster', legend_title_text='Cluster')
            st.plotly_chart(fig_box, use_container_width=True)

    # ---------------------------
    # Tabela de países e clusters
    # ---------------------------
    st.subheader("Países e seus Clusters")
    st.markdown("""
    A tabela abaixo mostra cada país e o grupo (cluster) ao qual foi atribuído.
    Os países estão ordenados por cluster para facilitar a interpretação dos agrupamentos.
    """)
    st.dataframe(
        labeled_display[['country', 'Cluster']].sort_values(by='Cluster'),
        use_container_width=True,
    )

    # ---------------------------
    # Conclusões
    # ---------------------------
    st.subheader("Conclusões e Próximos Passos")

    st.markdown("""
    **Resumo da análise:**
    - O **modelo K-Means (k = 4)** capturou uma estrutura coerente de desenvolvimento socioeconômico global.  
    - *Cluster 1* agrupa países **desenvolvidos**; *Cluster 0* representa países em estágio **intermediário**; *Cluster 2* reúne **países de baixo desenvolvimento**; *Cluster 3* (**Nigéria**) destaca-se como caso isolado e crítico.  
    - Os resultados reforçam a coerência entre os indicadores econômicos e sociais.

    **Sugestões de continuidade:**
    1. Testar valores alternativos de *k* para avaliar estabilidade dos grupos.  
    2. Aplicar transformação de potência (`transformation=True`) para suavizar outliers.  
    3. Incluir variáveis adicionais (educação, IDH, desigualdade, urbanização).  
    4. Avaliar evolução temporal para estudar mudanças nos clusters ao longo dos anos.  

    Essas análises complementares podem reforçar o entendimento das diferenças estruturais entre os países
    e apoiar políticas públicas e estratégias de desenvolvimento mais direcionadas.
    """)
