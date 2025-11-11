## Clusterização com PyCaret (Streamlit) — Country Data

Aplicativo Streamlit para aprendizado não supervisionado (clustering) com PyCaret 3.x, com foco em identificar 
padrões ocultos entre países. Permite carregar dados (exemplo Iris ou CSV), configurar pré‑processamento, treinar vários algoritmos, visualizar e interpretar resultados, e gerar um relatório completo em HTML.

### Destaques
- **Abas do app**: Orientações, EDA (profiling), Treino & Métricas, Relatório.
- **Fontes de dados**: Iris (PyCaret) ou upload de CSV (recomendado `data/raw/Country-data.csv`).
- **Pré‑processamento**: seleção de features numéricas, ignorar colunas (ex.: `country`), normalização, PCA opcional.
- **Modelos**: K‑Means, Hierarchical (hclust), DBSCAN, OPTICS, BIRCH, Spectral (seleção múltipla).
- **Métricas**: Silhouette, Calinski–Harabasz, Davies–Bouldin, com interpretação automática por modelo.
- **Visualizações**:
  - PyCaret: elbow, silhouette, t‑SNE (quando disponíveis);
  - Dendrograma (scipy);
  - Heatmap de médias por cluster (z‑score opcional e top‑N variáveis);
  - PCA 2D interativo;
  - Mapa mundial (choropleth por país/cluster);
  - Tamanho dos clusters em doughnut (percentual + quantidade);
  - Detecção de cluster(es) isolado(s);
  - Perfil socioeconômico médio por cluster (barras com valores até 3 casas decimais);
  - Distribuição por variável (boxplots das 9 variáveis).
- **Downloads**: clusters (CSV), modelo treinado (PKL) e, na aba Relatório, um botão único para baixar o **Relatório Completo em HTML** sobre os resultados da clusterização do dataset `Country-data.csv`.
- **UX**: botão flutuante “voltar ao topo” e textos de apoio em cada seção.

---

## Estrutura de Pastas

```text
projeto_clustering_matematica/
  ├─ environment.yml
  ├─ README.md
  ├─ data/
  │   ├─ raw/
  │   │   ├─ Country-data.csv                     # Dataset em CSV
  │   │   ├─ Country-data-dictionary.csv
  │   │   └─ unsupervised_learning_on_country_data.zip
  │   └─ processed/
  │       └─ Country-data_processed.csv           # Dataset sem a coluna "country"
  ├─ results/
  │   ├─ models/
  │   │   └─ modelo_cluster.pkl                   # Arquivo em PKL gerado pelo app ao salvar o modelo
  │   ├─ profiling/
  │   │   └─ pandas_profiling_country_data.html
  │   └─ relatorio_clusterizacao_completo.html    # Relatório em HTML dos resultados da clusterização
  ├─ notebooks/
  │   ├─ analise_exploratoria.ipynb
  │   ├─ clustering.ipynb
  │   └─ profiling.ipynb
  └─ src/
      └─ app.py                                   # App Streamlit
```

---

## Ambiente

Este projeto usa Miniconda. As dependências principais estão em `environment.yml` (PyCaret 3.x, Streamlit, scikit‑learn, plotly, scipy, matplotlib, ydata‑profiling/pandas‑profiling, etc.).

### Criar e ativar o ambiente (Conda)

```bash
conda env create -f environment.yml
conda activate ambiente_matematica_computacional
```

---

## Dataset (Kaggle)

Fonte: `https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data`

Você pode obter `Country-data.csv` via Kaggle API ou download manual e colocá-lo em `data/raw/`.

1) **Via Kaggle API**

```bash
# 1) Configure suas credenciais do Kaggle
#    Faça download do kaggle.json em: https://www.kaggle.com/<seu-usuario>/account
#    Coloque em: %USERPROFILE%/.kaggle/kaggle.json (Windows)

# 2) Baixe o dataset para a pasta data/
kaggle datasets download -d rohan0301/unsupervised-learning-on-country-data -p data
cd data && tar -xf unsupervised-learning-on-country-data.zip || unzip unsupervised-learning-on-country-data.zip && cd ..
```

2) **Download manual**: baixe pelo site do Kaggle e coloque `Country-data.csv` em `data/`.

---

## Executando o App

```bash
streamlit run src/app.py
```

Abra o link exibido no terminal (geralmente `http://localhost:8501`).

---

## Guia de Uso (resumo)

1) **Fonte de Dados**
   - Selecione “Iris (Exemplo)” ou “Upload CSV”. Para este projeto, use `data/raw/Country-data.csv`.

2) **EDA (profiling)**
   - Pré-visualize os dados, gere um relatório automático (ydata‑profiling/pandas‑profiling) e faça download do HTML em aba específica.

3) **Configuração das Features**
   - Ignore `country` (pré‑selecionada quando presente) e escolha as variáveis numéricas. Se nenhuma for escolhida, o app usa todas as numéricas não constantes.

4) **Pré‑processamento**
   - Normalização (recomendada) e PCA opcional (com número de componentes ajustável).

5) **Parâmetros e Desempenho**
   - Defina número de clusters `k` (para modelos baseados em k) e parâmetros de densidade para DBSCAN/OPTICS (`eps`, `min_samples`).
   - Use “Limitar amostras” para respostas mais rápidas em datasets maiores ou máquinas modestas.

6) **Rodar Clusterização**
   - Clique em “Rodar Clusterização”. O app:
     - Executa `setup` do PyCaret;
     - Treina os modelos selecionados;
     - Calcula métricas (Silhouette, Calinski–Harabasz, Davies–Bouldin) e exibe uma interpretação automática;
     - Permite escolher um modelo para análise detalhada.

7) **Análise Detalhada**
   - Visualizações PyCaret (elbow, silhouette, t‑SNE);
   - Dendrograma (para hierárquico ou sob demanda);
   - Heatmap das médias por cluster (com opção de z-score e seleção das top-N features);
   - PCA 2D, mapa mundial, tamanho dos clusters (doughnut) e detecção de clusters isolados;
   - Perfil socioeconômico médio (barras com rótulos em 3 casas) e distribuição por variável (boxplots).

8) **Downloads**
   - CSV com clusters atribuídos e modelo treinado (`results/models/modelo_cluster.pkl`).
   - Na aba “Relatório”, clique em “Baixar relatório completo (HTML)” para salvar um arquivo com os resultados da clusterização gerados pelo app (tabelas e gráficos interativos em HTML).

---

## Algoritmos e Métricas

- **Algoritmos**: K-Means, Hierarchical Clustering (agglomerative), DBSCAN, BIRCH, OPTICS, Spectral.
- **Métricas**:
  - Silhouette Score — quanto mais alto, melhor separação;
  - Calinski–Harabasz — maior é melhor;
  - Davies–Bouldin — menor é melhor.

> Reprodutibilidade: o app fixa `session_id=42` no PyCaret.

---

## Troubleshooting

- Se o dataset não carregar via upload, verifique o separador (`,`), encoding e presença de cabeçalho.
- Para erros do PyCaret, confira se as colunas selecionadas são numéricas e se não há colunas constantes.
- Se ocorrer lentidão, limite amostras e/ou reduza o número de algoritmos simultâneos.

---

## Licença

Este repositório é para fins educacionais. Verifique a licença e termos do dataset no Kaggle antes de redistribuir.