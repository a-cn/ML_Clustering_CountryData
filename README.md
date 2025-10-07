## Projeto de Clusterização com PyCaret — Country Data (Kaggle)

Este projeto realiza aprendizado não supervisionado (clustering) sobre o dataset Country Data do Kaggle, com foco em identificar padrões ocultos entre países. A aplicação principal é um app em Streamlit que utiliza PyCaret para automatizar o pipeline de clusterização, avaliação e visualização.

### Objetivos
- **Principal**: implementar algoritmos de clustering (K-Means, Hierarchical Clustering e DBSCAN) com PyCaret para descobrir padrões no dataset `Country-data.csv`.
- **Secundário**: desenvolver habilidades práticas em análise exploratória de dados (EDA) e validação de modelos não supervisionados.

### Requisitos Funcionais (implementados)
- **Carregamento e Exploração de Dados**: suporte a datasets da biblioteca PyCaret (ex.: Iris) ou upload de CSV (neste caso, o Country-data.csv). Exibe preview e estatísticas iniciais.
- **Pré-processamento**: seleção de features numéricas, normalização opcional e PCA opcional.
- **Clustering**: aplicação de pelo menos 3 algoritmos (K-Means, Hierarchical Clustering e DBSCAN). O app também suporta BIRCH, OPTICS e Spectral (opcionais).
- **Validação e Métricas**: cálculo de Silhouette Score, Calinski–Harabasz e Davies–Bouldin.
- **Visualizações**: comparação de modelos, gráficos do PyCaret (elbow, silhouette, t-SNE), dendrograma para hierárquico, heatmap de médias por cluster.
- **Interpretação**: sumarização textual automática das métricas por modelo e seção com análise detalhada do modelo selecionado.

### Requisitos Não Funcionais
- **Performance**: projetado para até ~10.000 linhas em < 5 minutos. Use o limitador de amostras no app para garantir responsividade em máquinas modestas.
- **Usabilidade**: interface Streamlit clara, com descrições nos painéis laterais e outputs explicativos.
- **Confiabilidade**: tratamento básico de erros, validação de inputs e reprodutibilidade (semente fixa `session_id=42`).

---

## Estrutura de Pastas

```text
projeto_clustering_matematica/
  ├─ .gitignore
  ├─ environment.yml
  ├─ README.md
  ├─ data/
  │   └─ processed/
  │   └─ raw/
  │       └─ Country-data.csv                     # Dataset em CSV
  ├─ results/
  │   └─ models/
  │       └─ modelo_cluster.pkl                   # Arquivo em PKL gerado pelo app ao salvar o modelo
  │   └─ profiling/
  │       └─ pandas_profiling_country_data.html   # Relatório EDA gerado pelo ydata-profiling
  ├─ notebooks/
  │   ├─ analise_exploratoria.ipynb
  │   └─ data_preprocessing.ipynb
  └─ src/
      └─ app.py                                   # App Streamlit
```

---

## Ambiente

Este projeto usa Conda. As dependências estão em `environment.yml` (inclui PyCaret, Streamlit e utilitários para EDA e Kaggle API).

### Criação do ambiente

```bash
conda env create -f environment.yml
conda activate ambiente_matematica_computacional
```

---

## Dataset (Kaggle)

Fonte: `https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data`

Você pode obter o arquivo `Country-data.csv` de duas formas:

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

## Guia de Uso do App

1) **Fonte de Dados**
   - Escolha entre “Iris (Exemplo)” (via `pycaret.datasets.get_data("iris")`) ou “Upload CSV”.
   - Para Country-data, use “Upload CSV” e selecione `data/raw/Country-data.csv`.

2) **Configuração das Features**
   - Selecione as colunas numéricas a considerar. Se nenhuma for escolhida, o app usa automaticamente colunas numéricas não constantes.

3) **Pré-processamento**
   - Opções: Normalizar, PCA (com número de componentes ajustável).

4) **Parâmetros de Cluster**
   - Defina número de clusters `k` (para K-Means, Hierarchical, BIRCH, Spectral); parâmetros de densidade para DBSCAN/OPTICS.
   - Use o limitador de amostras para melhorar desempenho em datasets maiores.

5) **Rodar Clusterização**
   - Clique em “Rodar Clusterização”. O app:
     - Executa `setup` do PyCaret;
     - Treina os modelos selecionados;
     - Calcula métricas (Silhouette, Calinski–Harabasz, Davies–Bouldin) e exibe uma interpretação automática;
     - Permite escolher um modelo para análise detalhada.

6) **Análise Detalhada**
   - Visualizações PyCaret (elbow, silhouette, t-SNE) quando suportadas;
   - Dendrograma (para hierárquico ou sob demanda);
   - Heatmap das médias por cluster (com opção de z-score e seleção das top-N features).

7) **Comparação com Rótulos (opcional)**
   - Caso haja uma coluna de rótulo (ex.: `species`, `class`), é possível comparar clusters × rótulos e calcular pureza.

8) **Downloads**
   - Baixe CSV com os clusters atribuídos;
   - Salve e baixe o modelo treinado (`models/modelo_cluster.pkl`).

---

## Algoritmos e Métricas

- **Algoritmos**: K-Means, Hierarchical Clustering (agglomerative), DBSCAN. (Opcionalmente BIRCH, OPTICS, Spectral.)
- **Métricas**:
  - Silhouette Score — quanto mais alto, melhor separação;
  - Calinski–Harabasz — maior é melhor;
  - Davies–Bouldin — menor é melhor.

> Reprodutibilidade: o app fixa `session_id=42` no PyCaret.

---

## EDA (Análise Exploratória)

- Use os notebooks em `notebooks/` para EDA detalhada.
- O app mostra preview dos dados e oferece heatmap de médias por cluster para interpretação rápida.
- Pacotes incluídos (ex.: `ydata-profiling`, `seaborn`) podem ser usados nos notebooks para relatórios mais completos.

---

## Boas Práticas de Performance

- Utilize o parâmetro de limitação de amostras na barra lateral para manter a interação fluida com datasets maiores.
- Reduza dimensionalidade com PCA quando houver muitas features correlacionadas.
- Se necessário, selecione apenas um subconjunto de variáveis com maior relação ao problema.

---

## Interpretação e Relatório Final

Inclua no relatório (ou nas anotações do notebook):
- **Resumo Executivo**: número de clusters, características distintivas, e implicações práticas.
- **Metodologia**: algoritmos utilizados, parâmetros (e.g., `k`, `eps`, `min_samples`), justificativas técnicas.
- **Resultados e Insights**: métricas, gráficos, dendrogramas e hipóteses/descobertas relevantes.

---

## Troubleshooting

- Se o dataset não carregar via upload, verifique o separador (`,`), encoding e presença de cabeçalho.
- Para erros do PyCaret, confira se as colunas selecionadas são numéricas e se não há colunas constantes.
- Se ocorrer lentidão, limite amostras e/ou reduza o número de algoritmos simultâneos.

---

## Licença

Este repositório é para fins educacionais. Verifique a licença e termos do dataset no Kaggle antes de redistribuir.