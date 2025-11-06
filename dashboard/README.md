# Dashboard Streamlit - Used Car Market Intelligence

Dashboard interativo para análise e predição de preços de carros usados.

## Instalação

Certifique-se de que todas as dependências estão instaladas:

```bash
pip install -r ../requirements.txt
```

## Execução

Para executar o dashboard, use um dos seguintes métodos:

### Método 1: Usando Streamlit diretamente

```bash
streamlit run dashboard/app.py
```

### Método 2: Usando o script helper

```bash
python dashboard/run_dashboard.py
```

### Método 3: A partir da raiz do projeto

```bash
cd dashboard
streamlit run app.py
```

## Funcionalidades

O dashboard possui 4 páginas principais:

1. **Início**: Visão geral com estatísticas rápidas do dataset
2. **Análise Exploratória**: Visualizações interativas com filtros
3. **Predição de Preço**: Interface para prever preços usando o modelo de ML
4. **Estatísticas**: Análises estatísticas detalhadas e insights

## Estrutura

- `app.py`: Aplicação principal do dashboard
- `run_dashboard.py`: Script helper para executar o dashboard
- `README.md`: Este arquivo

## Requisitos

- Python 3.8+
- Streamlit
- Todas as dependências listadas em `requirements.txt`

## Notas

O dashboard carrega automaticamente:
- O modelo treinado de `models/price_predictor_v1.pkl`
- Os dados processados de `data/processed/olx_cars_cleaned.csv`

Certifique-se de que esses arquivos existem antes de executar o dashboard.

