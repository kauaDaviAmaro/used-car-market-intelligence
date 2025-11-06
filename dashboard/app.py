import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import CarFeatures
from api.model_loader import ModelLoader
from api.feature_processor import FeatureProcessor

# Page configuration
st.set_page_config(
    page_title="Used Car Market Intelligence",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_model_and_data():
    """Load model and data once."""
    model_loader = ModelLoader()
    model_loader.load()
    feature_processor = FeatureProcessor(model_loader)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'processed', 'olx_cars_cleaned.csv')
    df = pd.read_csv(data_path)
    
    # Feature engineering
    CURRENT_YEAR = 2025
    df['log_price'] = np.log1p(df['price_clean'])
    df['car_age'] = CURRENT_YEAR - df['ano_limpo']
    df.loc[df['car_age'] <= 0, 'car_age'] = 0.5
    df['quilometragem_clean'] = df['quilometragem_clean'].fillna(0)
    df['km_per_year'] = df['quilometragem_clean'] / df['car_age']
    df['km_per_year'] = df['km_per_year'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return model_loader, feature_processor, df

# Load resources
try:
    model_loader, feature_processor, df = load_model_and_data()
except Exception as e:
    st.error(f"Erro ao carregar modelo ou dados: {str(e)}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Selecione uma página",
    ["Início", "Análise Exploratória", "Predição de Preço", "Estatísticas"]
)

# Home page
if page == "Início":
    st.markdown('<h1 class="main-header">Used Car Market Intelligence</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Carros", f"{len(df):,}")
    
    with col2:
        avg_price = df['price_clean'].mean()
        st.metric("Preço Médio", f"R$ {avg_price:,.0f}")
    
    with col3:
        total_brands = df['marca'].nunique()
        st.metric("Marcas Diferentes", total_brands)
    
    st.markdown("---")
    
    st.markdown("""
    ### Sobre o Dashboard
    
    Este dashboard oferece uma análise completa do mercado de carros usados, incluindo:
    
    - **Análise Exploratória**: Visualizações interativas dos dados
    - **Predição de Preço**: Interface para prever o preço de um carro
    - **Estatísticas**: Análises detalhadas do mercado
    
    ### Funcionalidades
    
    1. **Exploração de Dados**: Gráficos interativos sobre distribuição de preços, marcas, estados e características
    2. **Predição Inteligente**: Modelo de machine learning para estimar preços baseado em características do veículo
    3. **Análises Estatísticas**: Correlações, distribuições e insights do mercado
    """)
    
    # Quick stats
    st.markdown("### Estatísticas Rápidas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Marcas")
        top_brands = df['marca'].value_counts().head(10)
        st.bar_chart(top_brands)
    
    with col2:
        st.subheader("Distribuição por Estado")
        top_states = df['state_clean'].value_counts().head(10)
        st.bar_chart(top_states)

# Exploratory Data Analysis page
elif page == "Análise Exploratória":
    st.title("Análise Exploratória de Dados")
    
    # Filters
    st.sidebar.header("Filtros")
    
    # Brand filter
    all_brands = ['Todos'] + sorted(df['marca'].unique().tolist())
    selected_brand = st.sidebar.selectbox("Marca", all_brands)
    
    # State filter
    all_states = ['Todos'] + sorted(df['state_clean'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Estado", all_states)
    
    # Price range filter
    min_price = float(df['price_clean'].min())
    max_price = float(df['price_clean'].max())
    price_range = st.sidebar.slider(
        "Faixa de Preço (R$)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1000.0
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_brand != 'Todos':
        filtered_df = filtered_df[filtered_df['marca'] == selected_brand]
    if selected_state != 'Todos':
        filtered_df = filtered_df[filtered_df['state_clean'] == selected_state]
    filtered_df = filtered_df[
        (filtered_df['price_clean'] >= price_range[0]) & 
        (filtered_df['price_clean'] <= price_range[1])
    ]
    
    st.info(f"Mostrando {len(filtered_df):,} carros de {len(df):,} total")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Preços", "Marcas e Estados", "Características", "Correlações"])
    
    with tab1:
        st.subheader("Distribuição de Preços")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(
                filtered_df,
                x='price_clean',
                nbins=50,
                title='Distribuição de Preços',
                labels={'price_clean': 'Preço (R$)', 'count': 'Frequência'},
                height=400
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Log price distribution
            fig = px.histogram(
                filtered_df,
                x='log_price',
                nbins=50,
                title='Distribuição de Preços (Log)',
                labels={'log_price': 'Log(Preço)', 'count': 'Frequência'},
                height=400
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Price by age
        st.subheader("Preço vs Idade do Carro")
        # Filter out NaN values for size parameter
        scatter_df_age = filtered_df.dropna(subset=['quilometragem_clean', 'car_age', 'price_clean'])
        if len(scatter_df_age) > 0:
            fig = px.scatter(
                scatter_df_age,
                x='car_age',
                y='price_clean',
                color='marca',
                size='quilometragem_clean',
                hover_data=['marca', 'state_clean'],
                title='Preço vs Idade do Carro',
                labels={'car_age': 'Idade (anos)', 'price_clean': 'Preço (R$)'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para exibir este gráfico.")
        
        # Price by mileage
        st.subheader("Preço vs Quilometragem")
        # Filter out NaN values for size parameter
        scatter_df_km = filtered_df.dropna(subset=['motor_clean', 'quilometragem_clean', 'price_clean'])
        if len(scatter_df_km) > 0:
            fig = px.scatter(
                scatter_df_km,
                x='quilometragem_clean',
                y='price_clean',
                color='car_age',
                size='motor_clean',
                hover_data=['marca', 'state_clean'],
                title='Preço vs Quilometragem',
                labels={'quilometragem_clean': 'Quilometragem (km)', 'price_clean': 'Preço (R$)'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para exibir este gráfico.")
    
    with tab2:
        st.subheader("Análise por Marca")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top brands by count
            top_brands = filtered_df['marca'].value_counts().head(15)
            fig = px.bar(
                x=top_brands.values,
                y=top_brands.index,
                orientation='h',
                title='Top 15 Marcas (Quantidade)',
                labels={'x': 'Quantidade', 'y': 'Marca'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average price by brand
            avg_price_brand = filtered_df.groupby('marca')['price_clean'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(
                x=avg_price_brand.values,
                y=avg_price_brand.index,
                orientation='h',
                title='Top 15 Marcas (Preço Médio)',
                labels={'x': 'Preço Médio (R$)', 'y': 'Marca'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot by brand
        st.subheader("Distribuição de Preços por Marca")
        top_10_brands = filtered_df['marca'].value_counts().head(10).index
        df_top_brands = filtered_df[filtered_df['marca'].isin(top_10_brands)]
        
        fig = px.box(
            df_top_brands,
            x='marca',
            y='log_price',
            title='Distribuição de Preços (Log) por Marca',
            labels={'marca': 'Marca', 'log_price': 'Log(Preço)'},
            height=500
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # State analysis
        st.subheader("Análise por Estado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_states = filtered_df['state_clean'].value_counts().head(15)
            fig = px.bar(
                x=top_states.values,
                y=top_states.index,
                orientation='h',
                title='Top 15 Estados (Quantidade)',
                labels={'x': 'Quantidade', 'y': 'Estado'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_price_state = filtered_df.groupby('state_clean')['price_clean'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(
                x=avg_price_state.values,
                y=avg_price_state.index,
                orientation='h',
                title='Top 15 Estados (Preço Médio)',
                labels={'x': 'Preço Médio (R$)', 'y': 'Estado'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Características dos Veículos")
        
        # Transmission type
        col1, col2 = st.columns(2)
        
        with col1:
            if 'câmbio' in filtered_df.columns:
                cambio_counts = filtered_df['câmbio'].value_counts()
                fig = px.pie(
                    values=cambio_counts.values,
                    names=cambio_counts.index,
                    title='Distribuição por Tipo de Câmbio',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'combustível' in filtered_df.columns:
                fuel_counts = filtered_df['combustível'].value_counts()
                fig = px.pie(
                    values=fuel_counts.values,
                    names=fuel_counts.index,
                    title='Distribuição por Tipo de Combustível',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Luxury features impact
        st.subheader("Impacto de Características de Luxo")
        
        luxury_features = ['bancos_de_couro', 'teto_solar', 'tracao_4x4', 'blindado', 'unico_dono']
        available_features = [f for f in luxury_features if f in filtered_df.columns]
        
        if available_features:
            luxury_impact = {}
            for feature in available_features:
                with_feature = filtered_df[filtered_df[feature] == True]['price_clean'].mean()
                without_feature = filtered_df[filtered_df[feature] == False]['price_clean'].mean()
                luxury_impact[feature.replace('_', ' ').title()] = with_feature - without_feature
            
            impact_df = pd.DataFrame(list(luxury_impact.items()), columns=['Característica', 'Diferença de Preço'])
            impact_df = impact_df.sort_values('Diferença de Preço', ascending=True)
            
            fig = px.bar(
                impact_df,
                x='Diferença de Preço',
                y='Característica',
                orientation='h',
                title='Impacto Médio no Preço',
                labels={'Diferença de Preço': 'Diferença de Preço (R$)', 'Característica': 'Característica'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Matriz de Correlação")
        
        # Select numeric columns
        numeric_cols = ['log_price', 'car_age', 'km_per_year', 'quilometragem_clean', 'motor_clean']
        luxury_bool = ['bancos_de_couro', 'teto_solar', 'tracao_4x4', 'blindado', 'unico_dono']
        available_cols = [col for col in numeric_cols + luxury_bool if col in filtered_df.columns]
        
        corr_df = filtered_df[available_cols].corr()
        
        fig = px.imshow(
            corr_df,
            labels=dict(color="Correlação"),
            x=corr_df.columns,
            y=corr_df.columns,
            color_continuous_scale='RdBu',
            aspect="auto",
            title='Matriz de Correlação',
            height=600
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlação"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.subheader("Insights de Correlação")
        
        if 'log_price' in corr_df.columns:
            price_corr = corr_df['log_price'].drop('log_price').sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Maiores Correlações Positivas com Preço:**")
                for feature, corr in price_corr.head(5).items():
                    st.write(f"- {feature}: {corr:.3f}")
            
            with col2:
                st.write("**Maiores Correlações Negativas com Preço:**")
                for feature, corr in price_corr.tail(5).items():
                    st.write(f"- {feature}: {corr:.3f}")

# Price Prediction page
elif page == "Predição de Preço":
    st.title("Predição de Preço de Carros Usados")
    
    st.markdown("""
    Preencha as informações do veículo abaixo para obter uma predição de preço baseada em nosso modelo de machine learning.
    """)
    
    # Form for car features
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informações Básicas")
            
            ano = st.number_input("Ano do Veículo", min_value=1950, max_value=2025, value=2020, step=1)
            quilometragem = st.number_input("Quilometragem (km)", min_value=0.0, value=50000.0, step=1000.0)
            motor = st.number_input("Motor (litros)", min_value=0.0, value=1.6, step=0.1)
            marca = st.selectbox("Marca", sorted(df['marca'].unique().tolist()))
            state = st.selectbox("Estado", sorted(df['state_clean'].dropna().unique().tolist()))
            
            # Get available options from data
            cambio_options = [''] + sorted(df['câmbio'].dropna().unique().tolist())
            combustivel_options = [''] + sorted(df['combustível'].dropna().unique().tolist())
            direcao_options = [''] + sorted(df['direção'].dropna().unique().tolist())
            cor_options = [''] + sorted(df['cor'].dropna().unique().tolist())
            
            cambio = st.selectbox("Câmbio", cambio_options)
            combustivel = st.selectbox("Combustível", combustivel_options)
            direcao = st.selectbox("Direção", direcao_options)
            cor = st.selectbox("Cor", cor_options)
        
        with col2:
            st.subheader("Características Opcionais")
            
            portas = st.number_input("Número de Portas", min_value=2, max_value=5, value=4, step=1)
            potencia = st.number_input("Potência", min_value=0.0, value=None, step=1.0)
            final_de_placa = st.number_input("Final da Placa", min_value=0, max_value=9, value=None, step=1)
            
            st.subheader("Características de Luxo")
            
            col_lux1, col_lux2 = st.columns(2)
            
            with col_lux1:
                bancos_de_couro = st.checkbox("Bancos de Couro")
                teto_solar = st.checkbox("Teto Solar")
                tracao_4x4 = st.checkbox("Tração 4x4")
                blindado = st.checkbox("Blindado")
                unico_dono = st.checkbox("Único Dono")
                ar_condicionado = st.checkbox("Ar Condicionado")
                air_bag = st.checkbox("Air Bag")
                alarme = st.checkbox("Alarme")
            
            with col_lux2:
                sensor_de_re = st.checkbox("Sensor de Ré")
                camera_de_re = st.checkbox("Câmera de Ré")
                navegador_gps = st.checkbox("Navegador GPS")
                ipva_pago = st.checkbox("IPVA Pago")
                pneus_novos = st.checkbox("Pneus Novos")
                garantia_de_3_meses = st.checkbox("Garantia de 3 Meses")
                laudo_veicular = st.checkbox("Laudo Veicular")
                rodas_de_liga_leve = st.checkbox("Rodas de Liga Leve")
        
        submitted = st.form_submit_button("Prever Preço", use_container_width=True)
        
        if submitted:
            try:
                # Create CarFeatures object
                car = CarFeatures(
                    ano=float(ano),
                    quilometragem=quilometragem if quilometragem > 0 else None,
                    motor=motor if motor > 0 else None,
                    marca=marca,
                    state=state,
                    cambio=cambio if cambio else None,
                    combustivel=combustivel if combustivel else None,
                    direcao=direcao if direcao else None,
                    cor=cor if cor else None,
                    portas=float(portas) if portas else None,
                    potencia=potencia if potencia else None,
                    final_de_placa=float(final_de_placa) if final_de_placa is not None else None,
                    bancos_de_couro=bancos_de_couro,
                    teto_solar=teto_solar,
                    tracao_4x4=tracao_4x4,
                    blindado=blindado,
                    unico_dono=unico_dono,
                    ar_condicionado=ar_condicionado,
                    air_bag=air_bag,
                    alarme=alarme,
                    sensor_de_re=sensor_de_re,
                    camera_de_re=camera_de_re,
                    navegador_gps=navegador_gps,
                    ipva_pago=ipva_pago,
                    pneus_novos=pneus_novos,
                    garantia_de_3_meses=garantia_de_3_meses,
                    laudo_veicular=laudo_veicular,
                    rodas_de_liga_leve=rodas_de_liga_leve
                )
                
                # Prepare features
                features_df = feature_processor.prepare_features(car)
                
                # Make prediction
                model = model_loader.get_model()
                log_price_pred = model.predict(features_df)[0]
                price_pred = np.expm1(log_price_pred)
                
                # Display result
                st.success("Predição realizada com sucesso!")
                
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>Preço Estimado</h2>
                        <h1 style="font-size: 3rem; margin: 1rem 0;">R$ {price_pred:,.0f}</h1>
                        <p style="opacity: 0.9;">Baseado em {len(df):,} carros analisados</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                st.markdown("---")
                st.subheader("Informações Adicionais")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    car_age = 2025 - ano
                    st.metric("Idade do Veículo", f"{car_age} anos")
                
                with col2:
                    if quilometragem > 0 and car_age > 0:
                        km_per_year = quilometragem / car_age
                        st.metric("Km/Ano", f"{km_per_year:,.0f}")
                
                with col3:
                    # Compare with similar cars
                    similar = df[
                        (df['marca'] == marca) &
                        (df['ano_limpo'] >= ano - 2) &
                        (df['ano_limpo'] <= ano + 2)
                    ]
                    if len(similar) > 0:
                        avg_similar = similar['price_clean'].mean()
                        diff = price_pred - avg_similar
                        st.metric("vs. Similar no Mercado", f"R$ {diff:+,.0f}")
                
            except Exception as e:
                st.error(f"Erro ao fazer predição: {str(e)}")
                st.exception(e)

# Statistics page
elif page == "Estatísticas":
    st.title("Estatísticas Detalhadas")
    
    tab1, tab2, tab3 = st.tabs(["Resumo Estatístico", "Análise por Segmento", "Insights"])
    
    with tab1:
        st.subheader("Estatísticas Descritivas - Preços")
        
        price_stats = df['price_clean'].describe()
        st.dataframe(price_stats)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média", f"R$ {df['price_clean'].mean():,.0f}")
        with col2:
            st.metric("Mediana", f"R$ {df['price_clean'].median():,.0f}")
        with col3:
            st.metric("Desvio Padrão", f"R$ {df['price_clean'].std():,.0f}")
        with col4:
            st.metric("Coeficiente de Variação", f"{(df['price_clean'].std() / df['price_clean'].mean() * 100):.1f}%")
        
        st.subheader("Distribuição de Idade dos Veículos")
        age_stats = df['car_age'].describe()
        st.dataframe(age_stats)
        
        st.subheader("Distribuição de Quilometragem")
        km_stats = df['quilometragem_clean'].describe()
        st.dataframe(km_stats)
    
    with tab2:
        st.subheader("Análise por Segmento de Preço")
        
        # Define price segments
        df['price_segment'] = pd.cut(
            df['price_clean'],
            bins=[0, 30000, 60000, 100000, 200000, float('inf')],
            labels=['Econômico (<30k)', 'Popular (30k-60k)', 'Médio (60k-100k)', 'Alto (100k-200k)', 'Luxo (>200k)']
        )
        
        segment_analysis = df.groupby('price_segment').agg({
            'price_clean': ['count', 'mean', 'median'],
            'car_age': 'mean',
            'quilometragem_clean': 'mean',
            'motor_clean': 'mean'
        }).round(2)
        
        st.dataframe(segment_analysis)
        
        # Visualization
        fig = px.bar(
            x=df['price_segment'].value_counts().index,
            y=df['price_segment'].value_counts().values,
            title='Distribuição por Segmento de Preço',
            labels={'x': 'Segmento', 'y': 'Quantidade'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Características por Segmento")
        
        selected_segment = st.selectbox("Selecione um Segmento", df['price_segment'].unique())
        segment_df = df[df['price_segment'] == selected_segment]
        
        luxury_features = ['bancos_de_couro', 'teto_solar', 'tracao_4x4', 'blindado', 'ar_condicionado']
        available_features = [f for f in luxury_features if f in segment_df.columns]
        
        if available_features:
            feature_percentages = {}
            for feature in available_features:
                pct = (segment_df[feature].sum() / len(segment_df)) * 100
                feature_percentages[feature.replace('_', ' ').title()] = pct
            
            feature_df = pd.DataFrame(list(feature_percentages.items()), columns=['Característica', 'Percentual'])
            feature_df = feature_df.sort_values('Percentual', ascending=True)
            
            fig = px.bar(
                feature_df,
                x='Percentual',
                y='Característica',
                orientation='h',
                title=f'Percentual de Características no Segmento: {selected_segment}',
                labels={'Percentual': 'Percentual (%)', 'Característica': 'Característica'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Insights Principais")
        
        insights = [
            {
                "title": "Idade vs Preço",
                "content": f"A idade do carro tem uma correlação negativa forte de {df[['car_age', 'log_price']].corr().iloc[0,1]:.3f} com o preço. Carros mais novos tendem a ser significativamente mais caros."
            },
            {
                "title": "Quilometragem",
                "content": f"Quilometragem mostra correlação negativa de {df[['quilometragem_clean', 'log_price']].corr().iloc[0,1]:.3f} com preço. Quanto mais km rodados, menor o preço."
            },
            {
                "title": "Características de Luxo",
                "content": "Bancos de couro têm o maior impacto positivo no preço, seguidos por tração 4x4 e blindagem."
            },
            {
                "title": "Concentração Geográfica",
                "content": f"Mais de 50% dos carros estão concentrados em apenas 3 estados: {', '.join(df['state_clean'].value_counts().head(3).index.tolist())}."
            },
            {
                "title": "Distribuição de Preços",
                "content": f"O mercado é altamente concentrado em carros de até R$ {df['price_clean'].quantile(0.75):,.0f}, com apenas 25% dos carros acima deste valor."
            }
        ]
        
        for insight in insights:
            with st.expander(insight['title']):
                st.write(insight['content'])
        
        st.subheader("Comparações de Mercado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Marcas por Volume:**")
            top_5_brands = df['marca'].value_counts().head(5)
            for brand, count in top_5_brands.items():
                avg_price = df[df['marca'] == brand]['price_clean'].mean()
                st.write(f"- {brand}: {count} carros, Preço médio: R$ {avg_price:,.0f}")
        
        with col2:
            st.write("**Top 5 Marcas por Preço Médio:**")
            top_5_price = df.groupby('marca')['price_clean'].mean().sort_values(ascending=False).head(5)
            for brand, price in top_5_price.items():
                count = len(df[df['marca'] == brand])
                st.write(f"- {brand}: R$ {price:,.0f} (média), {count} carros")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Sobre
Dashboard desenvolvido para análise e predição de preços de carros usados.

**Versão:** 1.0.0
""")

