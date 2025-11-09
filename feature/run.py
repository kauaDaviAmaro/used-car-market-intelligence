import pandas as pd
import numpy as np
import os

def build_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    if df.empty:
        raise ValueError("O DataFrame está vazio")
    
    required_cols = ['price_clean', 'ano_limpo', 'state_clean', 'marca']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas necessárias não encontradas: {missing_cols}")
    
    CURRENT_YEAR = 2025
    df['log_price'] = np.log1p(df['price_clean'])
    df['car_age'] = CURRENT_YEAR - df['ano_limpo']
    df.loc[df['car_age'] <= 0, 'car_age'] = 0.5
    
    df['quilometragem_clean'] = df['quilometragem_clean'].fillna(0)
    df['km_per_year'] = df['quilometragem_clean'] / df['car_age']
    df['km_per_year'] = df['km_per_year'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    state_counts = df['state_clean'].value_counts()
    threshold = 50 
    rare_states = state_counts[state_counts < threshold].index
    df['state_clean'] = df['state_clean'].replace(rare_states, 'STATE_OTHER')
    
    brand_counts = df['marca'].value_counts()
    top_20_brands = brand_counts.head(20).index
    df['marca'] = df['marca'].apply(lambda x: x if x in top_20_brands else 'BRAND_OTHER')
    
    features_v1 = [
        'log_price',
        'car_age',
        'km_per_year',
        'motor_clean',
        'marca',
        'state_clean',
        'câmbio',
        'combustível',
        'bancos_de_couro',
        'teto_solar',
        'tracao_4x4',
        'blindado',
        'unico_dono'
    ]
    
    available_features = [f for f in features_v1 if f in df.columns]
    missing_features = [f for f in features_v1 if f not in df.columns]
    
    if missing_features:
        print(f"Features não encontradas (serão ignoradas): {missing_features}")
    
    df_features = df[available_features]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)

if __name__ == "__main__":
    import sys
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        input_path = os.path.join(project_root, 'data', 'processed', 'olx_cars_cleaned.csv')
        output_path = os.path.join(project_root, 'data', 'features', 'olx_cars_features_v1.csv')
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
        
        build_features(input_path, output_path)
        
    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro durante a construção de features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

