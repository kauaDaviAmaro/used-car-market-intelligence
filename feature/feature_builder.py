import pandas as pd
import numpy as np
import os

def build_features(input_path, output_path):
    
    df = pd.read_csv(input_path)
    
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
    
    df_features = df[features_v1]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_features.to_csv(output_path, index=False)
    print(f"Feature V1 dataset saved to {output_path}. Shape: {df_features.shape}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'data', 'processed', 'olx_cars_cleaned.csv')
    output_path = os.path.join(project_root, 'data', 'features', 'olx_cars_features_v1.csv')
    build_features(input_path, output_path)