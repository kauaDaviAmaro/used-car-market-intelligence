import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

V4_GOLDEN_PARAMS = {
    'n_estimators': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

def load_and_prep_data(project_root):
    """Carrega, engenheira e limpa os dados para o V4."""
    
    input_path = os.path.join(project_root, 'data', 'processed', 'olx_cars_cleaned.csv')
    df = pd.read_csv(input_path)

    CURRENT_YEAR = 2025
    df['log_price'] = np.log1p(df['price_clean'])
    
    df['car_age'] = CURRENT_YEAR - df['ano_limpo']
    df.loc[df['car_age'] <= 0, 'car_age'] = 0.5
    
    df['quilometragem_clean'] = df['quilometragem_clean'].fillna(0)
    df['km_per_year'] = df['quilometragem_clean'] / df['car_age']
    df['km_per_year'] = df['km_per_year'].replace([np.inf, -np.inf], np.nan).fillna(0)

    state_counts = df['state_clean'].value_counts()
    rare_states = state_counts[state_counts < 50].index
    df['state_clean'] = df['state_clean'].replace(rare_states, 'STATE_OTHER')

    brand_counts = df['marca'].value_counts()
    top_20_brands = brand_counts.head(20).index
    df['marca'] = df['marca'].apply(lambda x: x if x in top_20_brands else 'BRAND_OTHER')

    y = df['log_price']
    
    cols_to_drop = ['log_price', 'price_clean', 'url', 'title_list', 'description', 
                    'color_list', 'modelo', 'city_clean', 'neighborhood_clean', 'ano_limpo']
    
    valid_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=valid_cols_to_drop)
    
    return X, y, X.columns

def build_preprocessor(X_columns):
    """Constrói o ColumnTransformer V2 (65 features)."""
    
    numeric_features = [col for col in X_columns if X_columns.dtype(col) in ['float64', 'int64']]
    categorical_features = [col for col in X_columns if X_columns.dtype(col) == 'object']
    boolean_features = [col for col in X_columns if X_columns.dtype(col) == 'bool']

    all_cols = X_columns.tolist()
    
    numeric_features = [col for col in ['car_age', 'km_per_year', 'motor_clean', 'quilometragem_clean', 'final_de_placa', 'portas_clean', 'potencia_clean', 'zip_code_clean'] if col in all_cols]
    categorical_features = [col for col in ['marca', 'state_clean', 'câmbio', 'combustível', 'direção', 'cor', 'tipo_de_veículo', 'tipo_de_direção', 'possui_kit_gnv'] if col in all_cols]
    bool_candidates = [col for col in all_cols if col not in numeric_features and col not in categorical_features]
    boolean_features = [col for col in bool_candidates if df[col].dtype == 'bool']


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', 'passthrough', boolean_features)
        ])
    
    return preprocessor

def main():
    print("--- Iniciando Treinamento do Modelo Final (V4) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print("Carregando e preparando dados...")
    X_full, y_full, X_columns = load_and_prep_data(project_root)
    
    print("Construindo preprocessor...")
    preprocessor = build_preprocessor(X_columns)
    
    print("Construindo pipeline final com XGBoost...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**V4_GOLDEN_PARAMS))
    ])
    
    print("Treinando o modelo em 100% dos dados...")
    model_pipeline.fit(X_full, y_full)
    print("Treinamento concluído.")
    
    output_path = os.path.join(project_root, 'models', 'price_predictor_v4.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Salvando modelo treinado em: {output_path}")
    joblib.dump(model_pipeline, output_path)
    print("--- Modelo salvo com sucesso! ---")

if __name__ == "__main__":
    main()