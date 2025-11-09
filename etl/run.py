import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re

def limpar_colunas_localizacao(df: pd.DataFrame) -> pd.DataFrame:
    UFS_BRASIL = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
                  'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 
                  'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
    
    df['state_clean'] = df['state'].astype(str).str.upper().str.strip()
    df.loc[~df['state_clean'].isin(UFS_BRASIL), 'state_clean'] = np.nan
    
    df['zip_code_clean'] = df['zip_code'].astype(str).str.replace(r'\.0$', '', regex=True)
    df.loc[df['zip_code_clean'] == 'nan', 'zip_code_clean'] = np.nan
    
    df['city_clean'] = df['city'].astype(str).str.lower().str.strip()
    df.loc[df['city_clean'] == 'nan', 'city_clean'] = np.nan
    
    df['neighborhood_clean'] = df['neighborhood'].astype(str).str.lower().str.strip()
    df.loc[df['neighborhood_clean'] == 'nan', 'neighborhood_clean'] = np.nan
    
    def try_extract_state_from_city(row):
        if pd.isna(row['state_clean']) and pd.notna(row['city_clean']):
            match = re.search(r'\b(' + '|'.join(UFS_BRASIL) + r')$', row['city_clean'].upper())
            if match:
                return match.group(1)
        return row['state_clean']

    df['state_clean'] = df.apply(try_extract_state_from_city, axis=1)

    df['city_clean'] = df.apply(
        lambda row: re.sub(r'\b' + row['state_clean'] + r'$', '', row['city_clean'], flags=re.IGNORECASE).strip()
        if pd.notna(row['state_clean']) and pd.notna(row['city_clean']) else row['city_clean'],
        axis=1
    )
    
    df = df.dropna(subset=['state_clean'])
    
    return df

def clean_year(df):
    df = df.copy()
    df['ano_limpo'] = pd.to_numeric(df.get('ano'), errors='coerce')
    anos_do_titulo = df['title_list'].astype(str).str.extract(r'(\b\d{4}\b)', expand=False)
    df['ano_limpo'] = df['ano_limpo'].fillna(anos_do_titulo)
    df['ano_limpo'] = pd.to_numeric(df['ano_limpo'], errors='coerce')
    
    ano_atual = datetime.now().year
    df.loc[(df['ano_limpo'] < 1980) | (df['ano_limpo'] > (ano_atual + 1)), 'ano_limpo'] = np.nan
    
    df = df.fillna({'ano_limpo': df['ano_limpo'].median()})
    return df

def clean_price(df):
    df = df.copy()
    df['price_clean'] = df['price_list'].astype(str).str.replace(r'[R$.]', '', regex=True) \
                                         .str.replace(',', '.', regex=False) \
                                         .str.strip()
    df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    df.loc[(df['price_clean'] < 1000) | (df['price_clean'] > 1000000), 'price_clean'] = np.nan
    return df

def clean_km(df):
    df = df.copy()
    if 'km_list' in df.columns:
        df['km_clean'] = df['km_list'].astype(str).str.replace(r'[km.]', '', regex=True) \
                                       .str.replace(',', '.', regex=False) \
                                       .str.strip()
        df['km_clean'] = pd.to_numeric(df['km_clean'], errors='coerce')
        df.loc[(df['km_clean'] < 0) | (df['km_clean'] > 1000000), 'km_clean'] = np.nan
    
    if 'quilometragem' in df.columns:
        df['quilometragem_clean'] = pd.to_numeric(df['quilometragem'], errors='coerce')
        df.loc[(df['quilometragem_clean'] < 0) | (df['quilometragem_clean'] > 1000000), 'quilometragem_clean'] = np.nan
    
    return df

def clean_motor(df):
    df = df.copy()
    if 'motor_list' in df.columns:
        df['motor_clean'] = df['motor_list'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['motor_clean'] = pd.to_numeric(df['motor_clean'], errors='coerce')
        df.loc[(df['motor_clean'] < 0.5) | (df['motor_clean'] > 10), 'motor_clean'] = np.nan
    
    if 'potência_do_motor' in df.columns:
        df['potencia_clean'] = df['potência_do_motor'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['potencia_clean'] = pd.to_numeric(df['potencia_clean'], errors='coerce')
        df.loc[(df['potencia_clean'] < 0.5) | (df['potencia_clean'] > 10), 'potencia_clean'] = np.nan
    
    return df

def clean_portas(df):
    df = df.copy()
    if 'portas' in df.columns:
        df['portas_clean'] = df['portas'].astype(str).str.extract(r'(\d+)', expand=False)
        df['portas_clean'] = pd.to_numeric(df['portas_clean'], errors='coerce')
        df.loc[(df['portas_clean'] < 2) | (df['portas_clean'] > 5), 'portas_clean'] = np.nan
    return df

def clean_text_columns(df):
    df = df.copy()
    text_columns = ['marca', 'modelo', 'categoria', 'cor', 'combustível', 'câmbio', 'direção', 'tipo_de_veículo']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].replace(['nan', 'None', 'NULL', ''], np.nan)
    
    return df

def clean_boolean_columns(df):
    df = df.copy()
    boolean_columns = [col for col in df.columns if df[col].dtype == 'bool' or 
                       (df[col].dtype == 'object' and df[col].astype(str).str.strip().isin(['True', 'False', 'true', 'false', '1', '0', '']).all())]
    
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map({
                'True': True, 'true': True, '1': True,
                'False': False, 'false': False, '0': False,
                'nan': False, 'None': False, 'NULL': False, '': False
            }).fillna(False).astype(bool)
    
    return df

def remove_duplicates(df):
    df = df.copy()
    if 'url' in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['url'], keep='first')
        removed = initial_count - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate records based on URL")
    return df

def remove_dirty_columns(df):
    df = df.copy()
    
    dirty_to_clean = {
        'state': 'state_clean',
        'city': 'city_clean',
        'neighborhood': 'neighborhood_clean',
        'zip_code': 'zip_code_clean',
        'ano': 'ano_limpo',
        'price_list': 'price_clean',
        'km_list': 'km_clean',
        'quilometragem': 'quilometragem_clean',
        'motor_list': 'motor_clean',
        'potência_do_motor': 'potencia_clean',
        'portas': 'portas_clean',
    }
    
    columns_to_remove = [
        col for col, clean_col in dirty_to_clean.items()
        if col in df.columns and clean_col in df.columns
    ]
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} dirty columns: {', '.join(columns_to_remove)}")
    
    return df

def clean_data(df):
    print(f"Starting data cleaning. Initial shape: {df.shape}")
    
    df = remove_duplicates(df)
    df = clean_year(df)
    print(f"After cleaning year: {df.shape}")
    df = clean_price(df)
    print(f"After cleaning price: {df.shape}")
    df = clean_km(df)
    print(f"After cleaning mileage: {df.shape}")
    df = clean_motor(df)
    print(f"After cleaning motor: {df.shape}")
    df = clean_portas(df)
    print(f"After cleaning portas: {df.shape}")
    df = clean_text_columns(df)
    print(f"After cleaning text: {df.shape}")
    df = clean_boolean_columns(df)
    print(f"After cleaning boolean: {df.shape}")
    df = limpar_colunas_localizacao(df)
    print(f"After cleaning location: {df.shape}")
    
    critical_columns = ['price_clean', 'ano_limpo']
    before_drop = len(df)
    df = df.dropna(subset=critical_columns)
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"Removed {before_drop - after_drop} rows with missing critical data")
    
    df.drop(columns=[col for col in df.columns if df[col].nunique() <= 1], inplace=True)
    print(f"After removing constant columns: {df.shape}")

    df = remove_dirty_columns(df)
    print(f"After removing dirty columns: {df.shape}")
    
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    import sys
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        input_path = project_root / 'data' / 'raw' / 'olx_cars.csv'
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        
        if df.empty:
            raise ValueError("O arquivo de entrada está vazio")
        
        df_cleaned = clean_data(df)
        
        if df_cleaned.empty:
            raise ValueError("Após a limpeza, não restaram dados")
        
        output_path = project_root / 'data' / 'processed' / 'olx_cars_cleaned.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        
    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

