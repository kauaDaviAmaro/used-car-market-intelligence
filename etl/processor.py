import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import re # Garanta que você importou o 're'

def limpar_colunas_localizacao(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa, padroniza e valida as colunas de localização (city, state, zip_code)."""
    
    # Lista de UFs válidas no Brasil (para validação)
    UFS_BRASIL = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
                  'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 
                  'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
    
    # 1. Limpeza e Padronização do Estado (State)
    # Garante que o estado está em maiúsculas e remove espaços.
    df['state_clean'] = df['state'].astype(str).str.upper().str.strip()
    
    # Se o valor não é um UF válido, marca como NaN
    df.loc[~df['state_clean'].isin(UFS_BRASIL), 'state_clean'] = np.nan
    
    # 2. Limpeza do CEP (zip_code)
    # Remove o '.0' que o Pandas adiciona ao converter float para string
    df['zip_code_clean'] = df['zip_code'].astype(str).str.replace(r'\.0$', '', regex=True)
    df.loc[df['zip_code_clean'] == 'nan', 'zip_code_clean'] = np.nan
    
    # 3. Limpeza de Cidade e Bairro
    df['city_clean'] = df['city'].astype(str).str.lower().str.strip()
    df.loc[df['city_clean'] == 'nan', 'city_clean'] = np.nan
    
    df['neighborhood_clean'] = df['neighborhood'].astype(str).str.lower().str.strip()
    df.loc[df['neighborhood_clean'] == 'nan', 'neighborhood_clean'] = np.nan
    
    # 4. FALLBACK ROBUSTO: Tenta achar o UF na coluna City se State for NaN
    
    def try_extract_state_from_city(row):
        """Função que tenta usar Regex para encontrar um UF válido no nome da cidade."""
        # Só executa se state_clean estiver faltando E city_clean existir
        if pd.isna(row['state_clean']) and pd.notna(row['city_clean']):
            # Regex que procura por um dos UFs no final da string da cidade
            match = re.search(r'\b(' + '|'.join(UFS_BRASIL) + r')$', row['city_clean'].upper())
            if match:
                # Se encontrou, retorna o UF
                return match.group(1)
        return row['state_clean']

    # Aplica o fallback
    df['state_clean'] = df.apply(try_extract_state_from_city, axis=1)

    # Remove o UF da string da cidade se ele foi encontrado lá (ex: "são paulo sp" -> "são paulo")
    df['city_clean'] = df.apply(
        lambda row: re.sub(r'\b' + row['state_clean'] + r'$', '', row['city_clean'], flags=re.IGNORECASE).strip()
        if pd.notna(row['state_clean']) and pd.notna(row['city_clean']) else row['city_clean'],
        axis=1
    )
    
    # Remove linhas onde o State (crucial para o preço) ainda está faltando.
    df = df.dropna(subset=['state_clean'])
    
    return df

def clean_year(df):
    """Clean and standardize year column."""
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
    """Clean and convert price to numeric."""
    df = df.copy()
    # Clean price_list column
    df['price_clean'] = df['price_list'].astype(str).str.replace(r'[R$.]', '', regex=True) \
                                         .str.replace(',', '.', regex=False) \
                                         .str.strip()
    # Convert to numeric
    df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    # Remove unrealistic prices (too low or too high)
    df.loc[(df['price_clean'] < 1000) | (df['price_clean'] > 1000000), 'price_clean'] = np.nan
    return df

def clean_km(df):
    """Clean and convert kilometer/mileage to numeric."""
    df = df.copy()
    # Clean km_list column
    if 'km_list' in df.columns:
        df['km_clean'] = df['km_list'].astype(str).str.replace(r'[km.]', '', regex=True) \
                                       .str.replace(',', '.', regex=False) \
                                       .str.strip()
        df['km_clean'] = pd.to_numeric(df['km_clean'], errors='coerce')
        # Remove unrealistic values (negative or too high)
        df.loc[(df['km_clean'] < 0) | (df['km_clean'] > 1000000), 'km_clean'] = np.nan
    
    # Also clean quilometragem column if it exists
    if 'quilometragem' in df.columns:
        df['quilometragem_clean'] = pd.to_numeric(df['quilometragem'], errors='coerce')
        df.loc[(df['quilometragem_clean'] < 0) | (df['quilometragem_clean'] > 1000000), 'quilometragem_clean'] = np.nan
    
    return df

def clean_motor(df):
    """Clean motor/power column."""
    df = df.copy()
    if 'motor_list' in df.columns:
        # Extract numeric value from motor_list (e.g., "1.8" from "1.8")
        df['motor_clean'] = df['motor_list'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['motor_clean'] = pd.to_numeric(df['motor_clean'], errors='coerce')
        # Remove unrealistic values
        df.loc[(df['motor_clean'] < 0.5) | (df['motor_clean'] > 10), 'motor_clean'] = np.nan
    
    if 'potência_do_motor' in df.columns:
        df['potencia_clean'] = df['potência_do_motor'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['potencia_clean'] = pd.to_numeric(df['potencia_clean'], errors='coerce')
        df.loc[(df['potencia_clean'] < 0.5) | (df['potencia_clean'] > 10), 'potencia_clean'] = np.nan
    
    return df

def clean_portas(df):
    """Clean doors (portas) column by extracting numeric value."""
    df = df.copy()
    if 'portas' in df.columns:
        # Extract numeric value from portas (e.g., "4 Portas" -> 4, "2 Portas" -> 2)
        df['portas_clean'] = df['portas'].astype(str).str.extract(r'(\d+)', expand=False)
        df['portas_clean'] = pd.to_numeric(df['portas_clean'], errors='coerce')
        # Validate: typical car doors are 2, 3, 4, or 5
        df.loc[(df['portas_clean'] < 2) | (df['portas_clean'] > 5), 'portas_clean'] = np.nan
    return df

def clean_text_columns(df):
    """Clean text columns by removing extra whitespace and normalizing."""
    df = df.copy()
    text_columns = ['marca', 'modelo', 'categoria', 'cor', 'combustível', 'câmbio', 'direção', 'tipo_de_veículo']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].replace(['nan', 'None', 'NULL', ''], np.nan)
    
    return df

def clean_boolean_columns(df):
    """Ensure boolean columns are properly formatted."""
    df = df.copy()
    # Find columns that should be boolean (True/False values)
    boolean_columns = [col for col in df.columns if df[col].dtype == 'bool' or 
                       (df[col].dtype == 'object' and df[col].astype(str).str.strip().isin(['True', 'False', 'true', 'false', '1', '0', '']).all())]
    
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Use map to avoid FutureWarning about downcasting
            df[col] = df[col].map({
                'True': True, 'true': True, '1': True,
                'False': False, 'false': False, '0': False,
                'nan': False, 'None': False, 'NULL': False, '': False
            }).fillna(False).astype(bool)
    
    return df

def remove_duplicates(df):
    """Remove duplicate records based on URL."""
    df = df.copy()
    if 'url' in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['url'], keep='first')
        removed = initial_count - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate records based on URL")
    return df

def remove_dirty_columns(df):
    """Remove original dirty columns that have been cleaned."""
    df = df.copy()
    
    # Map of dirty columns to their clean versions
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
    
    # Only remove columns that exist and have a clean version
    columns_to_remove = [
        col for col, clean_col in dirty_to_clean.items()
        if col in df.columns and clean_col in df.columns
    ]
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} dirty columns: {', '.join(columns_to_remove)}")
    
    return df

def clean_data(df):
    """Main function to clean all data."""
    print(f"Starting data cleaning. Initial shape: {df.shape}")
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Clean year
    df = clean_year(df)
    print(f"After cleaning year: {df.shape}")
    
    # Clean price
    df = clean_price(df)
    print(f"After cleaning price: {df.shape}")
    
    # Clean mileage
    df = clean_km(df)
    print(f"After cleaning mileage: {df.shape}")
    
    # Clean motor/power
    df = clean_motor(df)
    print(f"After cleaning motor: {df.shape}")
    
    # Clean doors (portas)
    df = clean_portas(df)
    print(f"After cleaning portas: {df.shape}")
    
    # Clean text columns
    df = clean_text_columns(df)
    print(f"After cleaning text: {df.shape}")
    
    # Clean boolean columns
    df = clean_boolean_columns(df)
    print(f"After cleaning boolean: {df.shape}")

    # Clean location columns
    df = limpar_colunas_localizacao(df)
    print(f"After cleaning location: {df.shape}")
    
    # Remove rows with missing critical data
    critical_columns = ['price_clean', 'ano_limpo']
    before_drop = len(df)
    df = df.dropna(subset=critical_columns)
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"Removed {before_drop - after_drop} rows with missing critical data")
    
    # Remove column with all the same value
    df.drop(columns=[col for col in df.columns if df[col].nunique() <= 1], inplace=True)
    print(f"After removing constant columns: {df.shape}")

    # Remove dirty columns that have been cleaned
    df = remove_dirty_columns(df)
    print(f"After removing dirty columns: {df.shape}")
    
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Get project root directory (parent of etl directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Read raw data
    input_path = project_root / 'data' / 'raw' / 'olx_cars.csv'
    print(f"Reading data from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    output_path = project_root / 'data' / 'processed' / 'olx_cars_cleaned.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")