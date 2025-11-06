"""Feature processing utilities."""
import pandas as pd
import numpy as np
import sys
import os

# Handle imports for both module and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.models import CarFeatures
    from api.model_loader import ModelLoader
else:
    from .models import CarFeatures
    from .model_loader import ModelLoader

CURRENT_YEAR = 2025

class FeatureProcessor:
    """Processes car features for model prediction."""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
    
    def calculate_derived_features(self, car: CarFeatures) -> dict:
        """Calculate derived features like car_age and km_per_year."""
        car_age = CURRENT_YEAR - car.ano
        if car_age <= 0:
            car_age = 0.5
        
        quilometragem = car.quilometragem if car.quilometragem is not None else 0
        km_per_year = quilometragem / car_age if car_age > 0 else 0
        if not np.isfinite(km_per_year):
            km_per_year = 0
        
        return {
            'car_age': car_age,
            'km_per_year': km_per_year,
            'quilometragem_clean': quilometragem
        }
    
    def build_features_dict(self, car: CarFeatures) -> dict:
        """Build complete features dictionary from car input."""
        derived = self.calculate_derived_features(car)
        
        # Numeric features
        features = {
            'car_age': derived['car_age'],
            'km_per_year': derived['km_per_year'],
            'motor_clean': car.motor,
            'quilometragem_clean': derived['quilometragem_clean'],
            'final_de_placa': car.final_de_placa,
            'portas_clean': car.portas,
            'potencia_clean': car.potencia,
        }
        
        # Categorical features with mappings
        features.update({
            'marca': self.model_loader.get_brand_mapping(car.marca),
            'state_clean': self.model_loader.get_state_mapping(car.state),
            'câmbio': car.cambio,
            'combustível': car.combustivel,
            'direção': car.direcao,
            'cor': car.cor,
            'tipo_de_veículo': car.tipo_de_veiculo,
            'tipo_de_direção': car.tipo_de_direcao,
            'possui_kit_gnv': car.possui_kit_gnv,
        })
        
        # Boolean features - get all boolean fields from CarFeatures
        boolean_fields = [
            'air_bag', 'ar_condicionado', 'alarme', 'controle_automatico_de_velocidade',
            'trava_eletrica', 'vidro_eletrico', 'ipva_pago', 'pneus_novos',
            'sensor_de_re', 'historico_veicular', 'aceita_trocas', 'garantia_de_3_meses',
            'laudo_veicular', 'camera_de_re', 'com_manual', 'com_garantia',
            'entrega_do_veiculo', 'computador_de_bordo', 'transferencia_de_documentacao',
            'carro_de_leilao', 'rodas_de_liga_leve', 'unico_dono', 'conexao_usb',
            'bancos_de_couro', 'interface_bluetooth', 'higienizacao_do_veiculo',
            'tracao_4x4', 'tanque_cheio', 'laudo_cautelar', 'chave_reserva', 'som',
            'com_multas', 'primeira_revisao_gratis', 'blindado', 'navegador_gps',
            'revisoes_feitas_em_concessionaria', 'ipva_gratis', 'apoio_na_documentacao',
            'teto_solar', 'veiculo_em_financiamento', 'garantia_3_meses', 'veiculo_quitado',
            'financiado', 'garantia_do_motor', 'volante_multifuncional',
            'com_garantia_de_fabrica', 'com_chave_reserva'
        ]
        
        for field in boolean_fields:
            features[field] = getattr(car, field, False)
        
        return features
    
    def prepare_features(self, car: CarFeatures) -> pd.DataFrame:
        """Prepare features DataFrame for model prediction."""
        features_dict = self.build_features_dict(car)
        df = pd.DataFrame([features_dict])
        
        # Get expected columns
        expected_cols = self.model_loader.get_expected_columns()
        if expected_cols is None:
            # Fallback: use columns from features_dict
            return df
        
        # Ensure all expected columns exist with correct types
        for col in expected_cols:
            if col not in df.columns:
                dtype = self.model_loader.get_column_dtype(col)
                if dtype and 'bool' in str(dtype):
                    df[col] = False
                elif dtype and ('float' in str(dtype) or 'int' in str(dtype)):
                    df[col] = 0.0
                else:
                    df[col] = None
            else:
                # Ensure type matches training data
                dtype = self.model_loader.get_column_dtype(col)
                if dtype:
                    if 'bool' in str(dtype):
                        df[col] = df[col].astype(bool)
                    elif 'int' in str(dtype):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif 'float' in str(dtype):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
        
        # Return DataFrame with all expected columns in the correct order
        return df[expected_cols]

