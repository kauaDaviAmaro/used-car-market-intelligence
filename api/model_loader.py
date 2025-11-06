"""Model and data loader utilities."""
import pandas as pd
import numpy as np
import joblib
import os

class ModelLoader:
    """Handles model loading and data mappings."""
    
    def __init__(self):
        self.model = None
        self.training_data = None
        self.top_20_brands = None
        self.rare_states = None
        self.expected_columns = None
    
    def load(self):
        """Load model and prepare mappings."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Load model
        model_path = os.path.join(project_root, 'models', 'price_predictor_v4.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
        # Load training data for mappings
        data_path = os.path.join(project_root, 'data', 'processed', 'olx_cars_cleaned.csv')
        if os.path.exists(data_path):
            self.training_data = pd.read_csv(data_path)
            
            # Apply feature engineering (same as in train.py)
            CURRENT_YEAR = 2025
            self.training_data['log_price'] = np.log1p(self.training_data['price_clean'])
            self.training_data['car_age'] = CURRENT_YEAR - self.training_data['ano_limpo']
            self.training_data.loc[self.training_data['car_age'] <= 0, 'car_age'] = 0.5
            self.training_data['quilometragem_clean'] = self.training_data['quilometragem_clean'].fillna(0)
            self.training_data['km_per_year'] = self.training_data['quilometragem_clean'] / self.training_data['car_age']
            self.training_data['km_per_year'] = self.training_data['km_per_year'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Apply state mapping
            state_counts = self.training_data['state_clean'].value_counts()
            rare_states = state_counts[state_counts < 50].index
            self.training_data['state_clean'] = self.training_data['state_clean'].replace(rare_states, 'STATE_OTHER')
            self.rare_states = set(rare_states)
            
            # Apply brand mapping
            brand_counts = self.training_data['marca'].value_counts()
            top_20_brands = brand_counts.head(20).index
            self.training_data['marca'] = self.training_data['marca'].apply(lambda x: x if x in top_20_brands else 'BRAND_OTHER')
            self.top_20_brands = set(top_20_brands)
            
            # Get expected columns (after feature engineering)
            cols_to_drop = [
                'log_price', 'price_clean', 'url', 'title_list', 'description',
                'color_list', 'modelo', 'city_clean', 'neighborhood_clean', 'ano_limpo'
            ]
            valid_cols_to_drop = [col for col in cols_to_drop if col in self.training_data.columns]
            self.expected_columns = [
                col for col in self.training_data.columns if col not in valid_cols_to_drop
            ]
    
    def get_model(self):
        """Get loaded model."""
        return self.model
    
    def get_brand_mapping(self, marca: str) -> str:
        """Map brand to top 20 or BRAND_OTHER."""
        if self.top_20_brands and marca in self.top_20_brands:
            return marca
        return 'BRAND_OTHER'
    
    def get_state_mapping(self, state: str) -> str:
        """Map state to STATE_OTHER if rare."""
        if self.rare_states and state in self.rare_states:
            return 'STATE_OTHER'
        return state
    
    def get_expected_columns(self):
        """Get expected feature columns."""
        return self.expected_columns
    
    def get_column_dtype(self, col: str):
        """Get data type for a column."""
        if self.training_data is not None and col in self.training_data.columns:
            return self.training_data[col].dtype
        return None

