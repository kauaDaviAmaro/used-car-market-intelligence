import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    try:
        import pandas
        import numpy
        import joblib
        import xgboost
        import sklearn
        import fastapi
        import uvicorn
        import streamlit
        return True
    except ImportError as e:
        print(f"Dependência faltando: {e}")
        print("Execute: pip install -r requirements.txt")
        return False


def check_file_exists(file_path: Path, step_name: str) -> bool:
    if not file_path.exists():
        print(f"{step_name}: Arquivo não encontrado: {file_path}")
        return False
    return True


def run_scraping():
    try:
        script_path = project_root / "scrapping" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar scraping: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado no scraping: {e}")
        return False


def run_etl():
    input_file = project_root / "data" / "raw" / "olx_cars.csv"
    if not check_file_exists(input_file, "ETL"):
        print("Execute primeiro o scraping: python pipeline.py scraping")
        return False
    
    try:
        script_path = project_root / "etl" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar ETL: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado no ETL: {e}")
        return False


def run_features():
    input_file = project_root / "data" / "processed" / "olx_cars_cleaned.csv"
    if not check_file_exists(input_file, "Features"):
        print("Execute primeiro o ETL: python pipeline.py etl")
        return False
    
    try:
        script_path = project_root / "feature" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar feature engineering: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado na engenharia de features: {e}")
        return False


def run_train():
    input_file = project_root / "data" / "processed" / "olx_cars_cleaned.csv"
    if not check_file_exists(input_file, "Train"):
        print("Execute primeiro o ETL: python pipeline.py etl")
        return False
    
    try:
        script_path = project_root / "models" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar treinamento: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado no treinamento: {e}")
        return False


def run_api():
    model_file = project_root / "models" / "price_predictor_v1.pkl"
    if not model_file.exists():
        model_file = project_root / "models" / "price_predictor_v4.pkl"
        if not model_file.exists():
            print("Modelo não encontrado. Execute: python pipeline.py train")
            return False
    
    try:
        script_path = project_root / "api" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root)
        )
        return True
    except KeyboardInterrupt:
        return True
    except Exception as e:
        print(f"Erro ao iniciar API: {e}")
        return False


def run_dashboard():
    model_file = project_root / "models" / "price_predictor_v1.pkl"
    if not model_file.exists():
        model_file = project_root / "models" / "price_predictor_v4.pkl"
        if not model_file.exists():
            print("Modelo não encontrado. Execute: python pipeline.py train")
            return False
    
    try:
        script_path = project_root / "dashboard" / "run.py"
        if not script_path.exists():
            print(f"Script não encontrado: {script_path}")
            return False
        
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root)
        )
        return True
    except KeyboardInterrupt:
        return True
    except Exception as e:
        print(f"Erro ao iniciar dashboard: {e}")
        return False


def run_pipeline():
    steps = [
        ("Scraping", run_scraping),
        ("ETL", run_etl),
        ("Features", run_features),
        ("Train", run_train),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"Pipeline interrompido na etapa: {step_name}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Used Car Market Intelligence - Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python pipeline.py pipeline      # Executa todas as etapas em sequência
  python pipeline.py scraping       # Executa apenas o scraping
  python pipeline.py etl            # Executa apenas o ETL
  python pipeline.py features      # Executa apenas a engenharia de features
  python pipeline.py train          # Treina o modelo
  python pipeline.py api            # Inicia a API FastAPI
  python pipeline.py dashboard      # Inicia o dashboard Streamlit
        """
    )
    
    parser.add_argument(
        "command",
        choices=["scraping", "etl", "features", "train", "pipeline", "api", "dashboard"],
        help="Comando a executar"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Pula a verificação de dependências"
    )
    
    args = parser.parse_args()
    
    if not args.skip_checks and not check_dependencies():
        sys.exit(1)
    
    commands = {
        "scraping": run_scraping,
        "etl": run_etl,
        "features": run_features,
        "train": run_train,
        "pipeline": run_pipeline,
        "api": run_api,
        "dashboard": run_dashboard,
    }
    
    success = commands[args.command]()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

