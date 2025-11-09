import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import olx_scraper as olx
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    try:
        car_data = olx.scrape_olx()
        
        if not car_data or len(car_data) == 0:
            print("Nenhum dado foi coletado")
            sys.exit(1)
        
        olx.save_data(car_data)
        
    except Exception as e:
        print(f"Erro durante o scraping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

