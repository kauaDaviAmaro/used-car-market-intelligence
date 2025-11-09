# Used Cars Market Intelligence Pipeline

![Project Demo](videos/project.gif)

This is a complete, end-to-end data science project that scrapes, processes, and predicts used car prices from OLX (a major Brazilian classifieds site). The final model is served via a FastAPI endpoint and visualized in a Streamlit dashboard.

The champion model is a **Tuned XGBoost Regressor** that achieved:

  * **Final R² (Test Set): 0.8874**
  * **Final RMSE (Test Set): 0.1980** (on log-transformed price)

-----

## Architecture

```
┌─────────────────┐
│   OLX Website   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pipeline       │
│  (Scraping →    │
│   ETL →         │
│   Features →    │
│   Training)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Artifact │
│  (.pkl file)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  API   │ │Dashboard │
│FastAPI │ │Streamlit │
│:8000   │ │:8501     │
└────────┘ └──────────┘
```

-----

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kauadaviamaro/used-car-market-intelligence.git
cd used-car-market-intelligence

# Build and run all services (API + Dashboard)
docker-compose up --build
```

That's it. The services will be available at:
- **API:** `http://localhost:8000` (docs at `http://localhost:8000/docs`)
- **Dashboard:** `http://localhost:8501`

**Note:** If you don't have a trained model yet, run the pipeline first (see below).

-----

## Docker Commands

### Run Services

```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Run Pipeline (Generate Model)

If you need to generate the model from scratch:

```bash
# Run the full pipeline (scraping -> etl -> features -> train)
docker-compose run --rm pipeline python pipeline.py pipeline

# Or run individual steps
docker-compose run --rm pipeline python pipeline.py scraping
docker-compose run --rm pipeline python pipeline.py etl
docker-compose run --rm pipeline python pipeline.py features
docker-compose run --rm pipeline python pipeline.py train
```

### Rebuild Services

```bash
# Rebuild specific service
docker-compose build api
docker-compose build dashboard

# Rebuild all
docker-compose build --no-cache
```

-----

## Project Structure

```
used-car-market-intelligence/
│
├── api/                  # FastAPI application
│   ├── main.py           # API endpoints (/predict)
│   ├── model_loader.py   # Logic to load the .pkl model
│   ├── models.py         # Pydantic request/response models
│   └── run.py            # Script to run the API
│
├── dashboard/            # Streamlit dashboard
│   ├── app.py          # The dashboard UI code
│   └── run.py          # Script to run the dashboard
│
├── data/
│   ├── raw/            # Raw scraped .csv data
│   ├── processed/      # Cleaned .csv data (post-ETL)
│   └── features/       # Final feature-engineered .csv
│
├── etl/                # ETL scripts
│   └── run.py          # Cleans raw data
│
├── feature/            # Feature engineering scripts
│   └── run.py          # Creates V1 features (car_age, etc.)
│
├── models/             # Model training & final artifacts
│   ├── run.py          # Script to train and save the FINAL model
│   └── price_predictor_v1.pkl # The CHAMPION MODEL (V4)
│
├── scrapping/          # Playwright web scraper
│   ├── run.py          # Script to run scraping
│   └── olx_scraper.py  # Scraper implementation
│
├── pipeline.py          # Main pipeline orchestrator
├── docker-compose.yml   # Docker orchestration
├── Dockerfile.api      # API container
├── Dockerfile.dashboard # Dashboard container
├── Dockerfile.pipeline  # Pipeline container
└── requirements.txt    # Project dependencies
```

-----

## The Pipeline & Methodology

### 1. Scraping & ETL

  * The scraper (`scrapping/olx_scraper.py`) uses **Playwright** to handle dynamic JavaScript-loaded content on OLX, performing deep scraping to get vehicle details and optional extras.
  * The ETL script (`etl/run.py`) cleans the raw data, using Regex to extract `year` from titles and robustly parsing location data.

### 2. Feature Engineering

  * The "lab" (`notebooks/EDA.ipynb`) identified key features.
  * `feature/run.py` creates `log_price` (to normalize the target) and `car_age` (fixing a bug for future cars, `age <= 0`, by setting them to `0.5`).
  * **Key Strategy:** To handle high-cardinality features, `state_clean` was grouped into `STATE_OTHER` (for states with < 50 listings) and `marca` was grouped into `BRAND_OTHER` (for brands outside the Top 20).

### 3. Model Experimentation (The "V-Series")

The `notebooks/Model.ipynb` contains the full story of our model hunt. We used `scikit-learn`'s `Pipeline` and `ColumnTransformer` to ensure no data leakage.

| Model | Features | Algorithm | Test R² | Test RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **V1** | 12 (Baseline) | RandomForest | 0.8493 | 0.2291 |
| **V2** | 65 (All Features) | RandomForest | 0.8621 | 0.2191 |
| **V3** | 65 (All Features) | XGBoost (Default) | 0.8731 | 0.2102 |
| **V4** | **65 (All Features)** | **Tuned XGBoost** | **0.8874** | **0.1980** |

The V4 model was tuned with an exhaustive `RandomizedSearchCV` (**7500 total fits**) to find the optimal hyperparameters.

### 4. The Champion Model (V4)

The final script `models/run.py` retrains this V4 model on 100% of the data using the "golden" parameters found during tuning and saves the final `price_predictor_v1.pkl`.

  * **Algorithm:** `XGBRegressor`
  * **Golden Parameters:**
      * `n_estimators`: 700
      * `max_depth`: 5
      * `learning_rate`: 0.05
      * `subsample`: 0.8
      * `colsample_bytree`: 0.8

-----

## Local Development (Optional)

If you prefer to run without Docker:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
playwright install

# Run pipeline
python pipeline.py pipeline

# Run API
python pipeline.py api

# Run Dashboard
python pipeline.py dashboard
```

-----

## Project Goal

The objective was to build a full-stack data science portfolio piece, demonstrating competence in every stage of the MLOps lifecycle:

1.  **Data Acquisition:** Dynamic web scraping (Playwright).
2.  **ETL:** Robust data cleaning and validation (Pandas).
3.  **Feature Engineering:** Creating high-signal features and handling high-cardinality categorical data.
4.  **Model Experimentation:** Systematically testing and tuning models (from baseline to XGBoost).
5.  **Productionizing:** Saving a final model pipeline and serving it with a REST API (FastAPI).
6.  **Visualization:** Building a simple UI for interaction (Streamlit).
