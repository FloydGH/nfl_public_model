# nfl_public_model

Lightweight NFL DFS modeling pipeline:
- Ingest historical stats and odds
- Build team priors
- Generate projections
- Estimate ownership with a capped-simplex allocator

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m src.ingest.nfl_stats 2024
python -m src.ingest.odds
python -m src.features.game_priors
python -m src.models.project

# drop DraftKings DKSalaries.csv in data/external/
python -m src.features.apply_weather_lines
python -m src.models.ownership_analytic
python -m src.export.export_dk_csv

