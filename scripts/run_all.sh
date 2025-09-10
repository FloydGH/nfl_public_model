#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.ingest.nfl_stats 2024
python -m src.ingest.odds
python -m src.features.game_priors
python -m src.models.project
if [ -f data/out/dk_with_proj.csv ]; then
  python -m src.models.ownership_fieldsim 80000
  python -m src.export.export_dk_csv
fi
