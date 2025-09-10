.PHONY: install run own cheat topowned fmt clean

install:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && \
	python -m src.ingest.nfl_stats 2024 && \
	python -m src.ingest.odds && \
	python -m src.features.game_priors && \
	python -m src.models.project && \
	python -m src.features.apply_weather_lines && \
	python -m src.models.ownership_analytic && \
	python -m src.export.export_dk_csv

own:
	. .venv/bin/activate && python -m src.models.ownership_analytic && python -m src.export.export_dk_csv

cheat:
	. .venv/bin/activate && python scripts/ownership_cheatsheet.py

topowned:
	. .venv/bin/activate && python scripts/top_owned.py

fmt:
	. .venv/bin/activate && python - <<'PY'
from pathlib import Path
for p in Path("src").rglob("*.py"):
    s=p.read_text()
    s=s.replace("\t","    ")
    p.write_text(s)
print("tabs → spaces done")
PY

clean:
	rm -rf .venv data/interim/* data/out/*
