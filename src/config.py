import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
ROOT=Path(__file__).resolve().parents[1]
DATA_RAW=ROOT/"data/raw"
DATA_INTERIM=ROOT/"data/interim"
DATA_OUT=ROOT/"data/out"
DATA_EXTERNAL=ROOT/"data/external"
ODDS_API_KEY=os.getenv("ODDS_API_KEY","")
NWS_UA=os.getenv("NWS_USER_AGENT","nfl-public-model")
