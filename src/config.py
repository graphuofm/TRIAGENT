"""Global configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LEXICONS_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_DATA_DIR = RESULTS_DIR / "data"

for p in [RAW_DIR, PROCESSED_DIR, LEXICONS_DIR,
          RESULTS_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Local LLM (default L3 agent — open-source, runs on a single GPU)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Approximate hourly rental cost for the GPU we're running L3 on, used to
# convert measured generation latency into a $-cost number (same accounting
# style as FinBERT). Default reflects an A5000-class card on common cloud
# providers; override via env var if you're on a bigger / cheaper machine.
LOCAL_LLM_GPU_USD_PER_HR = float(os.getenv("LOCAL_LLM_GPU_USD_PER_HR", "0.40"))

# OpenAI fallback (kept for optional comparison runs via openai_llm_agent.py)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
LLM_PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o":      {"input": 2.500, "output": 10.000},
}

# SDI thresholds
SDI_HIGH = 0.7
SDI_LOW = 0.3

# Labels
LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL_TO_INT = {v: k for k, v in LABEL_MAP.items()}
LABELS_ORDERED = ['positive', 'neutral', 'negative']

# GPU
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
