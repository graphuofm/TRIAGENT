# Layer L1: Committee Data Collection

## Goal

Run all three agents (VADER, FinBERT, GPT-4o-mini) on the full Financial
PhraseBank and produce a single CSV that all downstream experiments depend on.

This is the **foundation layer**. Get this wrong and everything is poisoned.

## Output Schema: `results/data/committee_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `sentence_id` | int | Unique row id |
| `sentence` | str | Original sentence |
| `gold_label` / `label_text` | str | "positive"/"neutral"/"negative" |
| `label` | int | 0=neg, 1=neu, 2=pos (sklearn-friendly) |
| `vader_score` | float | VADER compound score [-1, 1] |
| `vader_label` | str | VADER prediction |
| `vader_confidence` | float | abs(compound) |
| `vader_latency_ms` | float | per-sample latency |
| `vader_cost_usd` | float | always 0 |
| `vader_pos`, `vader_neg`, `vader_neu` | float | VADER component scores |
| `finbert_score` | float | P(pos) - P(neg) ∈ [-1, 1] |
| `finbert_label` | str | argmax label |
| `finbert_confidence` | float | max softmax prob |
| `finbert_latency_ms` | float | per-sample (batched-amortized) |
| `finbert_cost_usd` | float | GPU rental fraction |
| `finbert_prob_pos`, `finbert_prob_neg`, `finbert_prob_neu` | float | softmax probs |
| `llm_score` | float | LLM continuous sentiment [-1, 1] |
| `llm_label` | str | LLM categorical |
| `llm_confidence` | float | LLM self-reported confidence |
| `llm_reasoning` | str | LLM's explanation text |
| `llm_input_tokens` | int | prompt tokens |
| `llm_output_tokens` | int | completion tokens |
| `llm_cost_usd` | float | actual API cost |
| `llm_latency_ms` | float | API round-trip time |

## Critical Implementation Details

### LLM Prompt — USE EXACTLY THIS

The quality of `llm_reasoning` matters because L4 may use it as an additional
feature. Use this prompt:

```
You are an expert financial analyst. Classify the sentiment of the following 
financial sentence as exactly one of: positive, negative, or neutral.

IMPORTANT FINANCIAL DOMAIN RULES:
- Consider financial impact, not surface-level word sentiment
- "Cut costs" / "reduced expenses" → positive (improves profitability)
- "Loss narrowed" / "loss decreased" → positive (improving)
- "Profit declined" / "earnings missed" → negative
- "Liability" / "debt" / "short" are often neutral accounting terms

Respond with ONLY a JSON object, no markdown:
{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0,
 "confidence": 0.0 to 1.0, "reasoning": "brief"}

Sentence: "{sentence}"
```

Use OpenAI's `response_format={"type": "json_object"}` to enforce JSON output.
This was a flaky point in the original Colab — fix it here.

### LLM Cost Tracking

```python
LLM_PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},  # USD per 1M tokens
}

cost = (
    response.usage.prompt_tokens * pricing["input"] / 1_000_000 +
    response.usage.completion_tokens * pricing["output"] / 1_000_000
)
```

### FinBERT Cost Approximation

GPU rental ≈ $0.50/hour for T4 (adjust if using A100/H100). Per-sample cost:

```python
GPU_RENTAL_USD_PER_HR = 0.50
COST_PER_INFERENCE = (GPU_RENTAL_USD_PER_HR / 3600) * (LATENCY_MS / 1000)
```

This is small (~$2e-7 per sample) but tracked for honest accounting.

### FinBERT Label Mapping (note the order!)

```python
LABEL_MAP = {0: 'positive', 1: 'negative', 2: 'neutral'}  # ProsusAI's order
```

Continuous score: `P(positive) - P(negative)` ∈ [-1, 1]

## CLI Interface

```bash
# Quick test on 100 samples (skip LLM cost)
python experiments/L1_data_collection.py --sample 100 --skip-llm

# Quick test with LLM (~$0.05)
python experiments/L1_data_collection.py --sample 100

# Full run (~$2-3 LLM cost)
python experiments/L1_data_collection.py --yes
```

## Checkpoint Criteria

Run L1, then check these. **All must pass before L2.**

### 1. Reproduction (within 1pp)

```
VADER:   F1-Macro ≈ 0.4889
FinBERT: F1-Macro ≈ 0.8822
LLM:     F1-Macro ≈ 0.84 (may vary ±2pp due to LLM stochasticity)
```

### 2. Latency Sanity

```
VADER  per-sample: < 1ms
FinBERT per-sample: < 10ms (batched)
LLM    per-sample: 500-2000ms
```

### 3. Cost Sanity

Total LLM cost should be **between $1 and $5**. If outside this range:
- Too low: API not actually being called, check error handling
- Too high: prompt or response is way longer than expected, check token counts

### 4. Reasoning Quality

Manually inspect 10 random `llm_reasoning` values. They should be:
- 1-2 sentences
- Financially-grounded (mention "profit", "decline", "earnings", etc.)
- Not hallucinated, not generic

Bad example: "This sentence has positive sentiment because it sounds good."
Good example: "Operating profit declined despite revenue growth, indicating
margin compression — negative for shareholders."

If reasoning quality is poor, **tune the prompt before running full**.

## Common Failure Modes

- **HuggingFace dataset load fails** → use the Kaggle CSV fallback
- **GPU OOM in FinBERT** → reduce `batch_size` from 32 to 16 or 8
- **LLM JSON parse errors** → check `response_format` is set; check prompt
- **VADER scores all near 0** → confirm `vaderSentiment` (not `vader-sentiment`) installed
- **Cost vastly higher than expected** → log and abort if a single response 
  exceeds 500 input tokens; the prompt may be malformed

## --- COMPLETE REFERENCE CODE ---

Below is the complete L1 implementation. Other layers will give you specs 
only (not full code). L1 gets full code because it's the foundation.

### `src/config.py`

```python
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
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_DATA_DIR = RESULTS_DIR / "data"

for p in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"

# LLM Pricing (USD per 1M tokens) - UPDATE if pricing changes
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
```

### `src/agents/base.py`

```python
"""Abstract base class for all sentiment agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentOutput:
    label: str
    score: float
    confidence: float
    latency_ms: float
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class SentimentAgent(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, text: str) -> AgentOutput:
        ...

    def predict_batch(self, texts: list[str]) -> list[AgentOutput]:
        return [self.predict(t) for t in texts]
```

### `src/agents/vader_agent.py`

```python
"""L1 agent: VADER lexicon-based sentiment."""
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .base import SentimentAgent, AgentOutput


class VADERAgent(SentimentAgent):
    name = "vader"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        scores = self.analyzer.polarity_scores(text)
        latency = (time.perf_counter() - t0) * 1000

        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return AgentOutput(
            label=label,
            score=compound,
            confidence=abs(compound),
            latency_ms=latency,
            cost_usd=0.0,
            extra={
                'pos': scores['pos'],
                'neg': scores['neg'],
                'neu': scores['neu'],
            }
        )
```

### `src/agents/finbert_agent.py`

```python
"""L2 agent: FinBERT domain-specific sentiment."""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from .base import SentimentAgent, AgentOutput
from ..config import DEVICE


class FinBERTAgent(SentimentAgent):
    name = "finbert"
    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP = {0: 'positive', 1: 'negative', 2: 'neutral'}

    GPU_RENTAL_USD_PER_HR = 0.50
    AVG_LATENCY_MS = 3.6
    COST_PER_INFERENCE = (GPU_RENTAL_USD_PER_HR / 3600) * (AVG_LATENCY_MS / 1000)

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=512, padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        latency = (time.perf_counter() - t0) * 1000

        pred_idx = probs.argmax().item()
        label = self.LABEL_MAP[pred_idx]
        score = float(probs[0] - probs[1])  # P(pos) - P(neg)

        return AgentOutput(
            label=label,
            score=score,
            confidence=float(probs[pred_idx]),
            latency_ms=latency,
            cost_usd=self.COST_PER_INFERENCE,
            extra={
                'prob_pos': float(probs[0]),
                'prob_neg': float(probs[1]),
                'prob_neu': float(probs[2]),
            }
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[AgentOutput]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc='FinBERT'):
            batch = texts[i:i+batch_size]
            t0 = time.perf_counter()
            inputs = self.tokenizer(
                batch, return_tensors='pt', truncation=True,
                max_length=512, padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_latency = (time.perf_counter() - t0) * 1000
            per_sample_latency = batch_latency / len(batch)

            for j in range(len(batch)):
                p = probs[j]
                pred_idx = p.argmax().item()
                results.append(AgentOutput(
                    label=self.LABEL_MAP[pred_idx],
                    score=float(p[0] - p[1]),
                    confidence=float(p[pred_idx]),
                    latency_ms=per_sample_latency,
                    cost_usd=self.COST_PER_INFERENCE,
                    extra={
                        'prob_pos': float(p[0]),
                        'prob_neg': float(p[1]),
                        'prob_neu': float(p[2]),
                    }
                ))
        return results
```

### `src/agents/llm_agent.py`

```python
"""L3 agent: GPT-4o-mini with full token tracking."""
import time
import json
from openai import OpenAI
from tqdm import tqdm
from .base import SentimentAgent, AgentOutput
from ..config import OPENAI_API_KEY, LLM_MODEL, LLM_PRICING


PROMPT = """You are an expert financial analyst. Classify the sentiment of the following financial sentence as exactly one of: positive, negative, or neutral.

IMPORTANT FINANCIAL DOMAIN RULES:
- Consider financial impact, not surface-level word sentiment
- "Cut costs" / "reduced expenses" → positive (improves profitability)
- "Loss narrowed" / "loss decreased" → positive (improving)
- "Profit declined" / "earnings missed" → negative
- "Liability" / "debt" / "short" are often neutral accounting terms

Respond with ONLY a JSON object, no markdown:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "reasoning": "brief"}}

Sentence: "{sentence}"
"""


class LLMAgent(SentimentAgent):
    name = "llm"

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL):
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Check .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.pricing = LLM_PRICING[model]

    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": PROMPT.format(sentence=text)}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            latency = (time.perf_counter() - t0) * 1000

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            label = parsed['sentiment'].lower()
            score = float(parsed.get('score', 0.0))
            confidence = float(parsed.get('confidence', 0.8))
            reasoning = parsed.get('reasoning', '')

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens * self.pricing['input'] / 1_000_000 +
                output_tokens * self.pricing['output'] / 1_000_000
            )

            return AgentOutput(
                label=label, score=score, confidence=confidence,
                latency_ms=latency, cost_usd=cost,
                extra={
                    'reasoning': reasoning,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            )
        except Exception as e:
            return AgentOutput(
                label='neutral', score=0.0, confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000, cost_usd=0.0,
                extra={'reasoning': f'ERROR: {e}', 'input_tokens': 0, 'output_tokens': 0}
            )

    def predict_batch(self, texts: list[str], delay: float = 0.05) -> list[AgentOutput]:
        results = []
        for t in tqdm(texts, desc=f'LLM ({self.model})'):
            results.append(self.predict(t))
            if delay > 0:
                time.sleep(delay)
        return results
```

### `src/utils/data_loader.py`

```python
"""Load Financial PhraseBank with multiple fallbacks."""
import pandas as pd
from ..config import RAW_DIR, LABEL_MAP


def load_fpb() -> pd.DataFrame:
    """Returns DataFrame with columns: ['sentence', 'label_text', 'label']"""
    cache_path = RAW_DIR / "financial_phrasebank.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"✓ Loaded from cache: {len(df)} samples")
        return df

    # Method 1: HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        df = pd.DataFrame(ds['train'])
        df = df.rename(columns={'label': 'label_int'})
        df['label_text'] = df['label_int'].map(LABEL_MAP)
        df['label'] = df['label_int']
        df = df[['sentence', 'label_text', 'label']]
        df.to_csv(cache_path, index=False)
        print(f"✓ Loaded from HuggingFace: {len(df)} samples")
        return df
    except Exception as e:
        print(f"HF failed: {e}")

    # Method 2: HF Parquet direct
    try:
        url = ("https://huggingface.co/datasets/financial_phrasebank/"
               "resolve/refs%2Fconvert%2Fparquet/sentences_allagree/train/0000.parquet")
        df = pd.read_parquet(url)
        df['label_text'] = df['label'].map(LABEL_MAP)
        df.to_csv(cache_path, index=False)
        print(f"✓ Loaded from HF parquet: {len(df)} samples")
        return df
    except Exception as e:
        print(f"Parquet failed: {e}")

    raise RuntimeError(
        "Could not load FPB. Manual fallback: download all-data.csv from "
        "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news "
        f"and place at {cache_path}"
    )
```

### `experiments/L1_data_collection.py`

```python
"""L1: Committee Data Collection.

Run:
    python experiments/L1_data_collection.py --sample 100  # quick test
    python experiments/L1_data_collection.py --yes         # full run with LLM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR
from src.utils.data_loader import load_fpb
from src.agents.vader_agent import VADERAgent
from src.agents.finbert_agent import FinBERTAgent
from src.agents.llm_agent import LLMAgent
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    df = load_fpb()
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"⚠  Using sample of {len(df)} rows")

    df['sentence_id'] = range(len(df))

    # VADER
    print("\n=== L1: VADER ===")
    vader = VADERAgent()
    vader_results = vader.predict_batch(df['sentence'].tolist())
    df['vader_score'] = [r.score for r in vader_results]
    df['vader_label'] = [r.label for r in vader_results]
    df['vader_confidence'] = [r.confidence for r in vader_results]
    df['vader_latency_ms'] = [r.latency_ms for r in vader_results]
    df['vader_cost_usd'] = [r.cost_usd for r in vader_results]
    df['vader_pos'] = [r.extra['pos'] for r in vader_results]
    df['vader_neg'] = [r.extra['neg'] for r in vader_results]
    df['vader_neu'] = [r.extra['neu'] for r in vader_results]

    # FinBERT
    print("\n=== L2: FinBERT ===")
    finbert = FinBERTAgent()
    finbert_results = finbert.predict_batch(df['sentence'].tolist(), batch_size=32)
    df['finbert_score'] = [r.score for r in finbert_results]
    df['finbert_label'] = [r.label for r in finbert_results]
    df['finbert_confidence'] = [r.confidence for r in finbert_results]
    df['finbert_latency_ms'] = [r.latency_ms for r in finbert_results]
    df['finbert_cost_usd'] = [r.cost_usd for r in finbert_results]
    df['finbert_prob_pos'] = [r.extra['prob_pos'] for r in finbert_results]
    df['finbert_prob_neg'] = [r.extra['prob_neg'] for r in finbert_results]
    df['finbert_prob_neu'] = [r.extra['prob_neu'] for r in finbert_results]

    # LLM
    if not args.skip_llm:
        print("\n=== L3: GPT-4o-mini ===")
        print("⚠  This will incur API costs (~$1-3 for full FPB)")
        if not args.yes:
            confirm = input("Continue? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

        llm = LLMAgent()
        llm_results = llm.predict_batch(df['sentence'].tolist(), delay=0.05)
        df['llm_score'] = [r.score for r in llm_results]
        df['llm_label'] = [r.label for r in llm_results]
        df['llm_confidence'] = [r.confidence for r in llm_results]
        df['llm_latency_ms'] = [r.latency_ms for r in llm_results]
        df['llm_cost_usd'] = [r.cost_usd for r in llm_results]
        df['llm_reasoning'] = [r.extra['reasoning'] for r in llm_results]
        df['llm_input_tokens'] = [r.extra['input_tokens'] for r in llm_results]
        df['llm_output_tokens'] = [r.extra['output_tokens'] for r in llm_results]

        print(f"\n💰 Total LLM cost: ${df['llm_cost_usd'].sum():.3f}")

    # Save
    out_path = RESULTS_DATA_DIR / "committee_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")

    # Reproduction check
    print("\n=== Reproduction Check ===")
    agents = ['vader', 'finbert'] + (['llm'] if not args.skip_llm else [])
    for agent in agents:
        acc = accuracy_score(df['label_text'], df[f'{agent}_label'])
        f1 = f1_score(df['label_text'], df[f'{agent}_label'], average='macro')
        latency = df[f'{agent}_latency_ms'].mean()
        cost = df[f'{agent}_cost_usd'].sum()
        print(f"  {agent:>8}: Acc={acc:.4f}  F1={f1:.4f}  "
              f"Latency={latency:.2f}ms  TotalCost=${cost:.4f}")

    print("\nExpected baselines:")
    print("  VADER:   Acc≈0.5433  F1≈0.4889")
    print("  FinBERT: Acc≈0.8894  F1≈0.8822")
    print("  LLM:     Acc≈0.84    F1≈0.84")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--yes', '-y', action='store_true')
    args = parser.parse_args()
    main(args)
```

### `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
vaderSentiment>=3.3.2
transformers>=4.35
torch>=2.1
datasets>=2.14
sentence-transformers>=2.2
openai>=1.10
yfinance>=0.2.30
matplotlib>=3.7
seaborn>=0.12
xgboost>=2.0
statsmodels>=0.14
tqdm>=4.66
python-dotenv>=1.0
```

### `.env.example`

```
OPENAI_API_KEY=sk-proj-xxxxx-replace-this-with-your-real-key
```

### `.gitignore`

```
.env
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
data/raw/market_data/
results/data/*.csv
results/figures/*.png
paper/main.pdf
paper/main.aux
paper/main.log
.vscode/
.idea/
```
