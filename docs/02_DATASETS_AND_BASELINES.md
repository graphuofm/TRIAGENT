# Datasets and Baselines

## Primary Dataset: Financial PhraseBank (FPB)

**Citation**: Malo et al. 2014, "Good debt or bad debt: Detecting semantic
orientations in economic texts"

**Subset**: `sentences_allagree` — highest annotation quality (5+ annotators 
all agreed), 4,846 sentences

**Label distribution**:
- Neutral: 2,879 (59.4%)
- Positive: 1,363 (28.1%)
- Negative: 604 (12.5%)

**Class imbalance note**: Negative class is small but disproportionately 
hard for VADER (the original paper's main finding). This is why F1-macro 
matters more than accuracy.

**Loading**:
```python
from datasets import load_dataset
ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
```

**Fallback** if HuggingFace fails: download `all-data.csv` from Kaggle:
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## Market Data (for L5 Backtest)

**Source**: yfinance (free, no API key)

**Tickers** (20 stocks across 4 sectors):
```python
TICKERS = [
    'TSLA', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META',  # Tech (7)
    'JPM', 'BAC', 'GS', 'V', 'MA',                              # Financial (5)
    'WMT', 'KO', 'PG', 'JNJ', 'PFE',                            # Consumer/Health (5)
    'XOM', 'CVX', 'BA',                                          # Energy/Industrial (3)
]  # Total: 20
DATE_RANGE = ('2023-01-01', '2024-12-31')
```

**Cache**: download once to `data/raw/market_data/{ticker}.csv` to avoid 
repeated yfinance calls.

## Optional: Chinese Financial News (L7 mini-experiment)

Build manually, ~150-200 sentences. Sources:
- 财联社 (cls.cn) — financial newswire
- 东方财富 (eastmoney.com)
- 雪球 (xueqiu.com) — investor commentary

Save as `data/raw/chinese_news.csv` with schema:
```
sentence_zh, sentence_en_translation, gold_label, source
```

For L7, only GPT-4o-mini is used (VADER and FinBERT don't handle Chinese 
robustly). The point is to show SDI generalizes — not a full benchmark.

## Baseline Models

### Single Models

| Model | Type | Source | Reference Performance on FPB |
|-------|------|--------|------------------------------|
| VADER | Lexicon | Hutto & Gilbert 2014 | Acc 54.3% / F1-Macro 48.9% |
| FinBERT | Domain BERT | `ProsusAI/finbert` | Acc 88.9% / F1-Macro 88.2% |
| GPT-4o-mini | Reasoning LLM | OpenAI | Acc ~84% / F1-Macro ~84% |

### Routing Strategies (the REAL competitors)

| ID | Name | Description |
|----|------|-------------|
| S0 | Always-L1 | All queries → VADER |
| S1 | Always-L2 | All queries → FinBERT |
| S2 | Always-L3 | All queries → GPT-4o-mini |
| S3 | Random-x% | x% random escalation L1→L2 |
| S4 | Confidence-x% | VADER low-confidence escalates L1→L2 |
| S5 | SDI-LE-x% | Top SDI_LE samples escalate L1→L2 ⭐ |
| S6 | SDI-Two-Stage | L1→L2 by SDI_LE; L2→L3 by SDI_ER ⭐⭐ |

⭐ = our basic method  
⭐⭐ = our full method (the headline contribution)

### Predictor Baselines (L4)

| Model | Features | Purpose |
|-------|----------|---------|
| Random | none | Worst-case baseline |
| LR — unigram | VADER + unigram triggers | Basic edge predictor |
| LR — uni+bigram | + bigram triggers | ⭐ Our main method |
| LR — uni+bi+trigram | + trigram triggers | Stretch |
| XGBoost — all | All features | Performance ceiling |

### Backtest Baselines (L5)

- Buy-and-Hold (sector-weighted)
- Oracle (gold labels — theoretical upper bound)
- Each single-agent strategy (Always-L1/L2/L3)
- Each routing strategy (S5, S6)

## Reproducibility Targets

After running L1 on full FPB:
```
VADER:   Acc≈0.5433  F1-Macro≈0.4889  Latency≈0.1ms   $0
FinBERT: Acc≈0.8894  F1-Macro≈0.8822  Latency≈3.6ms   ~$0
LLM:     Acc≈0.84    F1-Macro≈0.84    Latency≈1245ms  ~$2-3 total
```

Numbers may shift slightly if the LLM prompt is tuned — that's OK as long as 
the relative ordering holds (FinBERT > LLM > VADER on F1).

## Cost Sanity Reference

For L3's headline narrative:

```
Asset manager processing 100,000 queries/day, 252 trading days/year:

Always-L3 yearly cost ≈ $0.0003 × 100K × 252 = $7,560/year (single model)
                     OR ≈ $0.001 × 100K × 252 = $25,200/year (with longer prompts)
                     OR ≈ $30,000-50,000/year (typical real deployment)

SDI-Two-Stage Balanced ≈ ~15% of Always-L3 cost
                       ≈ $1,000-7,500/year
                       
Annual savings: $5,000-40,000+ per use case
```

These numbers make the ROI argument concrete in the Introduction.
