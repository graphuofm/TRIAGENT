"""Loaders for the 'customer-language' family of datasets.

  * load_tfns()              -- Twitter Financial News (English, ~12K)
  * load_finchina_sentiment()-- FinChinaSentiment (Chinese, ~5.7K, neg/neu only)

All loaders return the canonical:
    sentence    str
    label_text  one of {'negative', 'neutral', 'positive'}
    label       int per LABEL_TO_INT (negative=0, neutral=1, positive=2)
"""
from __future__ import annotations

import pandas as pd

from ..config import RAW_DIR, LABEL_TO_INT


# Twitter Financial News Sentiment label scheme:
#   0 = bearish (negative), 1 = bullish (positive), 2 = neutral
TFNS_LABEL_TO_TEXT = {0: 'negative', 1: 'positive', 2: 'neutral'}

# FinChinaSentiment label scheme (inferred from the dataset README + value
# distribution): the dataset has only NEGATIVE-skewed labels — no positive
# class is provided. We map to a 2-class problem (negative vs neutral)
# and surface the missing-positive caveat in any paper text using it.
#   '0' -> neutral
#   '-1' -> negative  (mild)
#   '-2' -> negative  (moderate)
#   '-3' -> negative  (strong)
# All -1/-2/-3 collapse to 'negative'; a positive class isn't represented.
FINCHINA_LABEL_TO_TEXT = {
    '0': 'neutral',
    '-1': 'negative',
    '-2': 'negative',
    '-3': 'negative',
}


def _from_tfns() -> pd.DataFrame:
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    all_ds = concatenate_datasets([ds[s] for s in ds.keys()])
    df = pd.DataFrame({
        'sentence':   all_ds['text'],
        'label_text': [TFNS_LABEL_TO_TEXT[int(l)] for l in all_ds['label']],
    })
    df['label'] = df['label_text'].map(LABEL_TO_INT).astype(int)
    df = df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    return df


def load_tfns() -> pd.DataFrame:
    """Twitter Financial News Sentiment — short conversational financial text."""
    cache_path = RAW_DIR / "twitter_financial_news.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"✓ Loaded TFNS from cache: {len(df)} samples")
        return df

    try:
        df = _from_tfns()
        df.to_csv(cache_path, index=False)
        print(f"✓ Loaded TFNS from HF: {len(df)} samples "
              f"(neg={int((df['label_text']=='negative').sum())}, "
              f"neu={int((df['label_text']=='neutral').sum())}, "
              f"pos={int((df['label_text']=='positive').sum())})")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Twitter Financial News Sentiment: {e}\n"
            "Manual fallback: download from "
            "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment "
            f"and save as {cache_path}"
        )


def _from_finchina() -> pd.DataFrame:
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("FinanceMTEB/FinChinaSentiment")
    all_ds = concatenate_datasets([ds[s] for s in ds.keys()])
    rows = []
    for ex in all_ds:
        lbl_raw = str(ex['label'])
        if lbl_raw not in FINCHINA_LABEL_TO_TEXT:
            continue
        rows.append({
            'sentence':   ex['sentence'],
            'label_text': FINCHINA_LABEL_TO_TEXT[lbl_raw],
        })
    df = pd.DataFrame(rows)
    df['label'] = df['label_text'].map(LABEL_TO_INT).astype(int)
    df = df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    return df


def load_finchina_sentiment() -> pd.DataFrame:
    """FinanceMTEB/FinChinaSentiment — Chinese financial news sentiment.

    CAVEAT: this dataset contains only `negative` (mild/moderate/strong
    collapsed) and `neutral` examples. There is **no positive class**.
    This makes it a 2-class subset of the canonical 3-class scheme.
    Treat F1 numbers reported on this dataset as 2-class (neg vs neu)
    and never compare directly to FPB / TFNS macro-F1. Mention the
    caveat explicitly in the paper.
    """
    cache_path = RAW_DIR / "finchina_sentiment.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"✓ Loaded FinChinaSentiment from cache: {len(df)} samples")
        return df

    try:
        df = _from_finchina()
        df.to_csv(cache_path, index=False)
        dist = df['label_text'].value_counts().to_dict()
        print(f"✓ Loaded FinChinaSentiment from HF: {len(df)} samples  "
              f"label dist={dist}  (NOTE: no positive class)")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FinChinaSentiment: {e}\n"
            "Manual fallback: download from "
            "https://huggingface.co/datasets/FinanceMTEB/FinChinaSentiment "
            f"and save as {cache_path}"
        )
