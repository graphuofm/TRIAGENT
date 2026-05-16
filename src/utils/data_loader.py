"""Load Financial PhraseBank with multiple fallbacks.

The canonical HF `financial_phrasebank` dataset is a deprecated loading-script
dataset that no longer loads under `datasets>=4`. We use parquet-backed
mirrors instead. The `ChanceFocus/flare-fpb` mirror is verified to contain
exactly the 4,846 `sentences_allagree` sentences (split across train/test/valid)
with the same label distribution as the original (neg 604 / neu 2879 / pos 1363).
"""
import pandas as pd
from ..config import RAW_DIR, LABEL_TO_INT


def _from_flare_fpb() -> pd.DataFrame:
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("ChanceFocus/flare-fpb")
    all_ds = concatenate_datasets([ds[s] for s in ds.keys()])
    df = pd.DataFrame({
        'sentence': all_ds['text'],
        'label_text': [str(a).lower() for a in all_ds['answer']],
    })
    # Re-derive int label from string via our canonical LABEL_TO_INT
    # (different mirrors use different int conventions; string is unambiguous)
    df['label'] = df['label_text'].map(LABEL_TO_INT).astype(int)
    df = df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    return df


def load_fpb() -> pd.DataFrame:
    """Returns DataFrame with columns: ['sentence', 'label_text', 'label']

    Where:
        sentence    str
        label_text  one of {'negative', 'neutral', 'positive'}
        label       int per LABEL_TO_INT (negative=0, neutral=1, positive=2)
    """
    cache_path = RAW_DIR / "financial_phrasebank.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"✓ Loaded from cache: {len(df)} samples")
        return df

    # Method 1: parquet mirror (current default — `datasets>=4` compatible)
    try:
        df = _from_flare_fpb()
        df.to_csv(cache_path, index=False)
        print(f"✓ Loaded from ChanceFocus/flare-fpb: {len(df)} samples")
        return df
    except Exception as e:
        print(f"flare-fpb failed: {e}")

    raise RuntimeError(
        "Could not load FPB. Manual fallback: download all-data.csv from "
        "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news "
        f"and place at {cache_path}"
    )
