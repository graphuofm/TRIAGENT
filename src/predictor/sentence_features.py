"""Sentence-level (contextual) features for the L4 edge predictor.

Adds the third granularity to the word/phrase/sentence feature hierarchy:
  word    -> unigram triggers (existing)
  phrase  -> bigram/trigram triggers (existing)
  sentence-> sentence-BERT embeddings, PCA-reduced (this module)

The model is `paraphrase-multilingual-MiniLM-L12-v2` (~118 MB, ~10 ms/sentence
on CPU). Multilingual chosen so the same module works for the L7 Chinese pilot.
PCA target dim defaults to 16 — keeps the LR feature count manageable while
preserving most of the variance in the predictor task.

Keeping the embedding extraction separate from the L4 trainer means we cache
embeddings to disk once per dataset and reuse across ablation runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_DATA_DIR


# Default model — multilingual so works for English (FPB, TFNS) and Chinese.
DEFAULT_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PCA_DIM = 16


def _embedding_cache_path(tag: str, model: str) -> Path:
    safe = model.replace('/', '__')
    return RESULTS_DATA_DIR / f"sbert_emb_{tag}__{safe}.npy"


def _pca_cache_path(tag: str, model: str, dim: int) -> Path:
    safe = model.replace('/', '__')
    return RESULTS_DATA_DIR / f"sbert_pca{dim}_{tag}__{safe}.npy"


def encode_sentences(
    sentences: list[str],
    *,
    cache_tag: str,
    model_name: str = DEFAULT_SBERT,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Returns (N, D) embeddings. Cached to disk by `cache_tag` + model name."""
    cache = _embedding_cache_path(cache_tag, model_name)
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(sentences):
            print(f"[sbert] Cache hit: {cache.name} {emb.shape}")
            return emb
        print(f"[sbert] Cache stale ({emb.shape[0]} vs {len(sentences)}); recomputing")

    from sentence_transformers import SentenceTransformer
    print(f"[sbert] Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"[sbert] Encoding {len(sentences)} sentences (batch_size={batch_size})...")
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cos-similarity ready
    )
    np.save(cache, emb)
    print(f"[sbert] Saved {cache.name} {emb.shape}")
    return emb


def fit_pca(
    train_emb: np.ndarray,
    *,
    cache_tag: str,
    model_name: str = DEFAULT_SBERT,
    pca_dim: int = DEFAULT_PCA_DIM,
):
    """Returns a fitted PCA object. Cached projection cached separately."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(train_emb)
    print(f"[sbert] PCA dim={pca_dim}, explained variance={pca.explained_variance_ratio_.sum():.3f}")
    return pca


def add_sentence_features_to_df(
    df: pd.DataFrame,
    *,
    cache_tag: str,
    train_idx: np.ndarray | None = None,
    model_name: str = DEFAULT_SBERT,
    pca_dim: int = DEFAULT_PCA_DIM,
    text_col: str = 'sentence',
) -> tuple[pd.DataFrame, list[str]]:
    """Append `sbert_pca_0..D-1` columns to df.

    If train_idx is given, PCA is fit on the train rows only (no leakage).
    Otherwise PCA is fit on all rows (use only for cached/full-data analysis).

    Returns (df_with_features, list_of_new_column_names).
    """
    sentences = df[text_col].fillna('').astype(str).tolist()
    emb = encode_sentences(sentences, cache_tag=cache_tag, model_name=model_name)

    if train_idx is not None:
        pca = fit_pca(emb[train_idx], cache_tag=cache_tag,
                      model_name=model_name, pca_dim=pca_dim)
    else:
        pca = fit_pca(emb, cache_tag=cache_tag,
                      model_name=model_name, pca_dim=pca_dim)
    proj = pca.transform(emb)

    new_cols = [f'sbert_pca_{i}' for i in range(pca_dim)]
    out = df.copy()
    for i, col in enumerate(new_cols):
        out[col] = proj[:, i]
    return out, new_cols


def add_llm_reasoning_features_to_df(
    df: pd.DataFrame,
    *,
    cache_tag: str,
    reasoning_col: str = 'llm_reasoning',
    train_idx: np.ndarray | None = None,
    model_name: str = DEFAULT_SBERT,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> tuple[pd.DataFrame, list[str]]:
    """NON-DEPLOYABLE feature: encode the LLM's R1 reasoning text.

    Used as an upper-bound ablation in L4: shows how much extra signal is
    in the LLM's natural-language explanation, even if not edge-deployable.
    """
    if reasoning_col not in df.columns:
        raise KeyError(f"{reasoning_col!r} not in df — run L1 with LLM first")
    texts = df[reasoning_col].fillna('').astype(str).tolist()
    emb = encode_sentences(
        texts,
        cache_tag=f'reason_{cache_tag}',
        model_name=model_name,
    )
    if train_idx is not None:
        pca = fit_pca(emb[train_idx], cache_tag=f'reason_{cache_tag}',
                      model_name=model_name, pca_dim=pca_dim)
    else:
        pca = fit_pca(emb, cache_tag=f'reason_{cache_tag}',
                      model_name=model_name, pca_dim=pca_dim)
    proj = pca.transform(emb)
    new_cols = [f'reason_pca_{i}' for i in range(pca_dim)]
    out = df.copy()
    for i, col in enumerate(new_cols):
        out[col] = proj[:, i]
    return out, new_cols
