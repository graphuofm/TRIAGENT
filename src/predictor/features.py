"""Edge-side features for the L4 predictor.

All features must be computable from VADER outputs + the raw sentence
(NO FinBERT or LLM access). This is what makes the predictor deployable:
the router decides whether to escalate before any expensive model has run.

Phrase-level (bigram + trigram) features are load-bearing — see
`docs/10_AGENTIC_PIVOT.md` ("customer-language framing"): real customer
text is fragmentary, and phrase triggers are more diagnostic than single
words ("loss narrowed" → positive vs "loss widened" → negative).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTRAST_WORDS  = {'but', 'however', 'while', 'although', 'yet', 'though', 'whereas'}
NEGATION_WORDS  = {'not', 'no', 'never', 'without', 'cannot', 'cant', 'wont', 'didnt', 'doesnt'}
CURRENCY_REGEX  = re.compile(r'[$€£¥]|EUR|USD|GBP|JPY|CNY|RMB')
NUMBER_REGEX    = re.compile(r'\b\d+(?:[.,]\d+)?\b')
WORD_REGEX      = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NGRAM_TOKEN_RE  = re.compile(r'[a-zA-Z]{3,}')


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_REGEX.findall(text or '')]


def _has_contrast(tokens: list[str]) -> int:
    return int(any(t in CONTRAST_WORDS for t in tokens))


def _has_negation(tokens: list[str]) -> int:
    return int(any(t in NEGATION_WORDS for t in tokens) or "n't" in ' '.join(tokens))


def _ngrams(tokens: list[str], n: int) -> set[str]:
    if n == 1:
        return set(tokens)
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_trigger_lexicon(path: Path) -> dict:
    """Load mined n-gram lexicon JSON produced by `mine_ngram_triggers`.

    Returns a dict with keys 'n1', 'n2', 'n3' → set of trigger strings.
    """
    with open(path) as f:
        payload = json.load(f)
    out = {f'n{i}': set() for i in (1, 2, 3)}
    for k, items in payload.items():
        # k looks like 'n1_high_sdi' or 'n2_low_sdi'
        n_key = k.split('_')[0]  # 'n1', 'n2', 'n3'
        if n_key in out:
            out[n_key].update(item['ngram'] for item in items)
    return out


def _safe_feature_name(prefix: str, ngram: str) -> str:
    return f"{prefix}_" + re.sub(r'[^a-zA-Z0-9]+', '_', ngram).strip('_').lower()


def extract_features(
    df: pd.DataFrame,
    lexicon: dict,
    *,
    text_col: str = 'sentence',
    per_trigger_top_k: int = 30,
) -> pd.DataFrame:
    """Per-sentence feature matrix.

    Composed of:
      - 6 VADER outputs
      - 6 sentence-structure features
      - 6 aggregate n-gram-trigger counts/presence
      - up to 3 * per_trigger_top_k per-trigger binary features
        (so the LR can learn signed weights per trigger, not just an aggregate)
    """
    rows = []
    n1, n2, n3 = lexicon.get('n1', set()), lexicon.get('n2', set()), lexicon.get('n3', set())

    # Stable lists for the per-trigger features (sorted for reproducibility)
    n1_top = sorted(n1)[:per_trigger_top_k]
    n2_top = sorted(n2)[:per_trigger_top_k]
    n3_top = sorted(n3)[:per_trigger_top_k]

    n1_cols = {ng: _safe_feature_name('uni', ng) for ng in n1_top}
    n2_cols = {ng: _safe_feature_name('bi', ng) for ng in n2_top}
    n3_cols = {ng: _safe_feature_name('tri', ng) for ng in n3_top}

    for _, r in df.iterrows():
        text = str(r[text_col])
        tokens = [w.lower() for w in NGRAM_TOKEN_RE.findall(text)]
        wordlist = [w.lower() for w in WORD_REGEX.findall(text)]
        unigrams = _ngrams(tokens, 1)
        bigrams  = _ngrams(tokens, 2)
        trigrams = _ngrams(tokens, 3)

        feat = {
            # VADER outputs (free at the edge)
            'vader_score':      float(r.get('vader_score', 0.0)),
            'vader_pos':        float(r.get('vader_pos', 0.0)),
            'vader_neg':        float(r.get('vader_neg', 0.0)),
            'vader_neu':        float(r.get('vader_neu', 0.0)),
            'vader_confidence': float(r.get('vader_confidence', 0.0)),
            'vader_score_abs':  abs(float(r.get('vader_score', 0.0))),

            # Sentence structure
            'word_count':         len(tokens),
            'char_count':         len(text),
            'has_number':         int(bool(NUMBER_REGEX.search(text))),
            'has_currency':       int(bool(CURRENCY_REGEX.search(text))),
            'has_contrast_word':  _has_contrast(wordlist),
            'has_negation':       _has_negation(wordlist),

            # N-gram trigger aggregates
            'unigram_trigger_any':   int(bool(unigrams & n1)),
            'unigram_trigger_count': len(unigrams & n1),
            'bigram_trigger_any':    int(bool(bigrams & n2)),
            'bigram_trigger_count':  len(bigrams & n2),
            'trigram_trigger_any':   int(bool(trigrams & n3)),
            'trigram_trigger_count': len(trigrams & n3),
        }
        # Per-trigger binary features
        for ng, col in n1_cols.items():
            feat[col] = int(ng in unigrams)
        for ng, col in n2_cols.items():
            feat[col] = int(ng in bigrams)
        for ng, col in n3_cols.items():
            feat[col] = int(ng in trigrams)

        rows.append(feat)

    out = pd.DataFrame(rows, index=df.index)
    # Stash the column-name groups as attributes on the DataFrame for the trainer
    out.attrs['per_trigger_uni_cols'] = list(n1_cols.values())
    out.attrs['per_trigger_bi_cols']  = list(n2_cols.values())
    out.attrs['per_trigger_tri_cols'] = list(n3_cols.values())
    return out


# Convenience grouped feature lists used by the L4 ablation
FEATURE_GROUPS = {
    'vader_only': [
        'vader_score', 'vader_pos', 'vader_neg', 'vader_neu',
        'vader_confidence', 'vader_score_abs',
    ],
    'vader_struct': [
        'vader_score', 'vader_pos', 'vader_neg', 'vader_neu',
        'vader_confidence', 'vader_score_abs',
        'word_count', 'char_count', 'has_number', 'has_currency',
        'has_contrast_word', 'has_negation',
    ],
    'unigram': None,    # vader_struct + unigram_trigger_*
    'unibi':   None,    # + bigram_trigger_*
    'unibitri':None,    # + trigram_trigger_*
}


def feature_columns_for(group: str, X: pd.DataFrame | None = None) -> list[str]:
    """Return the list of feature column names for a given ablation group.

    If `X` is provided and contains per-trigger features (set via
    `extract_features`), they are appended for the relevant n-gram groups.
    """
    base = FEATURE_GROUPS['vader_struct']
    uni_per = (X.attrs.get('per_trigger_uni_cols', []) if X is not None else [])
    bi_per  = (X.attrs.get('per_trigger_bi_cols', [])  if X is not None else [])
    tri_per = (X.attrs.get('per_trigger_tri_cols', []) if X is not None else [])

    if group == 'vader_only':
        return FEATURE_GROUPS['vader_only']
    if group == 'vader_struct':
        return base
    if group == 'unigram':
        return base + ['unigram_trigger_any', 'unigram_trigger_count'] + uni_per
    if group == 'unibi':
        return (base + ['unigram_trigger_any', 'unigram_trigger_count',
                        'bigram_trigger_any', 'bigram_trigger_count']
                + uni_per + bi_per)
    if group == 'unibitri':
        return (base + ['unigram_trigger_any', 'unigram_trigger_count',
                        'bigram_trigger_any', 'bigram_trigger_count',
                        'trigram_trigger_any', 'trigram_trigger_count']
                + uni_per + bi_per + tri_per)
    raise ValueError(f"Unknown feature group: {group}")
