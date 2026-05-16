"""Shared Consensus Dictionary (SCD) — embedding-cached committee decisions.

Architecture motivation
-----------------------

In the production deployment of an agentic financial system, many queries
are paraphrases of patterns the committee has already adjudicated. Running
the full V → F → L → critic/debate pipeline on each one is wasteful AND
fragile (when individual agents drift across versions, the committee output
drifts with them).

The Shared Consensus Dictionary (SCD) is an embedding-cached lookup over
canonical committee decisions:

    incoming sentence
        └─ encode (sentence-BERT)
        └─ k-NN search over cached entries (cos sim)
              ├── max sim ≥ τ  →  return cached label (no model call)
              └── max sim <  τ  →  run committee, write back to dictionary

This decouples committee throughput from model latency: hit-rate scales
with corpus repetition, and the same dictionary serves every member of the
committee — so a freshly added model gets cross-committee consistency for
free, mitigating the family-specific plateau gap we observe with
Mistral-7B.

Implementation notes
--------------------

* Embeddings: same multilingual sentence-BERT used in L4 — works for
  English (FPB, TFNS) and Chinese (L7) without swapping models.
* Storage: numpy arrays in memory; one .npy + one .json file on disk.
* Lookup: dot product over normalised embeddings (cos sim) — O(N) but
  trivially parallelised; for 100K cached entries this is microseconds
  on a single CPU core.
* No background re-ranking or eviction — kept deliberately simple.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DictHit:
    cached_label: str
    cached_score: float
    similarity:   float
    matched_sentence: str
    extra: dict = field(default_factory=dict)


class SharedConsensusDictionary:
    def __init__(
        self,
        embedder=None,
        threshold: float = 0.85,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    ):
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_name)
        else:
            self.embedder = embedder
        self.model_name = model_name
        self.threshold  = threshold
        self.embeddings: np.ndarray | None = None        # (N, D)
        self.entries:    list[dict] = []                 # parallel list of N entries

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = 'sentence',
        label_col: str = 'label_text',
        score_col: str | None = None,
        confidence_col: str | None = None,
        extra_cols: list[str] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Build SCD entries from a pandas DataFrame in one shot."""
        sentences = df[text_col].fillna('').astype(str).tolist()
        print(f"[SCD] Encoding {len(sentences)} entries on {self.embedder.device}...")
        self.embeddings = self.embedder.encode(
            sentences, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        self.entries = []
        for _, r in df.iterrows():
            entry = {
                'sentence': str(r[text_col]),
                'label':    r[label_col],
                'score':    float(r[score_col])      if score_col      else 0.0,
                'confidence': float(r[confidence_col]) if confidence_col else 1.0,
            }
            for c in (extra_cols or []):
                entry[c] = r[c] if c in r else None
            self.entries.append(entry)
        print(f"[SCD] Built dictionary with {len(self.entries)} entries, "
              f"embedding dim = {self.embeddings.shape[1]}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_one(self, sentence: str) -> DictHit | None:
        if self.embeddings is None or len(self.entries) == 0:
            return None
        qvec = self.embedder.encode(sentence, normalize_embeddings=True,
                                     convert_to_numpy=True)
        sims = self.embeddings @ qvec
        idx = int(sims.argmax())
        if sims[idx] < self.threshold:
            return None
        e = self.entries[idx]
        return DictHit(cached_label=e['label'], cached_score=e['score'],
                        similarity=float(sims[idx]), matched_sentence=e['sentence'],
                        extra={k: v for k, v in e.items()
                               if k not in ('sentence', 'label', 'score')})

    def query_batch(self, sentences: list[str], batch_size: int = 64
                    ) -> list[DictHit | None]:
        """Vectorised batch query — much faster than calling query_one in a loop."""
        if self.embeddings is None or len(self.entries) == 0:
            return [None] * len(sentences)
        qvecs = self.embedder.encode(sentences, batch_size=batch_size,
                                      show_progress_bar=False,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True)
        sims_matrix = qvecs @ self.embeddings.T          # (Q, N)
        best_idx = sims_matrix.argmax(axis=1)
        best_sims = sims_matrix[np.arange(len(sentences)), best_idx]

        results: list[DictHit | None] = []
        for i in range(len(sentences)):
            if best_sims[i] < self.threshold:
                results.append(None)
                continue
            e = self.entries[int(best_idx[i])]
            results.append(DictHit(
                cached_label=e['label'], cached_score=e['score'],
                similarity=float(best_sims[i]),
                matched_sentence=e['sentence'],
                extra={k: v for k, v in e.items()
                       if k not in ('sentence', 'label', 'score')},
            ))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / 'embeddings.npy', self.embeddings)
        with open(dir_path / 'entries.json', 'w') as f:
            json.dump({'model_name': self.model_name,
                       'threshold':  self.threshold,
                       'entries':    self.entries}, f, indent=2, default=str)
        print(f"[SCD] Saved to {dir_path} ({len(self.entries)} entries)")

    @classmethod
    def load(cls, dir_path: Path) -> 'SharedConsensusDictionary':
        dir_path = Path(dir_path)
        with open(dir_path / 'entries.json') as f:
            payload = json.load(f)
        scd = cls(model_name=payload['model_name'], threshold=payload['threshold'])
        scd.embeddings = np.load(dir_path / 'embeddings.npy')
        scd.entries = payload['entries']
        print(f"[SCD] Loaded {len(scd.entries)} entries from {dir_path}")
        return scd
