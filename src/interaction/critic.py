"""Critic protocol — LLM-as-judge.

When VADER and FinBERT disagree (or, more precisely, when SDI exceeds a
threshold), we ask the LLM to look at *both* their predictions plus the
original sentence and produce a final judgment. This is one extra LLM call
per triggered sample.

This is NOT a simple cascade — the LLM gets to see what the smaller agents
think, which is qualitatively different from running the LLM in isolation.
"""
from __future__ import annotations

import time

import pandas as pd
import torch
from tqdm import tqdm

from ..agents.llm_agent import LLMAgent, _parse_json
from .base import InteractionResult


CRITIC_PROMPT = """You are a senior financial analyst acting as a judge. Two junior models have independently classified the sentiment of the financial sentence below. Use their inputs as evidence — but make your own final decision.

Sentence: "{sentence}"

Junior model A (lexicon-based):
  prediction: {vader_label}
  continuous score (-1 = neg, +1 = pos): {vader_score:+.2f}
  self-reported confidence: {vader_conf:.2f}

Junior model B (financial transformer):
  prediction: {finbert_label}
  continuous score: {finbert_score:+.2f}
  self-reported confidence: {finbert_conf:.2f}

DOMAIN RULES (same as before):
- "Loss narrowed" / "loss decreased" -> positive
- "Profit declined" / "earnings missed" -> negative
- "Liability" / "debt" / "short" are often neutral accounting terms
- Disagreement between A and B is informative — one of them is likely wrong

Respond with ONLY a JSON object, no markdown:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "agreed_with": "A|B|neither", "reasoning": "brief"}}
"""


class CriticAgent:
    """Wraps an LLMAgent with the critic prompt."""

    def __init__(self, llm: LLMAgent):
        self.llm = llm

    def _build_prompt(self, row: pd.Series) -> str:
        return CRITIC_PROMPT.format(
            sentence=row['sentence'],
            vader_label=row['vader_label'],
            vader_score=row['vader_score'],
            vader_conf=row['vader_confidence'],
            finbert_label=row['finbert_label'],
            finbert_score=row['finbert_score'],
            finbert_conf=row['finbert_confidence'],
        )

    @torch.no_grad()
    def critique_batch(self, rows: list[pd.Series], batch_size: int = 8
                       ) -> list[InteractionResult]:
        """Batched LLM-as-judge over the given rows. Same per-row output as
        critique_row but ~6-8x faster on a single GPU."""
        tokenizer = self.llm.tokenizer
        model = self.llm.model
        results: list[InteractionResult] = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = []
            for r in chunk:
                chat = [{"role": "user", "content": self._build_prompt(r)}]
                prompts.append(tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True))
            enc = tokenizer(
                prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024
            ).to(model.device)
            in_lens = enc['attention_mask'].sum(dim=1).tolist()
            padded_len = enc['input_ids'].shape[1]

            t0 = time.perf_counter()
            outputs = model.generate(
                **enc, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            batch_latency_ms = (time.perf_counter() - t0) * 1000
            per_sample_ms = batch_latency_ms / len(chunk)

            for j in range(len(chunk)):
                new_ids = outputs[j, padded_len:]
                eos_id = tokenizer.eos_token_id
                pad_id = tokenizer.pad_token_id
                clean = []
                for tid in new_ids.tolist():
                    if tid == eos_id or tid == pad_id:
                        break
                    clean.append(tid)
                raw = tokenizer.decode(clean, skip_special_tokens=True)
                try:
                    parsed = _parse_json(raw)
                    label = str(parsed.get('sentiment', 'neutral')).lower()
                    if label not in ('positive', 'negative', 'neutral'):
                        label = 'neutral'
                    score = float(parsed.get('score', 0.0))
                    conf = float(parsed.get('confidence', 0.7))
                    reasoning = str(parsed.get('reasoning', ''))
                except Exception as e:
                    label, score, conf = 'neutral', 0.0, 0.0
                    reasoning = f'PARSE_ERROR: {e}'
                cost = (per_sample_ms / 1000.0) * (self.llm.gpu_usd_per_hr / 3600.0)
                results.append(InteractionResult(
                    label=label, score=score, confidence=conf,
                    extra_inferences=1, extra_cost_usd=cost,
                    extra_latency_ms=per_sample_ms,
                    rationale=reasoning, triggered=True,
                    extra={'in_tok': int(in_lens[j]), 'out_tok': len(clean)},
                ))
        return results


def apply_critic(
    df: pd.DataFrame,
    critic: CriticAgent,
    *,
    trigger_col: str = 'sdi_le',
    trigger_threshold: float = 0.3,
    fallback_label_col: str = 'finbert_label',
    fallback_score_col: str = 'finbert_score',
    fallback_conf_col: str = 'finbert_confidence',
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run the critic only on rows where `trigger_col` exceeds threshold;
    else use the fallback agent. SDI-gated, batched LLM inference."""
    out = df.copy()
    triggered_mask = out[trigger_col] > trigger_threshold
    n_trig = int(triggered_mask.sum())
    print(f"  Critic gate {trigger_col}>{trigger_threshold} triggers on "
          f"{n_trig}/{len(out)} rows ({100*n_trig/len(out):.1f}%)")

    triggered_rows = [out.iloc[i] for i in range(len(out)) if triggered_mask.iloc[i]]
    print(f"  Running batched critic (batch_size={batch_size}) over {len(triggered_rows)} rows...")
    triggered_results: list[InteractionResult] = []
    for i in tqdm(range(0, len(triggered_rows), batch_size), desc='critic'):
        chunk = triggered_rows[i:i + batch_size]
        triggered_results.extend(critic.critique_batch(chunk, batch_size=batch_size))

    # Stitch results back into row order
    labels, scores, confs, trig, costs, lats, rats = [], [], [], [], [], [], []
    t_iter = iter(triggered_results)
    for i in range(len(out)):
        if triggered_mask.iloc[i]:
            r = next(t_iter)
            labels.append(r.label); scores.append(r.score); confs.append(r.confidence)
            trig.append(True); costs.append(r.extra_cost_usd); lats.append(r.extra_latency_ms)
            rats.append(r.rationale)
        else:
            labels.append(out.iloc[i][fallback_label_col])
            scores.append(out.iloc[i][fallback_score_col])
            confs.append(out.iloc[i][fallback_conf_col])
            trig.append(False); costs.append(0.0); lats.append(0.0); rats.append('')

    out['critic_label']             = labels
    out['critic_score']             = scores
    out['critic_confidence']        = confs
    out['critic_triggered']         = trig
    out['critic_extra_cost_usd']    = costs
    out['critic_extra_latency_ms']  = lats
    out['critic_rationale']         = rats
    return out
