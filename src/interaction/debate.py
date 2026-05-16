"""Debate protocol — two-round structured exchange.

Round 1: Each agent gives label + 1-line rationale (we already have these
         from L1 / L1.5 — VADER's score is its rationale, FinBERT's softmax
         is its evidence, and the LLM's `llm_reasoning` column is its R1).
Round 2: We feed all three R1 outputs back to a *fresh* LLM call and ask
         it to either (a) endorse one peer or (b) give a tie-breaking new
         judgment. This second call is the "extra inference" we account for.

Costs one additional LLM call per triggered sample.
"""
from __future__ import annotations

import time

import pandas as pd
import torch
from tqdm import tqdm

from ..agents.llm_agent import LLMAgent, _parse_json
from .base import InteractionResult


DEBATE_R2_PROMPT = """You are a senior financial analyst chairing a debate. Three analysts have independently classified the sentiment of the financial sentence below. Reconcile their views and issue a final ruling.

Sentence: "{sentence}"

Analyst V (lexicon-based):
  prediction: {vader_label}  (score {vader_score:+.2f})

Analyst F (financial transformer):
  prediction: {finbert_label}  (score {finbert_score:+.2f}, confidence {finbert_conf:.2f})

Analyst L (reasoning LLM, your previous self):
  prediction: {llm_label}  (score {llm_score:+.2f}, confidence {llm_conf:.2f})
  rationale: "{llm_reasoning}"

DOMAIN RULES:
- "Loss narrowed" / "loss decreased" -> positive
- "Profit declined" / "earnings missed" -> negative
- "Liability" / "debt" / "short" are often neutral accounting terms
- F is usually right on neutrality; V is usually right on extreme polarity;
  L is usually right on context-dependent cases

Issue your final ruling. Respond with ONLY a JSON object:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "endorses": "V|F|L|none", "reasoning": "brief"}}
"""


class DebateAgent:
    """Wraps an LLMAgent for the round-2 debate adjudication."""

    def __init__(self, llm: LLMAgent):
        self.llm = llm

    def _build_prompt(
        self, row: pd.Series,
        llm_label_col: str, llm_score_col: str,
        llm_conf_col: str, llm_reasoning_col: str,
    ) -> str:
        return DEBATE_R2_PROMPT.format(
            sentence=row['sentence'],
            vader_label=row['vader_label'],
            vader_score=row['vader_score'],
            finbert_label=row['finbert_label'],
            finbert_score=row['finbert_score'],
            finbert_conf=row['finbert_confidence'],
            llm_label=row[llm_label_col],
            llm_score=row[llm_score_col],
            llm_conf=row[llm_conf_col],
            llm_reasoning=str(row[llm_reasoning_col])[:240],  # clamp long rationales
        )

    @torch.no_grad()
    def adjudicate_batch(self, rows: list[pd.Series], batch_size: int = 8,
                         **col_kwargs) -> list[InteractionResult]:
        """Batched round-2 adjudication."""
        tokenizer = self.llm.tokenizer
        model = self.llm.model
        results: list[InteractionResult] = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = []
            for r in chunk:
                chat = [{"role": "user", "content": self._build_prompt(r, **col_kwargs)}]
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


def apply_debate(
    df: pd.DataFrame,
    debater: DebateAgent,
    *,
    trigger_col: str = 'sdi_max',
    trigger_threshold: float = 0.5,
    llm_label_col: str = 'llm_label',
    llm_score_col: str = 'llm_score',
    llm_conf_col: str = 'llm_confidence',
    llm_reasoning_col: str = 'llm_reasoning',
    fallback_label_col: str = 'llm_label',
    fallback_score_col: str = 'llm_score',
    fallback_conf_col: str = 'llm_confidence',
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run debate only on rows where `trigger_col` > threshold; else fall back.
    Batched LLM inference for speed."""
    out = df.copy()
    triggered_mask = out[trigger_col] > trigger_threshold
    n_trig = int(triggered_mask.sum())
    print(f"  Debate gate {trigger_col}>{trigger_threshold} triggers on "
          f"{n_trig}/{len(out)} rows ({100*n_trig/len(out):.1f}%)")

    col_kwargs = dict(
        llm_label_col=llm_label_col,
        llm_score_col=llm_score_col,
        llm_conf_col=llm_conf_col,
        llm_reasoning_col=llm_reasoning_col,
    )

    triggered_rows = [out.iloc[i] for i in range(len(out)) if triggered_mask.iloc[i]]
    print(f"  Running batched debate (batch_size={batch_size}) over {len(triggered_rows)} rows...")
    triggered_results: list[InteractionResult] = []
    for i in tqdm(range(0, len(triggered_rows), batch_size), desc='debate'):
        chunk = triggered_rows[i:i + batch_size]
        triggered_results.extend(
            debater.adjudicate_batch(chunk, batch_size=batch_size, **col_kwargs))

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

    out['debate_label']            = labels
    out['debate_score']            = scores
    out['debate_confidence']       = confs
    out['debate_triggered']        = trig
    out['debate_extra_cost_usd']   = costs
    out['debate_extra_latency_ms'] = lats
    out['debate_rationale']        = rats
    return out
