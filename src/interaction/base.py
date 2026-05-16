"""Common types for interaction protocols.

A protocol takes a row from sdi_data.csv (which already contains every
agent's prediction, score, confidence) and produces a final committee
decision plus an optional list of additional inference calls that were
needed (used for cost accounting).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InteractionResult:
    label: str                       # final committee decision
    score: float                     # final continuous sentiment in [-1, 1]
    confidence: float                # final committee confidence
    extra_inferences: int = 0        # how many *new* LLM calls this row triggered
    extra_cost_usd: float = 0.0      # sum of $cost across new LLM calls
    extra_latency_ms: float = 0.0    # sum of latency across new LLM calls
    rationale: str = ''              # protocol-specific debug string
    triggered: bool = False          # whether the SDI gate fired
    extra: dict = field(default_factory=dict)
