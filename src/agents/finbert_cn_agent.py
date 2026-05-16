"""Chinese FinBERT agent: yiyanghkust/finbert-tone-chinese.

Same interface as the English FinBERTAgent but uses the Chinese-specific
checkpoint. Used in the L7 cross-lingual experiments to provide a real
Chinese specialist tier, replacing the English FinBERT which scored
F1≈0.06 on Chinese (i.e. random).
"""
from __future__ import annotations

import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .base import SentimentAgent, AgentOutput
from ..config import DEVICE


class FinBERTCNAgent(SentimentAgent):
    name = "finbert_cn"
    MODEL_NAME = "yiyanghkust/finbert-tone-chinese"
    # Inferred from model.config.id2label: {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    LABEL_MAP = {0: 'neutral', 1: 'positive', 2: 'negative'}

    GPU_RENTAL_USD_PER_HR = 0.50
    AVG_LATENCY_MS = 4.0  # similar to English FinBERT class
    COST_PER_INFERENCE = (GPU_RENTAL_USD_PER_HR / 3600) * (AVG_LATENCY_MS / 1000)

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=256, padding=True).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        latency = (time.perf_counter() - t0) * 1000

        pred_idx = probs.argmax().item()
        label = self.LABEL_MAP[pred_idx]
        # Continuous score: P(positive) - P(negative)
        score = float(probs[1] - probs[2])
        return AgentOutput(
            label=label, score=score, confidence=float(probs[pred_idx]),
            latency_ms=latency, cost_usd=self.COST_PER_INFERENCE,
            extra={'prob_neutral': float(probs[0]),
                   'prob_positive': float(probs[1]),
                   'prob_negative': float(probs[2])}
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[AgentOutput]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc='FinBERT-CN'):
            batch = texts[i:i + batch_size]
            t0 = time.perf_counter()
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                    max_length=256, padding=True).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_lat = (time.perf_counter() - t0) * 1000
            per_lat = batch_lat / len(batch)
            for j in range(len(batch)):
                p = probs[j]
                idx = p.argmax().item()
                results.append(AgentOutput(
                    label=self.LABEL_MAP[idx],
                    score=float(p[1] - p[2]),
                    confidence=float(p[idx]),
                    latency_ms=per_lat,
                    cost_usd=self.COST_PER_INFERENCE,
                    extra={'prob_neutral': float(p[0]),
                           'prob_positive': float(p[1]),
                           'prob_negative': float(p[2])},
                ))
        return results
