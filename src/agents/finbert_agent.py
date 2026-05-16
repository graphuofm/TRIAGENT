"""L2 agent: FinBERT domain-specific sentiment."""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from .base import SentimentAgent, AgentOutput
from ..config import DEVICE


class FinBERTAgent(SentimentAgent):
    name = "finbert"
    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP = {0: 'positive', 1: 'negative', 2: 'neutral'}

    GPU_RENTAL_USD_PER_HR = 0.50
    AVG_LATENCY_MS = 3.6
    COST_PER_INFERENCE = (GPU_RENTAL_USD_PER_HR / 3600) * (AVG_LATENCY_MS / 1000)

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=512, padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        latency = (time.perf_counter() - t0) * 1000

        pred_idx = probs.argmax().item()
        label = self.LABEL_MAP[pred_idx]
        score = float(probs[0] - probs[1])  # P(pos) - P(neg)

        return AgentOutput(
            label=label,
            score=score,
            confidence=float(probs[pred_idx]),
            latency_ms=latency,
            cost_usd=self.COST_PER_INFERENCE,
            extra={
                'prob_pos': float(probs[0]),
                'prob_neg': float(probs[1]),
                'prob_neu': float(probs[2]),
            }
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[AgentOutput]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc='FinBERT'):
            batch = texts[i:i+batch_size]
            t0 = time.perf_counter()
            inputs = self.tokenizer(
                batch, return_tensors='pt', truncation=True,
                max_length=512, padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_latency = (time.perf_counter() - t0) * 1000
            per_sample_latency = batch_latency / len(batch)

            for j in range(len(batch)):
                p = probs[j]
                pred_idx = p.argmax().item()
                results.append(AgentOutput(
                    label=self.LABEL_MAP[pred_idx],
                    score=float(p[0] - p[1]),
                    confidence=float(p[pred_idx]),
                    latency_ms=per_sample_latency,
                    cost_usd=self.COST_PER_INFERENCE,
                    extra={
                        'prob_pos': float(p[0]),
                        'prob_neg': float(p[1]),
                        'prob_neu': float(p[2]),
                    }
                ))
        return results
