"""L3 agent: GPT-4o-mini with full token tracking."""
import time
import json
from openai import OpenAI
from tqdm import tqdm
from .base import SentimentAgent, AgentOutput
from ..config import OPENAI_API_KEY, LLM_MODEL, LLM_PRICING


PROMPT = """You are an expert financial analyst. Classify the sentiment of the following financial sentence as exactly one of: positive, negative, or neutral.

IMPORTANT FINANCIAL DOMAIN RULES:
- Consider financial impact, not surface-level word sentiment
- "Cut costs" / "reduced expenses" → positive (improves profitability)
- "Loss narrowed" / "loss decreased" → positive (improving)
- "Profit declined" / "earnings missed" → negative
- "Liability" / "debt" / "short" are often neutral accounting terms

Respond with ONLY a JSON object, no markdown:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "reasoning": "brief"}}

Sentence: "{sentence}"
"""


class LLMAgent(SentimentAgent):
    name = "llm"

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL):
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Check .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.pricing = LLM_PRICING[model]

    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": PROMPT.format(sentence=text)}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            latency = (time.perf_counter() - t0) * 1000

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            label = parsed['sentiment'].lower()
            score = float(parsed.get('score', 0.0))
            confidence = float(parsed.get('confidence', 0.8))
            reasoning = parsed.get('reasoning', '')

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens * self.pricing['input'] / 1_000_000 +
                output_tokens * self.pricing['output'] / 1_000_000
            )

            return AgentOutput(
                label=label, score=score, confidence=confidence,
                latency_ms=latency, cost_usd=cost,
                extra={
                    'reasoning': reasoning,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            )
        except Exception as e:
            return AgentOutput(
                label='neutral', score=0.0, confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000, cost_usd=0.0,
                extra={'reasoning': f'ERROR: {e}', 'input_tokens': 0, 'output_tokens': 0}
            )

    def predict_batch(self, texts: list[str], delay: float = 0.05) -> list[AgentOutput]:
        results = []
        for t in tqdm(texts, desc=f'LLM ({self.model})'):
            results.append(self.predict(t))
            if delay > 0:
                time.sleep(delay)
        return results
