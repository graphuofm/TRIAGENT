"""L3 agent: local open-source LLM via HuggingFace transformers.

Default model is Qwen2.5-7B-Instruct (Apache 2.0, ungated, ~15GB bf16).
Runs comfortably on a single 24GB GPU (e.g. NVIDIA RTX A5000).

Cost is approximated from measured GPU latency × hourly rental rate
(same accounting style as FinBERT), since there is no per-token API price.
"""
import time
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from .base import SentimentAgent, AgentOutput
from ..config import DEVICE, LOCAL_LLM_MODEL, LOCAL_LLM_GPU_USD_PER_HR


PROMPT = """You are an expert financial analyst. Classify the sentiment of the following financial sentence as exactly one of: positive, negative, or neutral.

IMPORTANT FINANCIAL DOMAIN RULES:
- Consider financial impact, not surface-level word sentiment
- "Cut costs" / "reduced expenses" → positive (improves profitability)
- "Loss narrowed" / "loss decreased" → positive (improving)
- "Profit declined" / "earnings missed" → negative
- "Liability" / "debt" / "short" are often neutral accounting terms

Respond with ONLY a JSON object, no markdown, no extra text:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "reasoning": "brief"}}

Sentence: "{sentence}"
"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_json(raw: str) -> dict:
    """Robust JSON parsing for LLM output. Falls back to neutral on failure."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Find the first {...} block
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Give up
    raise ValueError(f"Could not parse JSON from: {raw[:200]!r}")


class LLMAgent(SentimentAgent):
    """Local HuggingFace causal-LM sentiment agent."""
    name = "llm"

    def __init__(
        self,
        model_name: str = LOCAL_LLM_MODEL,
        device: str = DEVICE,
        dtype: torch.dtype = torch.bfloat16,
        gpu_usd_per_hr: float = LOCAL_LLM_GPU_USD_PER_HR,
    ):
        self.model_name = model_name
        self.device = device
        self.gpu_usd_per_hr = gpu_usd_per_hr

        print(f"  Loading {model_name} on {device} ({dtype})...")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Causal LMs need left padding for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    def _build_prompt(self, sentence: str) -> str:
        messages = [{"role": "user", "content": PROMPT.format(sentence=sentence)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _cost_from_latency(self, latency_ms: float) -> float:
        return (latency_ms / 1000.0) * (self.gpu_usd_per_hr / 3600.0)

    def _parse_output(
        self,
        raw_text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> AgentOutput:
        try:
            parsed = _parse_json(raw_text)
            label = str(parsed.get('sentiment', 'neutral')).lower().strip()
            if label not in ('positive', 'negative', 'neutral'):
                label = 'neutral'
            score = float(parsed.get('score', 0.0))
            confidence = float(parsed.get('confidence', 0.8))
            reasoning = str(parsed.get('reasoning', ''))
        except Exception as e:
            label, score, confidence = 'neutral', 0.0, 0.0
            reasoning = f'PARSE_ERROR: {e} | raw={raw_text[:120]!r}'

        return AgentOutput(
            label=label,
            score=score,
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=self._cost_from_latency(latency_ms),
            extra={
                'reasoning': reasoning,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        )

    @torch.no_grad()
    def predict(self, text: str) -> AgentOutput:
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs['input_ids'].shape[1])

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        latency = (time.perf_counter() - t0) * 1000

        new_ids = outputs[0, input_tokens:]
        output_tokens = int(new_ids.shape[0])
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return self._parse_output(raw_text, input_tokens, output_tokens, latency)

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 8) -> list[AgentOutput]:
        results: list[AgentOutput] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f'LLM ({self.model_name.split("/")[-1]})'):
            batch = texts[i:i + batch_size]
            prompts = [self._build_prompt(t) for t in batch]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.model.device)
            input_lens = enc['attention_mask'].sum(dim=1).tolist()  # per-sample real input length
            padded_len = enc['input_ids'].shape[1]

            t0 = time.perf_counter()
            outputs = self.model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            batch_latency_ms = (time.perf_counter() - t0) * 1000
            per_sample_latency = batch_latency_ms / len(batch)

            for j in range(len(batch)):
                new_ids = outputs[j, padded_len:]
                # Strip trailing pad/eos tokens for clean decode + accurate token count
                eos_id = self.tokenizer.eos_token_id
                pad_id = self.tokenizer.pad_token_id
                clean_ids = []
                for tid in new_ids.tolist():
                    if tid == eos_id or tid == pad_id:
                        break
                    clean_ids.append(tid)
                output_tokens = len(clean_ids)
                raw_text = self.tokenizer.decode(clean_ids, skip_special_tokens=True)
                results.append(self._parse_output(
                    raw_text,
                    input_tokens=int(input_lens[j]),
                    output_tokens=output_tokens,
                    latency_ms=per_sample_latency,
                ))
        return results
