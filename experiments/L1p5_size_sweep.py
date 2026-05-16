"""L1.5: Multi-size LLM sweep.

For the agentic-pivot story we need to know how F1 (and cost) scales with
LLM parameter count. We re-run L3 across the Qwen2.5-Instruct family
{0.5B, 1.5B, 3B, 7B, 14B-4bit} and append the predictions back into the
canonical committee_data.csv.

VADER and FinBERT outputs are reused as-is from L1's CSV — no re-runs.

Usage:
    python experiments/L1p5_size_sweep.py --sample 100   # quick smoke test
    python experiments/L1p5_size_sweep.py                # full sweep
    python experiments/L1p5_size_sweep.py --sizes 0.5B,1.5B,3B  # subset

Output:
    results/data/committee_data.csv updated in-place with extra columns:
        llm_<SIZE>_score, llm_<SIZE>_label, llm_<SIZE>_confidence,
        llm_<SIZE>_latency_ms, llm_<SIZE>_cost_usd, llm_<SIZE>_reasoning,
        llm_<SIZE>_input_tokens, llm_<SIZE>_output_tokens
    where <SIZE> ∈ {'0p5b', '1p5b', '3b', '7b', '14b'}.
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR
from src.agents.llm_agent import LLMAgent


# Maps a friendly-key to (HF model id, suffix used in column names).
# 14B is loaded in 4-bit via bitsandbytes since bf16 won't fit on a 24GB card.
SIZE_REGISTRY: dict[str, dict] = {
    '0.5B':       {'hf': 'Qwen/Qwen2.5-0.5B-Instruct',         'suffix': '0p5b',     'load_4bit': False},
    '1.5B':       {'hf': 'Qwen/Qwen2.5-1.5B-Instruct',         'suffix': '1p5b',     'load_4bit': False},
    '3B':         {'hf': 'Qwen/Qwen2.5-3B-Instruct',           'suffix': '3b',       'load_4bit': False},
    '7B':         {'hf': 'Qwen/Qwen2.5-7B-Instruct',           'suffix': '7b',       'load_4bit': False},
    '14B':        {'hf': 'Qwen/Qwen2.5-14B-Instruct',          'suffix': '14b',      'load_4bit': True},
    # Cross-family validation (different vendor, same ~7B class)
    'Mistral-7B': {'hf': 'mistralai/Mistral-7B-Instruct-v0.3', 'suffix': 'mistral7b', 'load_4bit': False},
}


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_agent(hf_id: str, load_4bit: bool) -> LLMAgent:
    """Construct LLMAgent, optionally with 4-bit quantization for the 14B."""
    if not load_4bit:
        return LLMAgent(model_name=hf_id)

    # 4-bit path: monkey-patch the model load to go through bitsandbytes.
    # We avoid touching llm_agent.py by subclassing inline.
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    agent = LLMAgent.__new__(LLMAgent)
    agent.model_name = hf_id
    agent.device = 'cuda'
    from src.config import LOCAL_LLM_GPU_USD_PER_HR
    agent.gpu_usd_per_hr = LOCAL_LLM_GPU_USD_PER_HR
    print(f"  Loading {hf_id} in 4-bit (nf4)...")
    agent.tokenizer = AutoTokenizer.from_pretrained(hf_id)
    agent.tokenizer.padding_side = 'left'
    if agent.tokenizer.pad_token_id is None:
        agent.tokenizer.pad_token = agent.tokenizer.eos_token
    agent.model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map='auto',
    )
    agent.model.eval()
    return agent


def main(args):
    csv_path = RESULTS_DATA_DIR / "committee_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run L1 first to produce {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"⚠  Smoke test on {len(df)} samples")

    sizes = [s.strip() for s in args.sizes.split(',')] if args.sizes else list(SIZE_REGISTRY.keys())
    print(f"Sweeping sizes: {sizes}\n")

    for size_key in sizes:
        if size_key not in SIZE_REGISTRY:
            print(f"  SKIP unknown size {size_key!r}")
            continue
        cfg = SIZE_REGISTRY[size_key]
        suffix = cfg['suffix']

        # Skip if we already ran this size and --resume is set
        col = f'llm_{suffix}_label'
        if args.resume and col in df.columns and df[col].notna().all():
            print(f"=== {size_key}: already in CSV, skipping (--resume) ===")
            continue

        print(f"\n=== {size_key} ({cfg['hf']}) ===")
        agent = _build_agent(cfg['hf'], cfg['load_4bit'])
        results = agent.predict_batch(df['sentence'].tolist(), batch_size=args.batch_size)

        df[f'llm_{suffix}_score']         = [r.score for r in results]
        df[f'llm_{suffix}_label']         = [r.label for r in results]
        df[f'llm_{suffix}_confidence']    = [r.confidence for r in results]
        df[f'llm_{suffix}_latency_ms']    = [r.latency_ms for r in results]
        df[f'llm_{suffix}_cost_usd']      = [r.cost_usd for r in results]
        df[f'llm_{suffix}_reasoning']     = [r.extra['reasoning'] for r in results]
        df[f'llm_{suffix}_input_tokens']  = [r.extra['input_tokens'] for r in results]
        df[f'llm_{suffix}_output_tokens'] = [r.extra['output_tokens'] for r in results]

        acc = accuracy_score(df['label_text'], df[f'llm_{suffix}_label'])
        f1 = f1_score(df['label_text'], df[f'llm_{suffix}_label'], average='macro')
        avg_lat = df[f'llm_{suffix}_latency_ms'].mean()
        total_cost = df[f'llm_{suffix}_cost_usd'].sum()
        n_parse_err = df[f'llm_{suffix}_reasoning'].astype(str).str.startswith('PARSE_ERROR').sum()
        print(f"  → Acc={acc:.4f}  F1={f1:.4f}  Latency={avg_lat:.1f}ms  "
              f"TotalCost=${total_cost:.4f}  ParseErrors={n_parse_err}/{len(df)}")

        # Persist after every size — long sweeps shouldn't lose work on crash
        if not args.sample:
            df.to_csv(csv_path, index=False)
            print(f"  saved → {csv_path}")

        # Free model before loading next
        del agent
        _free_gpu()

    if args.sample:
        # Smoke runs: dump to a separate file so we don't pollute the canonical CSV
        out = RESULTS_DATA_DIR / f"committee_data_smoke_n{len(df)}.csv"
        df.to_csv(out, index=False)
        print(f"\nSmoke output → {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='If set, run on N random samples and write to a smoke CSV.')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated subset, e.g. "0.5B,1.5B,3B".')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true',
                        help='Skip a size if its column is already populated.')
    args = parser.parse_args()
    main(args)
