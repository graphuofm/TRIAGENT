"""L7: Chinese cross-lingual pilot.

Two complementary Chinese datasets:
  (a) FinChinaSentiment — real Chinese financial news, binary (neg/neu),
      news-article length. Tests whether the agentic framework holds
      on long-form Chinese text.
  (b) Translated FPB — we translate ~N FPB sentences to Chinese via the
      LLM, keeping the gold labels. Tests whether the per-size scaling
      curve and the critic-plateau finding generalise across languages,
      isolating language from dataset shift.

VADER and FinBERT do not handle Chinese. For the Chinese pipeline we
therefore run only the LLM (multi-size sweep) and report Qwen-N alone +
the critic protocol where the L1/L2 stand-ins are placeholder zero-info
agents (i.e., the critic protocol degenerates to "LLM with extra context
about the original sentence" on Chinese — still useful as an ablation).

Usage:
    # Translate 300 FPB sentences to Chinese with Qwen-7B (~3 min)
    python experiments/L7_chinese_pilot.py --translate --n 300

    # Sweep Qwen-N over translated FPB (committee_data_fpb_zh.csv)
    python experiments/L7_chinese_pilot.py --sweep --sizes 1.5B,3B,7B
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, RAW_DIR, LABEL_TO_INT
from src.utils.data_loader import load_fpb
from src.agents.llm_agent import LLMAgent


TRANSLATE_PROMPT = (
    "Translate the following English financial sentence into natural, "
    "fluent Mandarin Chinese (Simplified). Preserve all financial terminology "
    "and numerical values exactly. Output ONLY the Chinese translation, no "
    "explanation, no quotes, no English.\n\n"
    "Sentence: {sentence}"
)

# Reuse the same multilingual sentiment classification prompt structure as
# the English LLMAgent, but issue it in Chinese to bias the LLM's mode.
ZH_CLASSIFY_PROMPT = """你是一名资深金融分析师。请将下面这句金融文本的情感倾向分类为以下三类之一：positive（正面）、negative（负面）、neutral（中性）。

重要的金融领域规则：
- 关注的是对财务表现的影响，而不是字面情绪
- "削减成本" / "降低费用" → positive（提高利润率）
- "亏损收窄" / "亏损减少" → positive（在改善）
- "利润下滑" / "未达盈利预期" → negative
- "负债" / "债务" / "做空" 等通常是中性的会计术语

只输出一个 JSON 对象，无其他内容：
{{"sentiment": "positive|negative|neutral", "score": -1.0 到 1.0 之间的小数, "confidence": 0.0 到 1.0 之间, "reasoning": "简短理由（中文）"}}

文本："{sentence}"
"""


@torch.no_grad()
def _generate_batch(llm: LLMAgent, prompts: list[str], max_new_tokens: int = 200,
                    batch_size: int = 8) -> list[tuple[str, int, int, float]]:
    """Generic batched generation. Returns (text, in_tok, out_tok, latency_ms) per row."""
    tok = llm.tokenizer
    model = llm.model
    out: list[tuple[str, int, int, float]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc='gen'):
        chunk = prompts[i:i + batch_size]
        rendered = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                    for p in chunk]
        enc = tok(rendered, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(model.device)
        in_lens = enc['attention_mask'].sum(dim=1).tolist()
        padded = enc['input_ids'].shape[1]
        t0 = time.perf_counter()
        outs = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id)
        per_lat = (time.perf_counter() - t0) * 1000 / len(chunk)
        for j in range(len(chunk)):
            new_ids = outs[j, padded:]
            eos = tok.eos_token_id
            pad = tok.pad_token_id
            clean = []
            for tid in new_ids.tolist():
                if tid == eos or tid == pad:
                    break
                clean.append(tid)
            out.append((tok.decode(clean, skip_special_tokens=True),
                        int(in_lens[j]), len(clean), per_lat))
    return out


def cmd_translate(args):
    """Take N random FPB sentences, translate via Qwen-N, persist with gold labels."""
    fpb = load_fpb()
    sample = fpb.sample(args.n, random_state=42).reset_index(drop=True)
    prompts = [TRANSLATE_PROMPT.format(sentence=s) for s in sample['sentence']]

    print(f"Translating {len(sample)} FPB sentences via {args.translator}...")
    llm = LLMAgent(model_name=args.translator)
    results = _generate_batch(llm, prompts, max_new_tokens=160, batch_size=args.batch_size)
    sample['sentence_zh'] = [r[0].strip().strip('"').strip('"').strip('"') for r in results]
    sample['translation_in_tok']  = [r[1] for r in results]
    sample['translation_out_tok'] = [r[2] for r in results]
    sample['translation_lat_ms']  = [r[3] for r in results]

    out = pd.DataFrame({
        'sentence_en': sample['sentence'],
        'sentence':    sample['sentence_zh'],   # canonical text col → Chinese
        'label_text':  sample['label_text'],
        'label':       sample['label'],
    })
    cache_path = RAW_DIR / "fpb_translated_zh.csv"
    out.to_csv(cache_path, index=False)
    out_results = RESULTS_DATA_DIR / "committee_data_fpb_zh.csv"
    out.to_csv(out_results, index=False)
    print(f"✓ saved {cache_path}")
    print(f"✓ saved {out_results}")
    print(f"  label dist: {out['label_text'].value_counts().to_dict()}")
    print("\nFirst 3 translations:")
    for i in range(min(3, len(out))):
        print(f"  EN: {sample.iloc[i]['sentence'][:90]}...")
        print(f"  ZH: {out.iloc[i]['sentence'][:90]}...")
        print(f"  gold: {out.iloc[i]['label_text']}")


def cmd_sweep(args):
    """Sweep Qwen-N as a Chinese sentiment classifier on translated FPB."""
    src = RESULTS_DATA_DIR / "committee_data_fpb_zh.csv"
    if not src.exists():
        raise FileNotFoundError("Run --translate first to produce committee_data_fpb_zh.csv")
    df = pd.read_csv(src)
    print(f"Loaded {len(df)} translated FPB samples from {src}")

    SIZE_TO_HF = {
        '0.5B': 'Qwen/Qwen2.5-0.5B-Instruct',
        '1.5B': 'Qwen/Qwen2.5-1.5B-Instruct',
        '3B':   'Qwen/Qwen2.5-3B-Instruct',
        '7B':   'Qwen/Qwen2.5-7B-Instruct',
        '14B':  'Qwen/Qwen2.5-14B-Instruct',
    }
    SIZE_TO_SUFFIX = {'0.5B': '0p5b', '1.5B': '1p5b', '3B': '3b',
                       '7B': '7b', '14B': '14b'}
    sizes = [s.strip() for s in args.sizes.split(',')]

    import json, re
    JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

    def parse(raw: str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw); raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except Exception:
            m = JSON_RE.search(raw)
            if m:
                try: return json.loads(m.group(0))
                except Exception: pass
            raise ValueError(f"unparseable: {raw[:120]!r}")

    for key in sizes:
        hf = SIZE_TO_HF[key]; suffix = SIZE_TO_SUFFIX[key]
        col_label = f'llm_{suffix}_label_zh'
        if args.resume and col_label in df.columns:
            print(f"  {key}: column already populated, skipping")
            continue
        print(f"\n=== {key} ({hf}) ===")
        llm = LLMAgent(model_name=hf)
        prompts = [ZH_CLASSIFY_PROMPT.format(sentence=s) for s in df['sentence']]
        results = _generate_batch(llm, prompts, max_new_tokens=200, batch_size=args.batch_size)
        labels, scores, confs, lats, costs, parses = [], [], [], [], [], 0
        for r in results:
            text, in_tok, out_tok, lat = r
            try:
                p = parse(text)
                lbl = str(p.get('sentiment', 'neutral')).lower()
                if lbl not in ('positive', 'negative', 'neutral'): lbl = 'neutral'
                sc = float(p.get('score', 0.0)); cf = float(p.get('confidence', 0.7))
            except Exception:
                lbl, sc, cf = 'neutral', 0.0, 0.0; parses += 1
            labels.append(lbl); scores.append(sc); confs.append(cf); lats.append(lat)
            costs.append((lat / 1000.0) * (llm.gpu_usd_per_hr / 3600.0))
        df[f'llm_{suffix}_label_zh']      = labels
        df[f'llm_{suffix}_score_zh']      = scores
        df[f'llm_{suffix}_confidence_zh'] = confs
        df[f'llm_{suffix}_latency_ms_zh'] = lats
        df[f'llm_{suffix}_cost_usd_zh']   = costs
        acc = accuracy_score(df['label_text'], labels)
        f1 = f1_score(df['label_text'], labels, average='macro')
        print(f"  → Acc={acc:.4f}  F1={f1:.4f}  ParseErrors={parses}/{len(df)}  "
              f"AvgLat={sum(lats)/len(lats):.1f}ms  TotalCost=${sum(costs):.4f}")

        df.to_csv(src, index=False)
        del llm
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate', action='store_true',
                        help='Translate N FPB sentences via the chosen LLM')
    parser.add_argument('--sweep', action='store_true',
                        help='Run multi-size Qwen on the translated set')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--translator', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--sizes', default='1.5B,3B,7B')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not (args.translate or args.sweep):
        parser.error("pass --translate and/or --sweep")
    if args.translate:
        cmd_translate(args)
    if args.sweep:
        cmd_sweep(args)
