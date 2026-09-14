#!/usr/bin/env python3
"""Build the TriAgent Hugging Face dataset from results/data/*.csv.

Output: hf_dataset/ (gitignored), ready for scripts/upload_hf_dataset.sh.

Configs
  fpb              4,838 Financial PhraseBank sentences: every agent, every SDI,
                   every interaction protocol, and the persona-vote control.
  tfns             11,931 Twitter Financial News Sentiment tweets: agents, SDI,
                   and the two critic runs where the plateau fails.
  fpb_zh           1,500 FPB sentences machine-translated to Mandarin, with the
                   multi-size Qwen sweep and finbert-tone-chinese.
  fpb_adversarial  2,500 rows: 500 FPB sentences x {clean, synonym, negation,
                   numeric, chardrop}, with SDI under perturbation.
  scd_index        The Shared Consensus Dictionary: 384-d multilingual
                   sentence-BERT embeddings for the 4,838 FPB sentences.

Column names are rewritten from the internal CSV names (llm_*, llm_1p5b_*, ...)
to explicit model names (qwen7b_*, qwen1p5b_*, ...). Every agent and protocol
column is re-scored after renaming and must reproduce the paper's macro-F1, so
a mislabelled column fails the build instead of shipping.

FinChinaSentiment is deliberately excluded: its source declares no license, so
its text cannot be redistributed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "data"
OUT = ROOT / "hf_dataset"
CARD = ROOT / "scripts" / "hf_dataset_card.md"

AGENT_FIELDS = ["score", "label", "confidence", "latency_ms", "cost_usd",
                "reasoning", "input_tokens", "output_tokens"]
PROTO_FIELDS = ["label", "score", "confidence", "triggered",
                "extra_cost_usd", "extra_latency_ms", "rationale"]


def macro_f1(gold: pd.Series, pred: pd.Series) -> float:
    return f1_score(gold.astype(str).str.lower(), pred.astype(str).str.lower(), average="macro")


def check(name: str, got: float, want: float, tol: float = 0.0015) -> None:
    ok = abs(got - want) <= tol
    print(f"    {'ok ' if ok else 'BAD'} {name:34s} {got:.4f}  (paper {want:.4f})")
    if not ok:
        sys.exit(f"F1 check failed for {name}: got {got:.4f}, expected {want:.4f}")


def rename_agent(df: pd.DataFrame, old_prefix: str, new_prefix: str, suffix: str = "") -> pd.DataFrame:
    mapping = {f"{old_prefix}_{f}{suffix}": f"{new_prefix}_{f}"
               for f in AGENT_FIELDS if f"{old_prefix}_{f}{suffix}" in df.columns}
    return df.rename(columns=mapping)


def protocol_cols(path: Path, proto: str, new_prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    keep = {"sentence_id": "sentence_id"}
    keep.update({f"{proto}_{f}": f"{new_prefix}_{f}" for f in PROTO_FIELDS if f"{proto}_{f}" in d.columns})
    return d[list(keep)].rename(columns=keep)


# --------------------------------------------------------------------- fpb
def build_fpb() -> pd.DataFrame:
    print("\n[fpb]")
    df = pd.read_csv(SRC / "sdi_data.csv")
    df = rename_agent(df, "llm_0p5b", "qwen0p5b")
    df = rename_agent(df, "llm_1p5b", "qwen1p5b")
    df = rename_agent(df, "llm_3b", "qwen3b")
    df = rename_agent(df, "llm_14b", "qwen14b_4bit")
    df = rename_agent(df, "llm_mistral7b", "mistral7b")
    df = rename_agent(df, "llm", "qwen7b")  # canonical L3 = Qwen2.5-7B-Instruct; must run last

    protocols = [
        ("interaction_results_1p5b.csv",                "critic", "critic_qwen1p5b",   0.8707),
        ("interaction_results_3b.csv",                  "critic", "critic_qwen3b",     0.8702),
        ("interaction_results_critic_phi-3p5-mini.csv", "critic", "critic_phi3p5mini", 0.8567),
        ("interaction_results_critic_mistral-7b.csv",   "critic", "critic_mistral7b",  0.7898),
        ("interaction_results_debate_1p5b.csv",         "debate", "debate_qwen1p5b",   0.6858),
        ("interaction_results_debate_3b.csv",           "debate", "debate_qwen3b",     0.8108),
        ("interaction_results_7b.csv",                  "debate", "debate_qwen7b",     0.8722),
        ("interaction_results_debate_mistral-7b.csv",   "debate", "debate_mistral7b",  0.8215),
    ]
    for fname, proto, prefix, _ in protocols:
        p = protocol_cols(SRC / fname, proto, prefix)
        assert (p["sentence_id"].values == df["sentence_id"].values).all(), f"{fname} misaligned"
        df = df.merge(p, on="sentence_id", how="left", validate="one_to_one")

    persona = pd.read_csv(SRC / "same_size_multiagent.csv")
    assert (persona["gold"].str.lower().values == df["label_text"].str.lower().values).all(), "persona misaligned"
    persona = persona.drop(columns=["gold"]).rename(columns=lambda c: f"persona_qwen1p5b_{c.replace('_conf', '_confidence')}")
    df = pd.concat([df.reset_index(drop=True), persona.reset_index(drop=True)], axis=1)

    g = df["label_text"]
    for col, want in [("vader_label", .4894), ("finbert_label", .8828), ("qwen0p5b_label", .6273),
                      ("qwen1p5b_label", .6883), ("qwen3b_label", .6236), ("qwen7b_label", .8086),
                      ("qwen14b_4bit_label", .7910), ("mistral7b_label", .6980),
                      ("persona_qwen1p5b_vote_label", .6634)]:
        check(col, macro_f1(g, df[col]), want)
    for _, _, prefix, want in protocols:
        check(f"{prefix}_label", macro_f1(g, df[f"{prefix}_label"]), want)

    first = ["sentence_id", "sentence", "label", "label_text"]
    return df[first + [c for c in df.columns if c not in first]]


# --------------------------------------------------------------------- tfns
def build_tfns() -> pd.DataFrame:
    print("\n[tfns]")
    df = pd.read_csv(SRC / "sdi_data_tfns.csv")
    df = rename_agent(df, "llm_1p5b", "qwen1p5b")
    df = rename_agent(df, "llm_3b", "qwen3b")
    df = rename_agent(df, "llm", "qwen7b")
    for fname, prefix in [("interaction_results_critic_1p5b_tfns.csv", "critic_qwen1p5b"),
                          ("interaction_results_critic_7b_tfns.csv", "critic_qwen7b")]:
        p = protocol_cols(SRC / fname, "critic", prefix)
        assert (p["sentence_id"].values == df["sentence_id"].values).all(), f"{fname} misaligned"
        df = df.merge(p, on="sentence_id", how="left", validate="one_to_one")
    g = df["label_text"]
    for col, want in [("vader_label", .4400), ("finbert_label", .6633), ("qwen7b_label", .7571),
                      ("critic_qwen1p5b_label", .6595), ("critic_qwen7b_label", .6517)]:
        check(col, macro_f1(g, df[col]), want)
    first = ["sentence_id", "sentence", "label", "label_text"]
    return df[first + [c for c in df.columns if c not in first]]


# --------------------------------------------------------------------- fpb_zh
def build_fpb_zh(fpb: pd.DataFrame) -> pd.DataFrame:
    print("\n[fpb_zh]")
    df = pd.read_csv(SRC / "committee_data_fpb_zh_full.csv")
    ids = dict(zip(fpb["sentence"], fpb["sentence_id"]))
    df.insert(0, "sentence_id", df["sentence_en"].map(ids))
    assert df["sentence_id"].notna().all(), "unmapped Chinese rows"
    df = df.rename(columns={"sentence": "sentence_zh"})
    for size in ["0p5b", "1p5b", "3b", "7b"]:
        df = df.rename(columns={f"llm_{size}_{f}_zh": f"qwen{size}_{f}" for f in AGENT_FIELDS})
    df = df.rename(columns={c: c.replace("finbert_cn_", "finbert_tone_chinese_") for c in df.columns})
    df = df.rename(columns={c: c.replace("committee_zh_", "committee_qwen") for c in df.columns if c.startswith("committee_zh_")})
    g = df["label_text"]
    for col, want in [("qwen7b_label", .8005), ("qwen1p5b_label", .7189), ("finbert_tone_chinese_label", .7207)]:
        check(col, macro_f1(g, df[col]), want)
    first = ["sentence_id", "sentence_en", "sentence_zh", "label", "label_text"]
    return df[first + [c for c in df.columns if c not in first]]


# --------------------------------------------------------------------- fpb_adversarial
def build_adversarial(fpb: pd.DataFrame) -> pd.DataFrame:
    print("\n[fpb_adversarial]")
    df = pd.read_csv(SRC / "e1_adversarial_committee.csv")
    df = df.rename(columns={"gold_label": "label_text", "sentence": "sentence_perturbed",
                            "llm_score": "qwen7b_score", "llm_label": "qwen7b_label"})
    orig = fpb.set_index("sentence_id")["sentence"]
    df.insert(1, "sentence_original", df["sentence_id"].map(orig))
    assert df["sentence_original"].notna().all()
    print(f"    rows per perturbation: {df['perturbation'].value_counts().to_dict()}")
    first = ["sentence_id", "sentence_original", "perturbation", "sentence_perturbed", "label_text"]
    return df[first + [c for c in df.columns if c not in first]]


# --------------------------------------------------------------------- scd_index
def build_scd(fpb: pd.DataFrame) -> pd.DataFrame:
    print("\n[scd_index]")
    emb = np.load(SRC / "sbert_emb_fpb__sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.npy")
    assert emb.shape == (len(fpb), 384), emb.shape
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "embeddings not L2-normalised"
    print(f"    {emb.shape[0]} x {emb.shape[1]} float32, L2-normalised")
    return pd.DataFrame({
        "sentence_id": fpb["sentence_id"].values,
        "sentence": fpb["sentence"].values,
        "label_text": fpb["label_text"].values,
        "embedding": list(emb.astype(np.float32)),
    })


SUMMARIES = [
    "calibration_ece.csv", "f1_macro_bootstrap_ci.csv", "per_class_f1.csv", "e3_per_class.csv",
    "e3_bias_overlap.csv", "e2_hallucination.csv", "e1_auc.csv", "e1_delta_sdi.csv",
    "scd_threshold_sweep.csv", "scd_scaleup_16k.csv", "scd_cross_lingual.csv",
    "cross_lingual_consistency.csv", "pareto_points_7b.csv", "scaling_per_size.csv",
    "same_size_multiagent_summary.csv", "same_size_multiagent_consistency.csv",
    "predictor_results.csv", "predictor_results_tfns.csv",
    "backtest_results.csv", "backtest_summary_aggregate.csv",
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    (OUT / "summaries").mkdir()

    fpb = build_fpb()
    tables = {
        "fpb": fpb,
        "tfns": build_tfns(),
        "fpb_zh": build_fpb_zh(fpb),
        "fpb_adversarial": build_adversarial(fpb),
        "scd_index": build_scd(fpb),
    }

    print("\n[write]")
    for name, df in tables.items():
        path = OUT / "data" / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"    data/{name}.parquet  {len(df):6d} rows  {df.shape[1]:3d} cols  {path.stat().st_size/1e6:6.2f} MB")

    for f in SUMMARIES:
        shutil.copy(SRC / f, OUT / "summaries" / f)
    print(f"    summaries/  {len(SUMMARIES)} CSV tables")

    if CARD.exists():
        shutil.copy(CARD, OUT / "README.md")
        print("    README.md   copied from scripts/hf_dataset_card.md")
    else:
        print("    README.md   MISSING - write scripts/hf_dataset_card.md")

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\nbuilt {OUT.relative_to(ROOT)}/  total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
