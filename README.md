# TriAgent

**TriAgent is a divergence-aware multi-agent routing framework that cuts large language model (LLM) inference cost by 10–100× by sending each query to the cheapest model tier that can answer it correctly.** It stratifies agents by *contextual granularity* — a word-level lexicon, a sentence-level domain transformer, and a cross-sentence LLM reasoner — and uses the disagreement between those tiers, the **Semantic Divergence Index (SDI)**, as a free routing signal, a free multilingual cache key, and a free hallucination detector.

[![arXiv](https://img.shields.io/badge/arXiv-2607.19794-b31b1b.svg)](https://arxiv.org/abs/2607.19794)
[![CIKM 2026](https://img.shields.io/badge/CIKM%202026-Short%20Paper-blue.svg)](https://doi.org/10.1145/3799682.3839978)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3839978-blue.svg)](https://doi.org/10.1145/3799682.3839978)
[![FinLLM@IJCAI 2026](https://img.shields.io/badge/FinLLM%40IJCAI%202026-Long%20Oral%20Award-gold.svg)](https://arxiv.org/abs/2607.19794)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> 🏆 **Long Oral Paper Award — FinLLM @ IJCAI 2026 (Bremen).** 1 of 5 long-oral papers.
> 📄 **Accepted at CIKM 2026** (Rome, 7–11 November 2026) — DOI [`10.1145/3799682.3839978`](https://doi.org/10.1145/3799682.3839978)
> 📚 **arXiv preprint:** [arXiv:2607.19794](https://arxiv.org/abs/2607.19794)

---

## TL;DR — the five numbers

| Finding | Result | Where |
|---|---|---|
| **Critic plateau** | A 1.5B model used as a *critic* over two cheap agents reaches **F1 = 0.87**, matching 3B and 7B (bootstrap 95% CIs overlap). A same-size 3-persona vote collapses to **F1 = 0.66**. | [§ Critic plateau](#1-the-critic-plateau-interaction-substitutes-for-parameters) |
| **Cost at scale** | At 10M users × 10 queries/day: always-GPT-4 costs **$365M/yr**, always-GPT-4o-mini **$11M/yr**, TriAgent **$1.65M/yr** — a **$363M/yr** and **$9.3M/yr** saving respectively. | [§ Cost](#3-cost-at-production-scale) |
| **Multilingual cache** | A Shared Consensus Dictionary on multilingual sentence-BERT answers **95% of Chinese queries from an English cache at F1 = 0.99**. | [§ SCD](#4-the-shared-consensus-dictionary-scd-a-cross-lingual-cache) |
| **Free hallucination detector** | The same SDI signal predicts "the LLM is wrong" at **AUC = 0.90**, with zero extra models, labels, or training. | [§ Trust signal](#5-sdi-as-a-free-hallucination-detector) |
| **Trading back-test** | SDI single-stage routing attains **Sharpe = 3.50** on a 20-ticker back-test, beating always-FinBERT (1.36) and always-LLM (0.11). | [§ Back-test](#6-20-ticker-back-test) |

---

## The problem TriAgent solves

Production LLM pipelines — classification, retrieval re-ranking, entity linking, intent inference, RAG re-reading — share a **structural cost trap**: the reasoner LLM dominates the per-query bill, yet most queries are trivially handled by models that cost 1/1000th as much. Teams either overpay for every query or hand-roll brittle routing logic with no principled signal to decide when the cheap path is safe.

TriAgent's answer: **run cheap agents that fail in different ways, and let their disagreement tell you when you need the expensive one.**

---

## How it works

### Three tiers, stratified by contextual granularity

| Tier | Reference instance | Context window | Latency | Cost / 1k queries |
|---|---|---|---|---|
| **L1** | VADER (word-level lexicon) | single tokens | ~0.03 ms | ~$0 |
| **L2** | FinBERT (sentence-level transformer, 110M params) | one sentence | ~1.5 ms | ~$0.0005 |
| **L3** | Qwen2.5 / Mistral-7B / Phi-3.5-mini (cross-sentence reasoner) | full context | 100s of ms | ~$0.11 |

The tiers are chosen so their **failure modes are orthogonal by construction** — each consumes a strictly larger context window than the previous, so the errors they can commit are structurally different. Measured pairwise Cohen's κ is 0.19–0.61; pairwise Jaccard error overlap is only 0.13–0.15.

### The Semantic Divergence Index (SDI)

Given continuous scores `s_V`, `s_F`, `s_L` ∈ [−1, +1] from the three tiers:

```
SDI_LE = |s_V − s_F|      (lexicon vs. encoder)
SDI_LR = |s_V − s_L|      (lexicon vs. reasoner)
SDI_ER = |s_F − s_L|      (encoder vs. reasoner)
```

Thresholding `(SDI_LE, SDI_ER)` partitions queries into four behaviour quadrants — *consensus*, *domain shift*, *ambiguous*, *mixed* — each with a distinct routing rule. Thresholds are set per task by grid search on a 20% validation split; the sweep is monotone, so any `(θ_LE, θ_ER)` inside `[0.2, 0.4] × [0.6, 0.8]` is Pareto-comparable within 0.4 pp.

### Routing

```
                    ┌──────────────────────────────┐
   query ─────────► │ SCD multilingual cache lookup│──hit──► cached label
                    └──────────────┬───────────────┘
                                   │ miss
                    ┌──────────────▼───────────────┐
                    │  L1 lexicon  +  L2 encoder   │
                    └──────────────┬───────────────┘
                                   │
                        SDI_LE ≤ θ │ SDI_LE > θ
                    ┌──────────────┴───────────────┐
                    │                              │
              return L1                   ┌────────▼────────┐
                                          │ SDI_ER > θ_ER ? │
                                          └────────┬────────┘
                                     no ───────────┴────────── yes
                                      │                        │
                                 return L2          L3 interaction protocol
                                                   (vote / critic / debate)
```

Only **1.5% of queries** reach the expensive L3 tier at the Balanced operating point.

---

## Results

### 1. The critic plateau: interaction substitutes for parameters

When the LLM is re-tasked as a **critic** over the two cheap agents' outputs — rather than answering from scratch — macro-F1 plateaus regardless of model size:

| Protocol | Qwen-1.5B | Qwen-3B | Qwen-7B | Mistral-7B |
|---|---|---|---|---|
| LLM alone | 0.69 | 0.62 | 0.81 | 0.70 |
| **critic** | **0.87** | **0.87** | 0.86 | 0.79 |
| debate | 0.69 | 0.81 | 0.87 | 0.82 |

**critic@1.5B costs $0.04 per 1k queries; critic@7B costs $0.11 — 3× the price for the same F1.**

**The negative control that identifies the mechanism.** Three *personas* of the same Qwen-1.5B (bull / bear / neutral) voting together reach only **F1 = 0.66**, *below* the 0.69 that a single Qwen-1.5B achieves alone. Multi-agent voting per se does not produce the plateau — **granularity-stratified diversity does.** This rules out a self-consistency reading of the result.

### 2. The plateau is calibration-conditional

The plateau is not universal, and we report where it fails:

| Testbed | FinBERT (L2) F1 | Expected Calibration Error | critic@1.5B | Verdict |
|---|---|---|---|---|
| **FPB** (curated news) | 0.88 | **0.016** (well calibrated) | **0.87** | Plateau holds |
| **TFNS** (informal tweets) | 0.66 | — | 0.66 | Plateau **fails** — below LLM-alone (0.76) |

**Deployable rule:** run a one-shot critic-vs-LLM-alone check on your task before committing. If the specialist is miscalibrated, route around the critic.

### 3. Cost at production scale

Annual inference cost, 10 queries per user per day, at public 2025 API prices:

| Strategy | 10K users | 1M users | 10M users |
|---|---|---|---|
| Always-GPT-4 | $365K | $36.5M | **$365M** |
| Always-GPT-4o-mini | $11K | $1.10M | **$11.0M** |
| Always-Qwen-7B (self-hosted) | $1.05K | $105K | $1.05M |
| Always-FinBERT | $18 | $1.8K | $18K |
| **TriAgent** | — | — | **$1.65M** |

At 10M users TriAgent saves **$363M/yr against always-GPT-4** and **$9.3M/yr against always-GPT-4o-mini**. These are scenario estimates: they assume the cache hit rate and specialist accuracy we measured carry over to the production workload, and that per-query API prices stay within an order of magnitude of 2025 published rates.

### 4. The Shared Consensus Dictionary (SCD): a cross-lingual cache

The SCD caches **committee decisions** — not single-model outputs — keyed by multilingual sentence-BERT embeddings. Because the encoder is multilingual, the cache doubles as a **cross-lingual canonicaliser**:

| Similarity threshold τ | Chinese-query hit rate | F1 on cached labels |
|---|---|---|
| 0.50 | 100% | 0.98 |
| **0.70** | **95%** | **0.99** |
| 0.85 | 55% | 1.00 |

**A new Chinese deployment inherits canonical answers from the existing English cache without running a Chinese committee at all.**

A related inversion: in Mandarin, `finbert-tone-chinese` reaches F1 = 0.72 while Qwen-7B reaches 0.80 — **the specialist/LLM ranking flips in low-resource languages**, and the router should escalate L→F instead of F→L.

### 5. SDI as a free hallucination detector

Stratifying by whether the LLM was right or wrong:

- Mean `SDI_ER` when the LLM is **correct**: 0.17
- Mean `SDI_ER` when the LLM is **wrong**: 0.71 (4× higher)
- **ROC AUC = 0.90** (Welch t = 46.8, p ≈ 0)

Any deployment already running a specialist and an LLM in parallel gets this trust score for free — no extra model call, no labels, no training. Unlike SelfCheckGPT or self-consistency, which re-sample the same model, SDI reuses an output you already computed.

**Honest limitation:** SDI is a strong *hallucination* detector but only a partial *adversarial* detector — AUC ≈ 0.71 on negation insertion and ≈ 0.5 on synonym swap, numeric flips, and typos. Do not treat it as a substitute for purpose-built adversarial defences.

### 6. 20-ticker back-test

| Strategy | Sharpe | Return % |
|---|---|---|
| **SDI-Single (S5)** | **3.50** | **3.4** |
| SDI-Two-Stage (S6) | 2.90 | 2.8 |
| debate@7B | 1.77 | 1.0 |
| Always-VADER (L1) | 1.51 | 2.0 |
| Always-FinBERT (L2) | 1.36 | 0.8 |
| Always-Qwen-7B (L3) | 0.11 | 0.3 |

The always-LLM policy is not merely expensive — on this back-test it is **the worst strategy on a risk-adjusted basis**.

### 7. Per-class breakdown

Macro-F1 hides which class drives the gain. On the *negative* class — the smallest (604/4,838) and the operationally costly one for risk management:

| Strategy | neg. F1 | neu. F1 | pos. F1 |
|---|---|---|---|
| **debate@7B** | **0.893** | 0.880 | 0.844 |
| FinBERT | 0.879 | 0.901 | 0.866 |
| critic@1.5B | 0.876 | 0.889 | 0.843 |
| Qwen-7B | 0.852 | 0.872 | 0.708 |

debate@7B is the **only** protocol that strictly beats the specialist on any class, and the lift is precision-side (+2.1 pp) at near-equal recall.

---

## How TriAgent compares to related work

| System | Routing signal | Needs training data? | Caches? | Cross-lingual? | Trust signal? |
|---|---|---|---|---|---|
| **TriAgent** | Inter-tier divergence (SDI), **unsupervised** | **No** | Committee decisions | **Yes** — one cache serves all languages | **Yes** — free, AUC 0.90 |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) | Learned cascade scorer | Yes — labelled | No | No | No |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | Learned preference model | Yes — preference data | No | No | No |
| [GPTCache](https://doi.org/10.18653/v1/2023.nlposs-1.24) | — (cache only) | No | Single-model outputs | No | No |
| [SelfCheckGPT](https://doi.org/10.18653/v1/2023.emnlp-main.557) | — (detector only) | No | No | No | Yes — but costs *n* extra LLM samples |
| [Multiagent debate](https://arxiv.org/abs/2305.14325) | — (accuracy only) | No | No | No | No |

**The distinguishing claim:** TriAgent derives routing, caching, and trust from a *single* divergence signal that is computed from outputs you already have, with no learned router and no labelled routing data.

---

## Quick start

```bash
git clone https://github.com/graphuofm/TRIAGENT
cd TRIAGENT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY only needed for the GPT-4o-mini sanity check
```

Tested on Python 3.10 + PyTorch CUDA 12.4, single NVIDIA RTX A5000 (24 GB).

## Reproducing every number in the paper

Each layer is a stand-alone script that consumes the previous layer's CSV.

```bash
# L1 — three-agent data collection on Financial PhraseBank
python experiments/L1_data_collection.py --dataset fpb --yes

# L1.5 — multi-size LLM sweep (Qwen 0.5B/1.5B/3B/7B/14B-4bit, Mistral-7B)
for SIZE in 0.5B 1.5B 3B 14B Mistral-7B; do
    python experiments/L1p5_size_sweep.py --sizes "$SIZE"
done

# L2 — SDI + four-quadrant decomposition
python experiments/L2_sdi_analysis.py

# L2.5 — interaction protocols (vote / critic / debate) across sizes
for SIZE in 1.5B 3B 7B Mistral-7B; do
    python experiments/L2p5_interaction.py --protocols critic --llm-size "$SIZE"
    python experiments/L2p5_interaction.py --protocols debate --llm-size "$SIZE"
done

# L3 — cost-Pareto routing sweep + operating points
python experiments/L3_routing_pareto.py && python experiments/L3p5_scaling.py

# L4 — three-granularity edge predictor (XGBoost, AUC = 0.85)
python experiments/L4_predictor.py --with-reasoning

# L5 — 20-ticker back-test
python experiments/L5_backtest.py

# L5.5 — hallucination detector + adversarial probe
python experiments/L5p5_e2_hallucination_detector.py
python experiments/L5p5_e3_bias_diversity_detail.py
python experiments/L5p5_e1_adversarial.py --generate
python experiments/L5p5_e1_adversarial.py --run-committee
python experiments/L5p5_e1_adversarial.py --analyse

# L7 — cross-lingual pilot (EN → ZH)
python experiments/L7_chinese_pilot.py --translate --n 1500
python experiments/L7_chinese_pilot.py --sweep --sizes 0.5B,1.5B,3B,7B
python experiments/L7p5_cross_lingual_committee.py

# L8 — Shared Consensus Dictionary build + threshold sweep
python experiments/L8_public_dictionary.py

# L9 — same-size persona vote (the negative control)
python experiments/L9_same_size_multiagent.py

# Regenerate every paper figure from results/data/*.csv
for s in paper/figures/code/make_*.py; do python "$s"; done
```

End-to-end wall-clock on a single A5000: **6–8 hours** (LLM inference dominates). Full spec in [`paper/PIPELINE_README.md`](paper/PIPELINE_README.md).

---

## Frequently asked questions

**Does TriAgent only work for financial sentiment analysis?**
The *architecture* is task-agnostic: the router applies to any pipeline where a cheap specialist and a reasoner LLM co-exist — dense-retrieval re-rankers, entity linkers, intent classifiers, RAG re-readers. The *empirical findings* (plateau height, quadrant accuracies, threshold shape) are validated on two financial testbeds and must be re-checked per task with the one-shot recipe above. We report this boundary explicitly rather than over-claiming.

**Do I have to use VADER and FinBERT?**
No. They are reference instances for financial sentiment, chosen because they are the canonical open baselines in their tier. Any (cheap, mid, expensive) triple works — a keyword lookup, your fine-tuned domain BERT, and a GPT-class reasoner, for example.

**Is this a learned router?**
No, and that is the point. SDI is computed directly from tier outputs. There is no router to train and no labelled routing data to collect, which is what FrugalGPT-style cascades and RouteLLM both require.

**How much cheaper is it, really?**
At the Balanced operating point, 70% of queries are answered by L1, 28.5% by L2, and only 1.5% reach L3. Against always-GPT-4 that is roughly a 200× cost reduction; against always-GPT-4o-mini roughly 7×. Absolute savings depend on your workload mix.

**Why does a 1.5B critic match a 7B critic?**
Because the critic is not solving the task from scratch — it is adjudicating between two structured opinions from agents that fail in different ways. That is an easier problem than classification, and small models can do it. The 3-persona negative control (F1 = 0.66) confirms the scaffolding, not the voting, is what matters.

**What is the catch?**
Three, and we state them in the paper: (1) the plateau *height* is family-specific — Mistral-7B tops out at 0.79, not 0.87; (2) the plateau *fails entirely* when the L2 specialist is miscalibrated, as on TFNS; (3) SDI is a weak adversarial detector (AUC ≈ 0.5 on most perturbation types) even though it is a strong hallucination detector.

More questions and answers: [`FAQ.md`](FAQ.md). Term definitions: [`GLOSSARY.md`](GLOSSARY.md).

---

## Repository layout

```
paper/                       LaTeX source for the workshop/arXiv version
  figures/code/              matplotlib scripts regenerating every figure
  PIPELINE_README.md         end-to-end pipeline spec
src/                         agents, SDI, routing, SCD, predictor, backtest
experiments/                 L1 … L9 reproducible per-layer scripts
data/                        raw datasets and mined trigger lexicons
results/                     figures, tables, summary CSVs
docs/                        layer-by-layer design specs
FAQ.md                       extended question-and-answer reference
GLOSSARY.md                  definitions of SDI, SCD, critic plateau, tiers
RESULTS.md                   every reported number in one place
COMPARISON.md                head-to-head against related systems
triagent_paper_compiled.pdf  compiled paper
triagent_overleaf.zip        Overleaf-ready LaTeX bundle
```

---

## Publications

This work has appeared in three venues. All three describe the same system; the titles differ because each venue framed a different contribution.

| Venue | Title | Status |
|---|---|---|
| **CIKM 2026** (Rome, Nov 7–11) | *TriAgent: Granularity-Stratified Multi-Agent Routing with a Multilingual Semantic Cache and a Free Trust Signal for LLM Inference Pipelines* | Accepted — short paper, DOI [10.1145/3799682.3839978](https://doi.org/10.1145/3799682.3839978) |
| **FinLLM @ IJCAI 2026** (Bremen) | *TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient and Privacy-Preserving Financial Sentiment Analysis* | 🏆 Long Oral Paper Award |
| **arXiv** | *TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis* | [arXiv:2607.19794](https://arxiv.org/abs/2607.19794), 22 July 2026 |

## Cite

```bibtex
@inproceedings{xu2026triagent,
  title     = {TriAgent: Granularity-Stratified Multi-Agent Routing with a
               Multilingual Semantic Cache and a Free Trust Signal for
               {LLM} Inference Pipelines},
  author    = {Xu, Isabel and Xu, Cynthia and Ren, Rachel and
               Guo, Cong and Ding, Jiacheng},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM},
  doi       = {10.1145/3799682.3839978},
}

@article{xu2026triagent_arxiv,
  title   = {TriAgent: Divergence-Aware Multi-Agent Committees for
             Cost-Efficient Financial Sentiment Analysis},
  author  = {Xu, Isabel and Xu, Cynthia and Ren, Rachel and
             Guo, Cong and Ding, Jiacheng},
  journal = {arXiv preprint arXiv:2607.19794},
  year    = {2026},
  eprint  = {2607.19794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2607.19794},
}
```

## Authors

Isabel Xu (The Overlake School) · Cynthia Xu (The Overlake School) · Rachel Ren (Edwards Vacuum Inc.) · Cong Guo (The University of Memphis) · **Jiacheng Ding** (The University of Memphis, corresponding — `jding2@memphis.edu`)

## License

[MIT](LICENSE). Dependencies retain their own licenses: VADER (MIT), FinBERT (Apache 2.0), Qwen2.5 (Apache 2.0), Mistral-7B (Apache 2.0), Phi-3.5-mini (MIT), Financial PhraseBank (CC-BY-NC-SA), TFNS (research use).
