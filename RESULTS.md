# TriAgent — Complete Results Reference

Every quantitative result reported in [TriAgent (CIKM 2026)](https://doi.org/10.1145/3799682.3839978) and [arXiv:2607.19794](https://arxiv.org/abs/2607.19794), with the source file that reproduces it. All numbers were verified against the raw CSVs in `results/data/`.

**Primary testbed:** Financial PhraseBank (`sentences_allagree`) — 4,838 sentences, 604 negative / 2,872 neutral / 1,362 positive.
**Secondary testbed:** Twitter Financial News Sentiment (TFNS) — 11,931 tweets, 1,789 negative / 7,744 neutral / 2,398 positive.
**Hardware:** single NVIDIA RTX A5000 (24 GB), bf16 inference, 4-bit via bitsandbytes for 14B.
**Cost basis:** GPU rental amortised at $0.40/hour A5000-class; API prices from OpenAI public list, 2025.

---

## 1. Single-agent baselines (FPB, macro-F1)

| Agent | Macro-F1 | Bootstrap 95% CI | Notes |
|---|---|---|---|
| VADER (L1) | 0.4894 | [0.4746, 0.5067] | matches published reference to 0.001 |
| **FinBERT (L2)** | **0.8828** | **[0.8726, 0.8926]** | strongest single agent |
| Qwen2.5-0.5B | 0.6273 | [0.6111, 0.6432] | |
| Qwen2.5-1.5B | 0.6883 | [0.6708, 0.7033] | |
| Qwen2.5-3B | 0.6236 | [0.6051, 0.6393] | **anomalously below 1.5B** — over-predicts neutral |
| Qwen2.5-7B | 0.8086 | [0.7959, 0.8195] | |
| Qwen2.5-14B-4bit | 0.79 | — | **below 7B bf16** — quantisation wipes out the size gain |
| Mistral-7B | 0.70 | — | cross-family check |

**Key statistical claim:** FinBERT's CI [0.8726, 0.8926] and Qwen-7B's CI [0.7959, 0.8195] **do not overlap**. The domain specialist is statistically better than the general-purpose LLM, at roughly 1/200th the cost.

*Reproduce:* `experiments/L1_data_collection.py`, `experiments/L1p5_size_sweep.py` → `results/data/committee_data.csv`, `f1_macro_bootstrap_ci.csv`

---

## 2. Calibration (Expected Calibration Error)

| Agent | ECE | Mean accuracy | Mean stated confidence | Reading |
|---|---|---|---|---|
| **FinBERT** | **0.0165** | 0.8898 | 0.8747 | nearly perfectly calibrated |
| Qwen-7B | 0.1154 | 0.8258 | 0.9411 | over-confident by 11.5 points |
| VADER | 0.3482 | 0.5440 | 0.2488 | uncalibrated |

This table is the mechanism behind the calibration-conditional plateau: FinBERT's near-zero ECE is what makes its output usable as critic scaffolding.

*Reproduce:* `results/data/calibration_ece.csv`

---

## 3. Interaction protocols across model sizes (FPB, macro-F1)

| Protocol | Qwen-1.5B | Qwen-3B | Qwen-7B | Mistral-7B | Phi-3.5-mini |
|---|---|---|---|---|---|
| LLM alone | 0.6883 | 0.6236 | 0.8086 | 0.70 | — |
| **critic** | **0.8707** | **0.8702** | 0.8602 | 0.79 | 0.86 |
| debate | 0.69 | 0.81 | **0.8722** | 0.82 | — |

**Bootstrap CIs for the plateau claim:**

| Strategy | Macro-F1 | 95% CI |
|---|---|---|
| critic@1.5B | 0.8707 | [0.8602, 0.8796] |
| critic@3B | 0.8702 | [0.8598, 0.8796] |
| critic@7B | 0.8602 | [0.8502, 0.8703] |
| debate@7B | 0.8722 | [0.8624, 0.8816] |

All four intervals overlap — this is what makes the plateau statistically a plateau rather than a trend.

**Cost per 1,000 queries:** critic@1.5B = $0.04, critic@7B = $0.11. **3× the price for the same F1.**

*Reproduce:* `experiments/L2p5_interaction.py` → `results/data/interaction_summary_*.csv`

---

## 4. The negative control (same-size persona vote)

Three personas of the same Qwen2.5-1.5B, differing only by prompt:

| Configuration | Macro-F1 |
|---|---|
| persona-bull | 0.74 |
| persona-bear | 0.68 |
| persona-neutral | 0.55 |
| **3-persona vote** | **0.66** |
| single Qwen-1.5B (baseline) | 0.69 |
| cross-tier critic@1.5B | **0.87** |

Inter-persona agreement: 81%.

**The vote scores *below* a single instance of the same model.** Multi-agent voting per se does not produce the plateau; granularity-stratified diversity does. This rules out a self-consistency reading.

*Reproduce:* `experiments/L9_same_size_multiagent.py` → `results/data/same_size_multiagent_summary.csv`

---

## 5. The plateau fails on TFNS (calibration-conditional)

| Agent / protocol | FPB | TFNS |
|---|---|---|
| VADER | 0.49 | 0.44 |
| FinBERT | 0.88 | **0.66** (collapses) |
| Qwen-7B alone | 0.81 | 0.76 |
| critic@1.5B | **0.87** | **0.66** |
| critic@7B | 0.86 | **0.65** |

On TFNS both critic settings fall **below** the LLM-alone baseline of 0.76. When the L2 specialist is miscalibrated on fragmented customer text, the critic scaffolding actively drags the LLM down.

*Reproduce:* `experiments/L1_data_collection.py --dataset tfns`, `experiments/L2p5_interaction.py` → `results/data/interaction_summary_critic_*_tfns.csv`

---

## 6. Bias diversity (the orthogonality premise)

**Pairwise Cohen's κ:** κ(VADER, FinBERT) = 0.27, κ(FinBERT, Qwen-7B) = 0.61, κ(VADER, Qwen-7B) = 0.19.

**Pairwise Jaccard error-set overlap:**

| | VADER | FinBERT | Qwen-7B |
|---|---|---|---|
| **VADER** | 1.000 | 0.132 | 0.139 |
| **FinBERT** | 0.132 | 1.000 | 0.146 |
| **Qwen-7B** | 0.139 | 0.146 | 1.000 |

The three tiers get different things wrong — overlap 0.13–0.15 — which is the premise that makes SDI informative.

*Reproduce:* `experiments/L5p5_e3_bias_diversity_detail.py` → `results/data/e3_bias_overlap.csv`

---

## 7. Four-quadrant decomposition

Quadrants are assigned from `(SDI_LE, SDI_ER)` with `SDI_LOW = 0.3` and `SDI_HIGH = 0.7`. The rules are checked in the order listed, so a row where both divergences are high counts as ambiguous. Accuracy is measured separately for each agent inside each quadrant.

| Quadrant | Rule (checked in order) | Share of FPB | Qwen-7B acc. | FinBERT acc. | VADER acc. | Qwen-7B mean confidence |
|---|---|---|---|---|---|---|
| **Ambiguous** | `SDI_ER > 0.7` | 15.9% | **28.1%** | **70.8%** | 55.6% | 0.93 |
| **Consensus** | `SDI_LE < 0.3` and `SDI_ER < 0.3` | 39.4% | 96.5% | 96.4% | 74.6% | 0.97 |
| **Domain shift** | `SDI_LE > 0.7` and `SDI_ER < 0.3` | 13.5% | 95.5% | 95.5% | 10.3% | 0.90 |
| **Mixed** | everything else | 31.2% | 87.2% | 86.0% | 47.2% | 0.93 |

In the ambiguous quadrant the specialist's and the LLM's scores are far apart, and the LLM is the one that is wrong. Qwen-7B is right 28.1% of the time, below chance for a three-way task, while stating a mean confidence of 0.93. FinBERT is right on 70.8% of the same rows. A confidence threshold on the LLM would not flag these rows. Where the specialist and the LLM diverge sharply, the evidence favours the specialist.

In the domain-shift quadrant the lexicon is the odd one out, and its accuracy collapses to 10.3%.

*A correction.* Earlier versions of this page, and the paper's shorthand, described the ambiguous quadrant as "the two cheap tiers agree but the LLM dissents". That is not the rule: VADER and FinBERT agree on only 61% of ambiguous rows. Restricting to rows where their labels do agree and Qwen-7B dissents (12.2% of FPB) gives Qwen-7B 32.6% and FinBERT 66.9%, which supports the same conclusion.

Negative-class mean `SDI_LE` = 0.945.

*Reproduce:* `experiments/L2_sdi_analysis.py` → `results/data/sdi_data.csv`

---

## 8. SDI as a hallucination detector

| SDI variant | Mean when LLM correct | Mean when LLM wrong | Welch t | p | **AUC** |
|---|---|---|---|---|---|
| `SDI_LE` | 0.394 | 0.499 | 8.66 | 1.3e−17 | 0.620 |
| `SDI_LR` | 0.324 | 0.386 | 5.27 | 1.6e−07 | 0.575 |
| **`SDI_ER`** | **0.168** | **0.709** | **46.82** | **6.2e−255** | **0.898** |
| `SDI_max` | 0.443 | 0.797 | 32.21 | 3.2e−176 | 0.782 |

Only the **cross-granularity encoder-vs-reasoner** pair carries the trust signal. The other pairings are weak detectors, which validates that granularity stratification — not disagreement in general — is what makes SDI work.

*Reproduce:* `experiments/L5p5_e2_hallucination_detector.py` → `results/data/e2_hallucination.csv`

---

## 9. Adversarial robustness (the honest negative result)

| Attack type | SDI detection AUC |
|---|---|
| Negation insertion | 0.71 |
| Synonym swap | ≈ 0.50 |
| Numeric flip | ≈ 0.50 |
| Character drop / typo | ≈ 0.50 |

SDI is a strong hallucination detector and a **weak adversarial detector**. It catches negation attacks and essentially nothing else. Do not deploy it as an adversarial defence.

*Reproduce:* `experiments/L5p5_e1_adversarial.py` → `results/data/e1_auc.csv`

---

## 10. Shared Consensus Dictionary — monolingual trade-off

FPB, 3,386 build / 1,452 query split:

| τ | Hit rate | F1 loss vs. always-committee |
|---|---|---|
| 0.95 | 3% | −0.3 pp |
| **0.85** | **10%** | **−0.8 pp** |
| 0.50 | 83% | −14.5 pp |

Scaling the cache to 16,769 sentences (FPB + TFNS) preserves the curve, confirming the trade-off is not testbed-specific.

*Reproduce:* `experiments/L8_public_dictionary.py` → `results/data/scd_threshold_sweep.csv`, `scd_scaleup_16k.csv`

---

## 11. Shared Consensus Dictionary — cross-lingual canonicalisation

Chinese-translated FPB queries matched against the **English** cache:

| τ | Cross-lingual hit rate | Accuracy vs. gold | F1 vs. gold |
|---|---|---|---|
| 0.50 | 100.0% | 0.987 | 0.983 |
| 0.60 | 99.2% | 0.988 | 0.983 |
| **0.70** | **94.5%** | **0.990** | **0.987** |
| 0.75 | 86.9% | 0.992 | 0.990 |
| 0.80 | 75.3% | 0.997 | 0.996 |
| 0.85 | 54.9% | 1.000 | 1.000 |

**At τ = 0.70, 95% of Chinese queries are answered from the English cache at F1 = 0.99, without running a Chinese committee at all.**

*Reproduce:* `experiments/L7p5_cross_lingual_committee.py` → `results/data/scd_cross_lingual.csv`

---

## 12. Cross-lingual model ranking inversion

| Language | Specialist | Specialist F1 | Qwen-7B F1 | Winner |
|---|---|---|---|---|
| English | FinBERT | **0.88** | 0.81 | specialist |
| Mandarin | `finbert-tone-chinese` | 0.72 | **0.80** | **LLM** |

Chinese multi-size sweep on 1,500 translated FPB sentences: Qwen-0.5B 0.59, 1.5B 0.72, 3B 0.59, 7B 0.80. The 3B anomaly persists across languages.

On **FinChinaSentiment** (5,738 genuine Chinese sentences), English VADER and English FinBERT both score F1 = 0.06 — essentially random, because they do not process Chinese at all.

**Implication:** optimal committee architecture is language-specific. Where no strong specialist exists, the LLM *is* the specialist and the router should invert.

*Reproduce:* `experiments/L7_chinese_pilot.py`, `experiments/L7p5_cross_lingual_committee.py`

---

## 13. Per-class F1 (FPB), sorted by negative-class performance

| Strategy | negative | neutral | positive |
|---|---|---|---|
| **debate@7B** | **0.893** | 0.880 | 0.844 |
| FinBERT | 0.879 | 0.901 | 0.866 |
| critic@7B | 0.878 | 0.880 | 0.707 |
| critic@1.5B | 0.876 | 0.889 | 0.843 |
| critic@3B | 0.876 | 0.894 | 0.855 |
| Qwen-7B | 0.852 | 0.872 | 0.708 |
| debate@Mistral-7B | 0.821 | 0.866 | 0.781 |
| debate@3B | 0.785 | 0.870 | 0.781 |

`debate@7B` is the **only** protocol that strictly beats the specialist on any class. The +1.4 pp lift on negative is **precision-side** (+2.1 pp) at near-equal recall (0.972 vs 0.970) — the operationally meaningful regime for risk management.

*Reproduce:* `experiments/L5p5_e3_bias_diversity_detail.py` → `results/data/e3_per_class.csv`

---

## 14. Ablation — which tier is irreplaceable?

| Configuration | Macro-F1 | Cost per 1k | Note |
|---|---|---|---|
| V + F + L (Balanced) | 0.88 | $0.0006 | full committee |
| **V + F** (drop the LLM) | **0.70** | **$0.00008** | 7.5× cheaper, only 1.8 pp behind |
| V + L (drop the encoder) | −13 pp | — | **removing L2 is catastrophic** |

**The specialist is the irreplaceable tier; the LLM is the most expendable.** Pick the domain specialist first — the LLM is the cherry, not the cake.

*Reproduce:* `experiments/L3_routing_pareto.py` → `results/data/pareto_points_7b.csv`

---

## 15. Annual inference cost at deployment scale

10 queries per user per day, public 2025 API prices:

| Strategy | 10K users | 1M users | 10M users |
|---|---|---|---|
| Always-GPT-4 | $365K | $36.5M | **$365M** |
| Always-GPT-4o-mini | $11K | $1.10M | **$11.0M** |
| Always-Qwen-7B (self-hosted L3) | $1.05K | $105K | $1.05M |
| Always-FinBERT (L2) | $18 | $1.8K | $18K |
| **TriAgent** | — | — | **$1.65M** |

**Savings at 10M users: $363M/yr against always-GPT-4; $9.3M/yr against always-GPT-4o-mini.**

Query distribution at the Balanced operating point: **70% L1 / 28.5% L2 / 1.5% L3**.

These are scenario estimates conditioned on the measured cache hit rate and specialist accuracy transferring to the production workload, and on per-query API prices remaining within an order of magnitude of 2025 published rates.

*Reproduce:* `experiments/L3p5_scaling.py` → `paper/figures/code/make_fig_cost_at_scale.py`

---

## 16. Edge-side routing predictor

Predicting high-SDI (escalation-needed) queries from features available *before* running the committee:

| Feature set | AUC |
|---|---|
| LR, unigram | 0.69 |
| LR, unigram + bigram | 0.70 |
| LR, unigram + bigram + sentence-embedding | 0.74 |
| **XGBoost, all features** | **0.85** |
| XGBoost + LLM reasoning text | 0.94 |

The 0.94 row is a **non-deployable upper bound** — it requires the LLM output you were trying to avoid computing. The deployable 0.85 captures 90% of that ceiling.

Bigram lift is small on both formal (+0.003 on FPB) and informal (+0.004 on TFNS) text, which walks back an earlier hypothesis that phrase features matter more for customer language.

*Reproduce:* `experiments/L4_predictor.py --with-reasoning` → `results/data/predictor_results.csv`

---

## 17. 20-ticker trading back-test

| Strategy | Sharpe | Return % | Max drawdown % | Win rate | Trades |
|---|---|---|---|---|---|
| **SDI-Single (S5)** | **3.499** | **3.36** | −1.27 | 59.3% | 41 |
| SDI-Two-Stage (S6) | 2.896 | 2.76 | −1.44 | 58.7% | 42 |
| debate@7B | 1.77 | 1.03 | −1.1 | 53.6% | — |
| critic@Mistral-7B | 1.574 | 2.03 | −1.73 | 52.9% | 42 |
| Always-VADER (L1) | 1.508 | 1.97 | −1.81 | 55.0% | 49 |
| Always-FinBERT (L2) | 1.361 | 0.82 | −0.99 | 51.6% | 19 |
| Oracle | 0.602 | 0.54 | −1.31 | 51.3% | 19 |
| **Always-Qwen-7B (L3)** | **0.112** | 0.26 | −0.83 | 47.5% | 12 |

Two observations worth noting. The most expensive policy is the **worst** on a risk-adjusted basis. And always-VADER (Sharpe 1.51) beats always-FinBERT (1.36) despite far worse classification F1 — accuracy and risk-adjusted return are only loosely coupled.

*Reproduce:* `experiments/L5_backtest.py` → `results/data/backtest_summary_aggregate.csv`

---

## Reproducing everything

```bash
git clone https://github.com/graphuofm/TRIAGENT && cd TRIAGENT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# then run experiments/L1 … L9 in order — see README.md
```

End-to-end wall-clock: **6–8 hours** on one A5000, dominated by LLM inference.
