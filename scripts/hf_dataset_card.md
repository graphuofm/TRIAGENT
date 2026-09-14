---
license: cc-by-nc-sa-3.0
language:
- en
- zh
pretty_name: "TriAgent: Multi-Agent Committee Predictions for Financial Sentiment"
size_categories:
- 10K<n<100K
task_categories:
- text-classification
task_ids:
- sentiment-classification
tags:
- finance
- financial-sentiment-analysis
- llm-routing
- llm-evaluation
- model-cascade
- multi-agent
- hallucination-detection
- semantic-cache
- calibration
- cross-lingual
source_datasets:
- extended|financial_phrasebank
- extended|zeroshot/twitter-financial-news-sentiment
configs:
- config_name: fpb
  default: true
  data_files:
  - split: evaluation
    path: data/fpb.parquet
- config_name: tfns
  data_files:
  - split: evaluation
    path: data/tfns.parquet
- config_name: fpb_zh
  data_files:
  - split: evaluation
    path: data/fpb_zh.parquet
- config_name: fpb_adversarial
  data_files:
  - split: evaluation
    path: data/fpb_adversarial.parquet
- config_name: scd_index
  data_files:
  - split: evaluation
    path: data/scd_index.parquet
---

# TriAgent: Multi-Agent Committee Predictions for Financial Sentiment

Every per-sentence prediction behind **TriAgent**, a divergence-aware multi-agent routing framework for cost-efficient LLM inference.

- **Paper (CIKM 2026):** [doi.org/10.1145/3799682.3839978](https://doi.org/10.1145/3799682.3839978)
- **arXiv:** [arxiv.org/abs/2607.19794](https://arxiv.org/abs/2607.19794)
- **Code:** [github.com/graphuofm/TRIAGENT](https://github.com/graphuofm/TRIAGENT)
- **Project page:** [graphuofm.github.io/TRIAGENT](https://graphuofm.github.io/TRIAGENT/)

The dataset holds 25,607 rows across five configurations. It records what each of ten models predicted for each sentence, with its score, confidence, latency, cost and free-text reasoning. It also holds the Semantic Divergence Index (SDI) between tiers, and the output of eight multi-agent interaction protocols. With it you can re-derive every table in the paper without running a GPU, or test a new routing policy against the same predictions.

## What TriAgent found, and what this data lets you check

TriAgent runs three agents stratified by contextual granularity. **L1** is a word-level lexicon (VADER). **L2** is a sentence-level domain transformer (FinBERT). **L3** is a cross-sentence LLM reasoner (Qwen2.5). The routing signal is the absolute difference between two agents' scores, the SDI.

| Finding | Result | Verify with |
|---|---|---|
| **Critic plateau** | An LLM adjudicating between L1 and L2 reaches macro-F1 0.871 at 1.5B and 0.870 at 3B. Alone, those models score 0.688 and 0.624. | `fpb`: `critic_qwen1p5b_label`, `critic_qwen3b_label` |
| **The mechanism is not voting** | Three prompts of the same Qwen-1.5B voting together score 0.663, below one instance alone (0.688). | `fpb`: `persona_qwen1p5b_vote_label` |
| **The plateau is calibration-conditional** | On tweets, where FinBERT drops to 0.663, critic@1.5B scores 0.660. That is below Qwen-7B alone at 0.757. | `tfns`: `critic_qwen1p5b_label` |
| **Free hallucination detector** | `sdi_er` separates correct from incorrect Qwen-7B outputs at ROC AUC 0.898. | `fpb`: `sdi_er` vs. `qwen7b_label == label_text` |
| **Specialist beats LLM** | FinBERT reaches 0.883; Qwen-7B reaches 0.809. | `fpb`: `finbert_label`, `qwen7b_label` |
| **Ranking inverts in Mandarin** | `finbert-tone-chinese` reaches 0.721; Qwen-7B reaches 0.801. | `fpb_zh` |
| **SDI is a weak adversarial detector** | AUC is near 0.5 on synonym, numeric and character-drop attacks. | `fpb_adversarial` |

Every macro-F1 above is recomputed from these columns by the build script, and the build fails if any column misses the paper's number.

## Configurations

| Config | Rows | Source | Contents |
|---|---|---|---|
| `fpb` *(default)* | 4,838 | Financial PhraseBank, `sentences_allagree` | 3 base agents, 7 LLM sweeps, 5 SDI measures, quadrant label, 8 interaction protocols, persona-vote control |
| `tfns` | 11,931 | Twitter Financial News Sentiment | Base agents, Qwen 1.5B/3B/7B, SDI, critic@1.5B and critic@7B |
| `fpb_zh` | 1,500 | FPB sentences machine-translated to Mandarin | Qwen 0.5B–7B in Chinese, `finbert-tone-chinese`, Chinese committee labels |
| `fpb_adversarial` | 2,500 | 500 FPB sentences × 5 perturbations | Perturbed text, agent labels and SDI for `clean`, `synonym`, `negation`, `numeric`, `chardrop` |
| `scd_index` | 4,838 | FPB | The Shared Consensus Dictionary: 384-d L2-normalised multilingual sentence-BERT embeddings |

Each configuration has a single split named `evaluation`. None of this is a training split. Every row was used for evaluation, and gold labels come from the source corpora.

`summaries/` holds 20 CSV aggregate tables: calibration error, bootstrap confidence intervals, per-class F1, the cache threshold sweeps, the cost Pareto points, predictor AUCs and the 20-ticker back-test. They are the direct inputs to the paper's tables.

## Loading

```python
from datasets import load_dataset

fpb = load_dataset("dingjiacheng/triagent", "fpb", split="evaluation")
tfns = load_dataset("dingjiacheng/triagent", "tfns", split="evaluation")
```

Reproducing the critic plateau from the default config:

```python
from sklearn.metrics import f1_score

df = fpb.to_pandas()
for col in ["qwen1p5b_label", "critic_qwen1p5b_label", "qwen3b_label", "critic_qwen3b_label"]:
    print(f"{col:24s} {f1_score(df['label_text'], df[col], average='macro'):.4f}")
```

Evaluating a new routing policy with no model calls:

```python
import numpy as np

route_to_llm = df["sdi_er"] > 0.7          # your policy here
pred = np.where(route_to_llm, df["qwen7b_label"], df["finbert_label"])
cost = np.where(route_to_llm, df["qwen7b_cost_usd"], df["finbert_cost_usd"]).sum()
print(f1_score(df["label_text"], pred, average="macro"), f"${cost:.4f}")
```

## Column reference

### Models

| Column prefix | Model | Tier |
|---|---|---|
| `vader_` | VADER lexicon | L1 |
| `finbert_` | `ProsusAI/finbert` | L2 |
| `qwen0p5b_`, `qwen1p5b_`, `qwen3b_`, `qwen7b_` | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` | L3 |
| `qwen14b_4bit_` | `Qwen/Qwen2.5-14B-Instruct`, 4-bit via bitsandbytes | L3 |
| `mistral7b_` | `mistralai/Mistral-7B-Instruct-v0.3` | L3 |
| `finbert_tone_chinese_` | `yiyanghkust/finbert-tone-chinese` (`fpb_zh` only) | L2 |

**`qwen7b_` is the canonical L3.** All SDI columns are computed with Qwen-7B as the reasoner.

Each agent has these fields where they apply:

| Field | Meaning |
|---|---|
| `_label` | `negative` / `neutral` / `positive` |
| `_score` | continuous sentiment score in [−1, +1]; the SDI is computed from this |
| `_confidence` | the model's stated confidence in [0, 1] |
| `_latency_ms`, `_cost_usd` | measured wall-clock time and amortised cost for that call |
| `_reasoning` | the LLM's free-text rationale |
| `_input_tokens`, `_output_tokens` | token counts |

VADER additionally has `vader_pos`, `vader_neg` and `vader_neu`. FinBERT additionally has `finbert_prob_pos`, `finbert_prob_neg` and `finbert_prob_neu`.

### Semantic Divergence Index

| Column | Definition |
|---|---|
| `sdi_le` | \|`vader_score` − `finbert_score`\| |
| `sdi_lr` | \|`vader_score` − `qwen7b_score`\| |
| `sdi_er` | \|`finbert_score` − `qwen7b_score`\|, the trust signal |
| `sdi_max`, `sdi_mean` | max and mean of the three |
| `quadrant` | `consensus` / `domain_shift` / `ambiguous` / `mixed` (`fpb` only) |
| `disagreement_entropy` | entropy of the three agents' label distribution (`fpb` only) |

### Interaction protocols

Named `{protocol}_{model}_`, for example `critic_qwen1p5b_label`.

| Protocol | What the LLM sees |
|---|---|
| `critic_` | the sentence plus the L1 and L2 predictions; asked for a final label |
| `debate_` | a second round that also shows all three round-1 outputs, including its own |

The fields are `_label`, `_score`, `_confidence`, `_triggered`, `_extra_cost_usd`, `_extra_latency_ms` and `_rationale`. `_triggered` is True when SDI escalated the row to the protocol. On non-triggered rows the label is the router's default answer and **`_rationale` is null by design**, because the LLM was never called. That leaves 2,797 empty critic rationales and 2,647 empty debate rationales in `fpb`, and 7,017 empty critic rationales in `tfns`. The paper's macro-F1 is computed over all rows.

In `fpb` the protocols are critic with Qwen-1.5B, Qwen-3B, Phi-3.5-mini (`microsoft/Phi-3.5-mini-instruct`) and Mistral-7B, and debate with Qwen-1.5B, Qwen-3B, Qwen-7B and Mistral-7B. In `tfns` they are critic with Qwen-1.5B and Qwen-7B.

### Persona-vote control (`fpb`)

`persona_qwen1p5b_{bull,bear,neutral}_{label,score,confidence}` and `persona_qwen1p5b_vote_label` hold three prompts of the same `Qwen/Qwen2.5-1.5B-Instruct`. This is the negative control that separates granularity stratification from simple multi-agent voting.

## Licensing

**This dataset is released under CC BY-NC-SA 3.0.**

- **Financial PhraseBank** is licensed CC BY-NC-SA 3.0, and its ShareAlike term covers derived works. That applies to `fpb`, `scd_index`, `fpb_adversarial` and `fpb_zh`, which contain FPB sentences or their translations and perturbations. The `fpb` rows were loaded via `ChanceFocus/flare-fpb`. That repackaging carries an MIT tag, but it cannot relax the upstream license, so the upstream terms apply.
- **Twitter Financial News Sentiment** (`zeroshot/twitter-financial-news-sentiment`) is MIT-licensed. MIT permits redistribution under these terms with attribution, which this card provides.
- **Chinese translations** in `fpb_zh` were generated with `Qwen/Qwen2.5-7B-Instruct`.
- **FinChinaSentiment is excluded.** The paper reports results on it, but its source declares no license, so its text is not redistributed here.

The non-commercial restriction comes from Financial PhraseBank. The TriAgent code is separately MIT-licensed.

## Limitations

- **Two testbeds, both financial sentiment.** The architecture is task-agnostic; the measured quantities in these columns are not, and should not be assumed to transfer.
- **The largest unquantised model is 7B.** Qwen-14B was run at 4-bit and scores below Qwen-7B at bf16, so it is not a clean scale point.
- **Costs are amortised estimates.** `_cost_usd` uses GPU rental amortised at $0.40/hour for an A5000-class card. Treat it as relative, not as a bill.
- **The Chinese set is machine-translated.** `fpb_zh` measures cross-lingual transfer on translated text, not native Chinese financial language.
- **Six Mistral-7B rationales in `fpb` and four Qwen-3B rationales in `tfns` are null**, where generation returned empty. The labels on those rows are present.

## Citation

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
```

Please also cite the source corpora:

```bibtex
@article{malo2014good,
  title   = {Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts},
  author  = {Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal = {Journal of the Association for Information Science and Technology},
  year    = {2014},
}
```

## Contact

Jiacheng Ding, The University of Memphis, `jding2@memphis.edu`
