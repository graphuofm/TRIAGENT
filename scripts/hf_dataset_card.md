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

This dataset contains every per-sentence prediction behind **TriAgent**, a divergence-aware multi-agent routing framework for cost-efficient LLM inference.

- **Paper (CIKM 2026):** [doi.org/10.1145/3799682.3839978](https://doi.org/10.1145/3799682.3839978)
- **arXiv:** [arxiv.org/abs/2607.19794](https://arxiv.org/abs/2607.19794)
- **Code:** [github.com/graphuofm/TRIAGENT](https://github.com/graphuofm/TRIAGENT)
- **Project page:** [graphuofm.github.io/TRIAGENT](https://graphuofm.github.io/TRIAGENT/)

## Dataset summary

The dataset holds 25,607 rows across five configurations. For each sentence it records what each of ten models predicted, with a score, a confidence, latency, cost and, for LLMs, free-text reasoning. It also records the Semantic Divergence Index (SDI) between model tiers and the output of eight multi-agent interaction protocols.

Two uses motivate the release:

- **Reproduce the paper without a GPU.** Every macro-F1 in the paper can be recomputed from these columns.
- **Evaluate a new routing policy offline.** Any rule that picks among the stored predictions can be scored for accuracy and cost with no model calls.

TriAgent stratifies agents by contextual granularity. **L1** is a word-level lexicon (VADER). **L2** is a sentence-level domain transformer (FinBERT). **L3** is a cross-sentence LLM reasoner (Qwen2.5). The routing signal is the absolute difference between two agents' scores.

## What the data shows

| Finding | Result | Columns |
|---|---|---|
| **Critic plateau** | An LLM that adjudicates between L1 and L2 reaches macro-F1 0.871 at 1.5B and 0.870 at 3B. The same models alone score 0.688 and 0.624. | `fpb`: `critic_qwen1p5b_label`, `critic_qwen3b_label` |
| **The mechanism is not voting** | Three persona prompts of the same Qwen-1.5B, voting together, score 0.663. That is below a single instance at 0.688. | `fpb`: `persona_qwen1p5b_vote_label` |
| **The plateau is calibration-conditional** | On tweets, FinBERT drops to 0.663 and critic@1.5B to 0.659, below Qwen-7B alone at 0.757. | `tfns`: `critic_qwen1p5b_label` |
| **Free hallucination detector** | `sdi_er` separates correct from incorrect Qwen-7B outputs at ROC AUC 0.898. | `fpb`: `sdi_er`, `qwen7b_label` |
| **Confidently wrong LLM** | Where `sdi_er > 0.7`, Qwen-7B is right 28.1% of the time at a mean stated confidence of 0.93. FinBERT is right 70.8% of the time on the same rows. | `fpb`: `quadrant == "ambiguous"` |
| **Specialist beats LLM** | FinBERT reaches 0.883 and Qwen-7B reaches 0.809. | `fpb`: `finbert_label`, `qwen7b_label` |
| **Ranking inverts in Mandarin** | `finbert-tone-chinese` reaches 0.721 and Qwen-7B reaches 0.801. | `fpb_zh` |
| **Weak adversarial detector** | SDI detects synonym, numeric and character-drop perturbations at AUC 0.46–0.51. | `fpb_adversarial` |

The build script recomputes every agent and protocol macro-F1 from these columns. It fails if any result differs from the paper.

## Configurations

| Config | Rows | Source | Contents |
|---|---|---|---|
| `fpb` *(default)* | 4,838 | Financial PhraseBank, `sentences_allagree` | 3 base agents, 7 LLM runs, SDI, quadrants, 8 interaction protocols, persona-vote control |
| `tfns` | 11,931 | Twitter Financial News Sentiment | Base agents, Qwen 1.5B/3B/7B, SDI, critic@1.5B and critic@7B |
| `fpb_zh` | 1,500 | FPB sentences machine-translated to Mandarin | Qwen 0.5B–7B in Chinese, `finbert-tone-chinese`, Chinese committee labels |
| `fpb_adversarial` | 2,500 | 500 FPB sentences × 5 conditions | Perturbed text, agent labels and SDI for `clean`, `synonym`, `negation`, `numeric`, `chardrop` |
| `scd_index` | 4,838 | FPB | The Shared Consensus Dictionary: 384-d L2-normalised multilingual sentence embeddings |

Each configuration has one split, named `evaluation`. No row is training data: every row was used for evaluation, and gold labels come from the source corpora.

The `summaries/` folder holds 20 CSV aggregate tables. They cover calibration error, bootstrap confidence intervals, per-class F1, cache threshold sweeps, cost Pareto points, predictor AUCs and the 20-ticker back-test, and they are the direct inputs to the paper's tables.

## Loading

```python
from datasets import load_dataset

fpb = load_dataset("dingjiacheng/triagent", "fpb", split="evaluation")
tfns = load_dataset("dingjiacheng/triagent", "tfns", split="evaluation")
```

To reproduce the critic plateau:

```python
from sklearn.metrics import f1_score

df = fpb.to_pandas()
for col in ["qwen1p5b_label", "critic_qwen1p5b_label", "qwen3b_label", "critic_qwen3b_label"]:
    print(f"{col:24s} {f1_score(df['label_text'], df[col], average='macro'):.4f}")
```

To evaluate a routing policy with no model calls:

```python
import numpy as np

route_to_llm = df["sdi_er"] > 0.7          # your policy here
pred = np.where(route_to_llm, df["qwen7b_label"], df["finbert_label"])
cost = np.where(route_to_llm, df["qwen7b_cost_usd"], df["finbert_cost_usd"]).sum()
print(f1_score(df["label_text"], pred, average="macro"), f"${cost:.4f}")
```

## Dataset structure

### Common columns

| Column | Meaning |
|---|---|
| `sentence_id` | Row identifier within the source corpus, stable across configs derived from FPB |
| `sentence` | Source text (`sentence_en` and `sentence_zh` in `fpb_zh`) |
| `label` | Gold label as an integer: 0 = negative, 1 = neutral, 2 = positive |
| `label_text` | Gold label as a string |

### Models

| Column prefix | Model | Tier |
|---|---|---|
| `vader_` | VADER lexicon | L1 |
| `finbert_` | `ProsusAI/finbert` | L2 |
| `qwen0p5b_`, `qwen1p5b_`, `qwen3b_`, `qwen7b_` | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct`, bf16 | L3 |
| `qwen14b_4bit_` | `Qwen/Qwen2.5-14B-Instruct`, 4-bit via bitsandbytes | L3 |
| `mistral7b_` | `mistralai/Mistral-7B-Instruct-v0.3` | L3 |
| `finbert_tone_chinese_` | `yiyanghkust/finbert-tone-chinese` (`fpb_zh` only) | L2 |

**`qwen7b_` is the canonical L3.** All SDI columns and quadrants use Qwen-7B as the reasoner.

### Per-agent fields

`_label` is always one of `negative`, `neutral` or `positive`. `_score` and `_confidence` mean different things for each model type, so they are not directly comparable across tiers.

| Model type | `_score` in [−1, +1] | `_confidence` in [0, 1] |
|---|---|---|
| VADER | the compound polarity score | the absolute value of the compound score, **not a probability** |
| FinBERT | P(positive) − P(negative) | the highest softmax probability |
| LLMs | **self-reported** in the model's JSON reply | **self-reported** in the model's JSON reply |

Other fields:

| Field | Meaning |
|---|---|
| `_latency_ms` | measured wall-clock time for the call |
| `_cost_usd` | amortised GPU cost for the call, at $0.40 per GPU-hour |
| `_reasoning` | the LLM's free-text rationale |
| `_input_tokens`, `_output_tokens` | token counts for the call |
| `vader_pos`, `vader_neg`, `vader_neu` | VADER's component scores |
| `finbert_prob_pos`, `finbert_prob_neg`, `finbert_prob_neu` | FinBERT's softmax probabilities |

VADER labels a sentence positive at compound ≥ 0.05, negative at ≤ −0.05, and neutral otherwise.

### Semantic Divergence Index

| Column | Definition |
|---|---|
| `sdi_le` | \|`vader_score` − `finbert_score`\| |
| `sdi_lr` | \|`vader_score` − `qwen7b_score`\| |
| `sdi_er` | \|`finbert_score` − `qwen7b_score`\| |
| `sdi_max`, `sdi_mean` | the maximum and mean of the three |
| `disagreement_entropy` | Shannon entropy, in bits, of the three agents' labels. It takes only three values: 0 when all agree, 0.918 when two agree, and 1.585 when all differ (`fpb` only). |

`quadrant` (`fpb` only) applies these rules in order and assigns the first one that matches:

| Order | Quadrant | Rule | Share of FPB | Qwen-7B acc. | FinBERT acc. |
|---|---|---|---|---|---|
| 1 | `ambiguous` | `sdi_er > 0.7` | 15.9% | 28.1% | 70.8% |
| 2 | `consensus` | `sdi_le < 0.3` and `sdi_er < 0.3` | 39.4% | 96.5% | 96.4% |
| 3 | `domain_shift` | `sdi_le > 0.7` and `sdi_er < 0.3` | 13.5% | 95.5% | 95.5% |
| 4 | `mixed` | everything else | 31.2% | 87.2% | 86.0% |

Because `ambiguous` is checked first, a row with both divergences high counts as `ambiguous`. That quadrant is where the encoder and the reasoner strongly disagree. It is not where the two cheap models agree: VADER and FinBERT share a label on only 61% of `ambiguous` rows.

### Interaction protocols

Protocol columns are named `{protocol}_{model}_`, for example `critic_qwen1p5b_label`. Each protocol calls the LLM only on rows where a gate fires. Every other row copies a fallback answer.

| Protocol | Gate | Fallback when not triggered | What the LLM sees |
|---|---|---|---|
| `critic_` | `sdi_le > 0.4` | FinBERT's label, score and confidence | the sentence, plus VADER's and FinBERT's labels, scores and confidences |
| `debate_` | `sdi_max > 0.5` | the same-size LLM's own label | the sentence, plus VADER's, FinBERT's and the LLM's round-1 output, including its own rationale |

The fields are `_label`, `_score`, `_confidence`, `_triggered`, `_extra_cost_usd`, `_extra_latency_ms` and `_rationale`. On rows that did not trigger, **`_rationale` is null by design** because the LLM was never called. The paper's macro-F1 covers all rows.

| Run | Rows triggered | Empty rationales |
|---|---|---|
| each `critic_*` in `fpb` | 2,041 (42.2%) | 2,797 |
| each `debate_*` in `fpb` | 2,191 (45.3%) | 2,647 |
| each `critic_*` in `tfns` | 4,914 (41.2%) | 7,017 |

The runs in `fpb` are critic with Qwen-1.5B, Qwen-3B, Phi-3.5-mini (`microsoft/Phi-3.5-mini-instruct`) and Mistral-7B, and debate with Qwen-1.5B, Qwen-3B, Qwen-7B and Mistral-7B. The runs in `tfns` are critic with Qwen-1.5B and Qwen-7B.

### Persona-vote control (`fpb`)

`persona_qwen1p5b_{bull,bear,neutral}_{label,score,confidence}` and `persona_qwen1p5b_vote_label` come from three system prompts on the same `Qwen/Qwen2.5-1.5B-Instruct`:

| Persona | System prompt intent |
|---|---|
| `bull` | a sell-side equity analyst with a long bias, leaning positive |
| `bear` | a short-only credit analyst, leaning negative |
| `neutral` | a conservative compliance analyst, leaning neutral unless the evidence is unambiguous |

The vote is their majority label. This is the negative control that separates granularity stratification from plain multi-agent voting.

### Other configs

**`fpb_zh`** has `qwen{0p5b,1p5b,3b,7b}_*` and `finbert_tone_chinese_*` columns scored on the Chinese text. `committee_qwen{1p5b,3b,7b}` combines FinBERT-tone-chinese with the matching Qwen model. When the two agree, their label wins. When they disagree, the model with higher confidence wins if the gap exceeds 0.05, and the result is `neutral` otherwise.

**`fpb_adversarial`** has `sentence_original`, `perturbation`, `sentence_perturbed`, VADER/FinBERT/Qwen-7B labels and scores, and the SDI columns recomputed on the perturbed text.

**`scd_index`** has `sentence_id`, `sentence`, `label_text` and `embedding`. The embedding is a 384-d float32 vector from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, L2-normalised, so a dot product gives cosine similarity.

## Dataset creation

### Source data

- **Financial PhraseBank**, the `sentences_allagree` subset: 4,838 sentences, 604 negative, 2,872 neutral and 1,362 positive. It was loaded through `ChanceFocus/flare-fpb`.
- **Twitter Financial News Sentiment** (`zeroshot/twitter-financial-news-sentiment`): 11,931 tweets, 1,789 negative, 7,744 neutral and 2,398 positive.
- **Chinese translations**: 1,500 FPB sentences translated with `Qwen/Qwen2.5-7B-Instruct`, which was prompted to preserve numerical values and to output only the translation.

### How the predictions were produced

- **Base LLM prompt.** Each LLM receives an expert-financial-analyst instruction with four domain rules. For example, "loss narrowed" counts as positive, and "liability" or "debt" are often neutral accounting terms. The model must reply with JSON containing `sentiment`, `score`, `confidence` and `reasoning`.
- **Decoding.** All LLM calls use greedy decoding (`do_sample=False`) with at most 200 new tokens. Critic and debate inputs are truncated to 1,024 tokens, and every prompt goes through the model's chat template.
- **Hardware.** Everything ran on a single NVIDIA RTX A5000 (24 GB). Models ran in bf16, except Qwen-14B, which ran in 4-bit.
- **Parse failures.** If an LLM's reply cannot be parsed as JSON, the row falls back to `neutral`, with score 0.0 and confidence 0.0. This happened on 39 Qwen-0.5B rows and 327 Qwen-1.5B rows in `fpb`. In `tfns` it happened on 1,403 Qwen-1.5B rows and on one row each for Qwen-3B and Qwen-7B. It lowers the small models' reported accuracy, and it is part of the measured result.

## Personal and sensitive information

`fpb` and its derived configs contain sentences from financial news and company releases.

`tfns` contains public tweets as distributed by the source dataset. 3.1% of rows include an `@handle`, mostly institutional accounts, and 47.0% include a shortened `t.co` link. No text was removed or altered. If you redistribute derived data, consider whether handles should be masked.

## Limitations

- **The critic gate differs from the code default.** The code in the repository defaults to a critic threshold of 0.3, but these runs used `sdi_le > 0.4`. The stored `_triggered` column reflects the 0.4 gate exactly on both corpora. To reproduce the paper's costs, use `_triggered`.
- **There are only two testbeds, and both are financial sentiment.** The architecture is task-agnostic, but these measured quantities should not be assumed to transfer.
- **Confidence is not calibrated evidence.** VADER's confidence is not a probability, and LLM confidence is self-reported: Qwen-7B states between 0.70 and 1.00 on every row.
- **The largest unquantised model is 7B.** Qwen-14B was run at 4-bit and scores below Qwen-7B in bf16, so it is not a clean scale point.
- **Costs are amortised estimates.** Treat `_cost_usd` as relative, not as a bill.
- **The Chinese set is machine-translated.** `fpb_zh` measures cross-lingual transfer on translated text, not native Chinese financial language.
- **A few LLM rationales are missing where generation returned no text.** This affects six Mistral-7B rows in `fpb` and four Qwen-3B rows in `tfns`. The labels are present.

## Licensing

**This dataset is released under CC BY-NC-SA 3.0.**

- **Financial PhraseBank** is licensed CC BY-NC-SA 3.0. Its ShareAlike term covers derived works, so it applies to `fpb`, `scd_index`, `fpb_adversarial` and `fpb_zh`. The `ChanceFocus/flare-fpb` repackaging carries an MIT tag, but it cannot relax the upstream license.
- **Twitter Financial News Sentiment** is MIT-licensed. MIT permits redistribution under these terms with attribution, which this card provides.
- **FinChinaSentiment is excluded.** The paper reports results on it, but its source declares no license, so its text is not redistributed.

The non-commercial restriction comes from Financial PhraseBank. The TriAgent code is MIT-licensed separately.

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

Please also cite Financial PhraseBank:

```bibtex
@article{malo2014good,
  title   = {Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts},
  author  = {Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal = {Journal of the Association for Information Science and Technology},
  volume  = {65},
  number  = {4},
  pages   = {782--796},
  year    = {2014},
}
```

## Contact

Jiacheng Ding, The University of Memphis, `jding2@memphis.edu`
