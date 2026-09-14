# TriAgent Glossary

Definitions of the terms introduced or used in a specific sense by TriAgent ([CIKM 2026](https://doi.org/10.1145/3799682.3839978), [arXiv:2607.19794](https://arxiv.org/abs/2607.19794)).

---

## Core concepts

### Semantic Divergence Index (SDI)

The pairwise absolute difference between two agents' continuous scores, used as an unsupervised routing signal. Given scores `s_V`, `s_F`, `s_L` in [−1, +1] from the lexicon, encoder, and reasoner tiers, TriAgent defines three SDIs:

| Name | Definition | Interpretation |
|---|---|---|
| `SDI_LE` | \|`s_V` − `s_F`\| | lexicon vs. encoder — is the surface form misleading? |
| `SDI_LR` | \|`s_V` − `s_L`\| | lexicon vs. reasoner |
| `SDI_ER` | \|`s_F` − `s_L`\| | encoder vs. reasoner — the trust signal |

Each SDI lies in [0, 2]. SDI is *not* a learned quantity: it is arithmetic over outputs the pipeline already computed, which is why TriAgent needs no labelled routing data.

> Disambiguation: "SDI" is overloaded in other literature (Shannon Diversity Index, Sustainable Development Indicator, Spatial Data Infrastructure). In TriAgent it always means Semantic Divergence Index.

### Granularity stratification

The design principle that committee members should be chosen so each consumes a **strictly larger context window** than the previous one — word, then sentence, then cross-sentence. This makes their failure modes orthogonal *by construction* rather than by luck. Measured: pairwise Cohen's κ 0.19–0.61, pairwise Jaccard error overlap 0.13–0.15.

This is the property that makes SDI informative. Its absence is what makes same-model persona voting fail (F1 = 0.66, below a single agent's 0.69).

### Critic plateau

The empirical finding that when an LLM is re-tasked as a *critic* over cheaper agents' outputs, macro-F1 saturates at roughly 0.87 across Qwen-1.5B, 3B, and 7B, with overlapping bootstrap 95% confidence intervals. Summarised as **interaction substitutes for parameters**: a well-scaffolded small model matches a large one, at one third the cost.

The plateau is *calibration-conditional* — see below.

### Calibration-conditional

A property that holds only when the mid-tier specialist is well calibrated. The critic plateau is calibration-conditional: it holds on Financial PhraseBank, where FinBERT's Expected Calibration Error is 0.016, and **fails** on TFNS, where FinBERT collapses to F1 = 0.66 and the critic protocol drops to 0.65, below the LLM-alone baseline of 0.76.

### Shared Consensus Dictionary (SCD)

A semantic cache storing **committee decisions**, keyed by multilingual sentence-BERT embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions). A query is embedded, k-NN searched by cosine similarity, and served the cached label when the best match σ* ≥ τ.

Two properties distinguish it from a conventional LLM cache: it caches committee output rather than single-model output, and its multilingual encoder turns it into a **cross-lingual canonicaliser** — at τ = 0.70, 95% of Chinese queries hit English entries at F1 = 0.99.

### Four-quadrant routing

The partition of query space induced by thresholding `(SDI_LE, SDI_ER)`:

| Quadrant | Condition | Meaning | Action |
|---|---|---|---|
| **Consensus** | both low | all tiers agree | return the cheap answer |
| **Domain shift** | `SDI_LE` high, `SDI_ER` low | lexicon disagrees with the other two | trust the encoder |
| **Ambiguous** | `SDI_LE` low, `SDI_ER` high | cheap pair agrees, reasoner dissents | **skip the LLM** — its accuracy here is 28% |
| **Mixed** | both high | genuine difficulty | fire an interaction protocol |

The ambiguous quadrant is the paper's most counter-intuitive result: it is exactly where a confidence-threshold cascade would escalate, and exactly where escalating is wrong.

---

## The three tiers

### L1 — word-level lexicon

The cheapest tier. Scores text by looking up individual tokens in a weighted dictionary. Reference instance: **VADER** (~0.03 ms/query, effectively $0). Generic substitutes: any keyword table, regex rule set, or hand-built trigger lexicon.

### L2 — sentence-level domain transformer

The mid tier. A fine-tuned encoder that reads one sentence of context. Reference instance: **FinBERT** (110M parameters, ~1.5 ms/query, ~$0.0005 per 1k). Generic substitutes: any domain-tuned BERT-class classifier.

Called *the irreplaceable tier* in the paper: removing L2 costs 13 percentage points of F1, whereas removing L3 costs only 1.8 points.

### L3 — cross-sentence reasoner

The expensive tier. A generative LLM that reads full context and can reason across sentences. Reference instances: **Qwen2.5-Instruct** (0.5B–14B-4bit), **Mistral-7B**, **Phi-3.5-mini**. Hundreds of milliseconds, ~$0.11 per 1k queries self-hosted.

---

## Interaction protocols

Fired only when divergence is high. All three take the same inputs and differ in how the LLM is prompted.

### Vote

Each agent emits a label; majority wins. Included mainly as a baseline — it does not produce the plateau.

### Critic

The LLM receives the original query **plus** the L1 and L2 predictions, and is asked for a final label. This is the protocol that plateaus: 0.87 at 1.5B, 3B, and 7B alike.

### Debate

A second round in which the LLM additionally sees all three round-1 outputs, including its own rationale, and reconciles. Does **not** plateau — 0.69 at 1.5B, 0.81 at 3B, 0.87 at 7B — because a small model's own round-1 rationale is noise it cannot discount. At 7B it becomes the strongest protocol on the negative class (F1 = 0.893).

### Persona vote (the negative control)

Three prompts of the *same* model — bull, bear, neutral — voting together. Reaches F1 = 0.66, below the 0.69 that a single instance of the same model achieves. This experiment exists to rule out "multi-agent voting" and "self-consistency" as explanations for the critic plateau.

---

## Datasets

### FPB — Financial PhraseBank

4,838 curated financial news sentences (`sentences_allagree` split): 604 negative, 2,872 neutral, 1,362 positive. The **well-specified regime** — FinBERT reaches F1 = 0.88 with Expected Calibration Error 0.016.

### TFNS — Twitter Financial News Sentiment

11,931 informal tweets: 1,789 negative, 7,744 neutral, 2,398 positive. The **hard regime** — FinBERT collapses to F1 = 0.66, and the critic plateau fails. Included specifically to test whether the architectural claims are testbed-conditional.

---

## Metrics

### Expected Calibration Error (ECE)

The gap between a model's stated confidence and its realised accuracy, bucketed over the confidence range. Lower is better. Measured here: **FinBERT 0.016** (nearly perfect), **Qwen-7B 0.115** (over-confident — claims 94% confidence, achieves 82.6% accuracy), **VADER 0.348** (uncalibrated).

ECE is the mechanism that explains *when* the critic plateau holds.

### Macro-F1

Unweighted mean of per-class F1, so every class counts equally regardless of size. Used throughout because the negative class is both the smallest (604/4,838) and operationally the most valuable for risk management.

### Bootstrap 95% confidence interval

Resampling-based interval used to support the plateau claim. Key intervals: Qwen-7B alone [0.796, 0.820], FinBERT alone [0.873, 0.893] — non-overlapping, so the "specialist beats LLM" result is statistically separated. All committee and critic strategies sit in [0.86, 0.88] with overlapping intervals, which is what makes the plateau a plateau rather than a trend.

### Sharpe ratio

Risk-adjusted return, used in the 20-ticker back-test. Higher is better. SDI-Single reaches 3.50; always-Qwen-7B reaches 0.11.

---

## Related terms from other systems

| Term | System | Relation to TriAgent |
|---|---|---|
| Model cascade | [FrugalGPT](https://arxiv.org/abs/2305.05176) | Same shape; escalates on *learned confidence* rather than inter-model divergence, and requires labelled training data |
| LLM router | [RouteLLM](https://arxiv.org/abs/2406.18665) | Same goal; trains on human preference data, where SDI is unsupervised |
| Semantic cache | [GPTCache](https://doi.org/10.18653/v1/2023.nlposs-1.24) | SCD is a semantic cache that additionally stores *committee* decisions and is cross-lingual |
| Self-consistency | [Wang et al.](https://arxiv.org/abs/2203.11171) | Re-samples one model *n* times; SDI compares *different* models at zero marginal cost |
| Multi-agent debate | [Du et al.](https://arxiv.org/abs/2305.14325) | Homogeneous agents differing by prompt; TriAgent's negative control shows that configuration does not reproduce the plateau |
| LLM-as-a-judge | [Zheng et al.](https://arxiv.org/abs/2306.05685) | The critic protocol is judge-shaped, but judges a *small* model over cheap agents rather than a large model over generations |
