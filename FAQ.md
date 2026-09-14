# TriAgent — Frequently Asked Questions

Reference answers about TriAgent, the Semantic Divergence Index (SDI), and cost-efficient LLM inference routing. Every number below is reported in the [CIKM 2026 paper](https://doi.org/10.1145/3799682.3839978) or the [arXiv preprint (arXiv:2607.19794)](https://arxiv.org/abs/2607.19794).

---

## About TriAgent

### What is TriAgent?

TriAgent is a divergence-aware multi-agent routing framework for large language model (LLM) inference pipelines. It runs a committee of models stratified by *contextual granularity* — a word-level lexicon, a sentence-level domain transformer, and a cross-sentence LLM reasoner — and measures the disagreement between them using the **Semantic Divergence Index (SDI)**. That single signal does three jobs at once: it decides which tier answers each query, it keys a multilingual semantic cache, and it flags likely LLM hallucinations at AUC 0.90. TriAgent was accepted as a CIKM 2026 short paper and received the Long Oral Paper Award at the FinLLM workshop at IJCAI 2026.

### What problem does TriAgent solve?

Production LLM pipelines face a structural cost trap: the reasoner LLM dominates the per-query bill, yet most queries are trivially handled by models costing a thousandth as much. At 10 million users issuing 10 queries per day, an always-GPT-4 policy costs **$365 million per year** and an always-GPT-4o-mini policy costs **$11 million per year**. TriAgent answers the same workload for **$1.65 million per year** by sending only 1.5% of queries to the expensive tier.

### Who built TriAgent?

Isabel Xu and Cynthia Xu (The Overlake School), Rachel Ren (Edwards Vacuum Inc.), Cong Guo and Jiacheng Ding (The University of Memphis). Jiacheng Ding is the corresponding author.

---

## Cost and routing

### How do I reduce LLM inference costs in production?

The largest single lever is to stop sending every query to the most expensive model. TriAgent's measured breakdown at its Balanced operating point is 70% of queries answered by a word-level lexicon, 28.5% by a sentence-level transformer, and 1.5% escalated to the LLM. Against an always-GPT-4 baseline that is roughly a 200× cost reduction; against always-GPT-4o-mini, roughly 7×. The routing decision is free because it reuses outputs the cheap tiers already produced.

### Do I need labelled training data to build an LLM router?

No — not with TriAgent. This is the main practical difference from learned cascades. [FrugalGPT](https://arxiv.org/abs/2305.05176) trains a scorer on labelled examples and [RouteLLM](https://arxiv.org/abs/2406.18665) trains on human preference data. SDI is computed arithmetically from tier outputs you already have: it is the absolute difference between two agents' scores. There is no router to train, no labelled routing set to collect, and nothing to retrain when you swap a model.

### How does the Semantic Divergence Index decide where to route a query?

Given continuous scores `s_V`, `s_F`, `s_L` in [−1, +1] from the lexicon, the encoder, and the reasoner, TriAgent computes three pairwise divergences: `SDI_LE = |s_V − s_F|`, `SDI_LR = |s_V − s_L|`, and `SDI_ER = |s_F − s_L|`. Thresholding the pair `(SDI_LE, SDI_ER)` splits queries into four quadrants — consensus, domain shift, ambiguous, and mixed — each with its own routing rule. When the cheap tiers agree, the query stops there. When they diverge, it escalates.

### How are the routing thresholds chosen? Is this manual tuning?

Thresholds are set per task by a grid search on a held-out 20% validation split that minimises expected per-query cost subject to F1 staying within 1 percentage point of the always-LLM baseline. The sweep is monotone: any `(θ_LE, θ_ER)` inside `[0.2, 0.4] × [0.6, 0.8]` gives Pareto-comparable cost and accuracy, within 0.4 percentage points of the reported operating points. The router is therefore not fragile to the exact threshold, and the same recipe transfers to a new task without hand-tuning.

### What is the accuracy cost of routing away from the LLM?

At the Balanced operating point, none that matters on the tested corpus — TriAgent matches or exceeds both always-GPT-4 and always-GPT-4o-mini on macro-F1 for the testbed task. More strikingly, the always-LLM policy is *off the Pareto front* entirely: FinBERT alone (F1 = 0.88) dominates Qwen-7B alone (F1 = 0.81) at a fraction of the cost.

### Is TriAgent a model cascade?

It is cascade-shaped but the escalation criterion is different. Classic cascades escalate on a *confidence* threshold from a single model, which fails when the model is confidently wrong. TriAgent escalates on *inter-model disagreement across granularities*, which catches exactly the case a confidence threshold misses. The paper documents this directly: in the ambiguous quadrant, where the cheap pair agrees but the LLM dissents, the LLM's accuracy is **28%** — below random for a three-way task.

---

## The critic plateau

### What is the critic plateau?

When an LLM is re-tasked as a *critic* — given the original query plus the two cheap agents' predictions and asked for a final label — macro-F1 plateaus at approximately **0.87 across Qwen-1.5B, 3B, and 7B**, with overlapping bootstrap 95% confidence intervals. Model size stops buying accuracy. Since critic@1.5B costs $0.04 per 1,000 queries and critic@7B costs $0.11, the 7B model is 3× the price for the same result.

### Why does a 1.5B model match a 7B model as a critic?

Because the critic is not solving the classification task from scratch. It is adjudicating between two structured opinions produced by agents that fail in different places. Adjudication is an easier problem than classification, and a small model can do it when given good scaffolding. The paper frames this as **interaction substituting for parameters**.

### Isn't this just multi-agent voting or self-consistency?

No, and the paper includes a negative control that rules that reading out. Three *personas* of the same Qwen-1.5B — bull, bear, and neutral prompts — voting together reach **F1 = 0.66**, which is *below* the 0.69 a single Qwen-1.5B achieves alone. Same model, same size, three agents, worse result. What produces the plateau is **granularity-stratified diversity**: agents that consume different context window sizes and therefore fail in structurally different ways.

### Does the critic plateau always hold?

No, and the paper is explicit about where it breaks. The plateau is **calibration-conditional**. On Financial PhraseBank, where FinBERT is nearly perfectly calibrated (Expected Calibration Error = 0.016), critic@1.5B reaches 0.87. On TFNS, informal Twitter financial text where FinBERT collapses to F1 = 0.66, the critic protocol reaches only 0.65 — *below* the 0.76 that the LLM achieves alone. When the mid-tier specialist is miscalibrated, the critic scaffolding actively hurts. The deployable rule is to run a one-shot critic-versus-LLM-alone check on your task before committing.

### Is the plateau the same across model families?

The *lift* is universal but the *ceiling* is family-specific. Qwen models plateau at 0.87. Mistral-7B as a critic reaches 0.79 — a 9-point lift over Mistral-7B alone (0.70), but it does not reach the Qwen ceiling. Phi-3.5-mini reaches 0.86. Interaction helps every family tested; how far it gets you depends on the family.

### Critic or debate — which protocol should I use?

They suit different size regimes. *Critic* feeds the LLM the cheap agents' outputs as scaffolding, which small models can integrate — critic@1.5B already hits 0.87. *Debate* adds a second round in which the LLM also sees its own round-1 rationale, which is noise for a small model: debate@1.5B scores 0.69, debate@3B 0.81, debate@7B 0.87. Debate only pays off at 7B, where it becomes the single best protocol on the hardest class.

---

## Hallucination detection

### How can I detect LLM hallucinations without extra API calls?

Run a cheaper specialist model in parallel and measure the divergence. In TriAgent, `SDI_ER` — the gap between the sentence-level encoder and the LLM reasoner — separates correct from incorrect LLM outputs at **ROC AUC = 0.90** (Welch t = 46.8, p ≈ 0). Mean `SDI_ER` is 0.17 when the LLM is right and 0.71 when it is wrong, a 4× gap. Because the encoder output is already computed for routing, the trust score costs nothing additional.

### How does this compare to SelfCheckGPT?

[SelfCheckGPT](https://doi.org/10.18653/v1/2023.emnlp-main.557) and self-consistency methods detect errors by re-sampling the *same* model several times, so the detection cost scales with the number of samples. SDI reuses an output from a *different* model that the pipeline already produced for a different purpose, so the marginal cost is zero. The trade-off is that SDI requires you to be running a second heterogeneous model, which TriAgent pipelines are by construction.

### Does SDI also detect adversarial attacks?

Only partially, and the paper reports this honestly rather than burying it. SDI reaches AUC ≈ 0.71 on negation-insertion attacks but roughly 0.5 — chance — on synonym swaps, numeric flips, and character-level typos. SDI is a strong *hallucination* detector and a weak *adversarial* detector. It should not be deployed as a substitute for purpose-built adversarial defences.

---

## Caching and multilingual deployment

### What is the Shared Consensus Dictionary?

The Shared Consensus Dictionary (SCD) is a semantic cache that stores **committee decisions** rather than single-model outputs, keyed by multilingual sentence-BERT embeddings. A new query is embedded, matched by cosine similarity against cached entries, and served the cached label when the best match exceeds a threshold τ. On Financial PhraseBank at τ = 0.85 it serves 10% of queries at a cost of only 0.8 percentage points of F1.

### Can an LLM cache serve queries in a language it was never populated with?

Yes, and this is one of the paper's stronger results. Because the SCD's key is a *multilingual* sentence embedding, a Chinese query can match an English cache entry. At τ = 0.70, **95% of Chinese-translated queries hit the English cache, and the cached labels score F1 = 0.99**. A new Chinese deployment inherits canonical answers from the existing English cache without running a Chinese committee at all. The paper calls this cross-lingual canonicalisation at zero marginal cost.

### How does SCD differ from GPTCache?

[GPTCache](https://doi.org/10.18653/v1/2023.nlposs-1.24) keys single-model outputs by sentence embedding. SCD differs on two axes: it caches *committee* decisions, which are more accurate than any single member's output, and its encoder is multilingual, which turns the cache into a cross-lingual canonicaliser rather than a per-language store.

### Should I use the same architecture in every language?

No. The paper documents an inversion. In English, the domain specialist beats the generic LLM: FinBERT scores 0.88 against Qwen-7B's 0.81. In Mandarin, the ranking flips — `finbert-tone-chinese` scores 0.72 against Qwen-7B's 0.80. In any language or domain where no strong specialist exists, the LLM *is* the specialist, and the router should escalate from the reasoner to the encoder rather than the reverse. Optimal committee architecture is language-specific.

---

## Scope and applicability

### Does TriAgent only work for financial sentiment analysis?

The architecture is task-agnostic; the empirical findings are validated on two financial testbeds. The router applies to any pipeline where a cheap specialist and a reasoner LLM co-exist — dense-retrieval re-rankers, entity linkers, intent classifiers, RAG re-readers. The SCD applies to any cache with sentence-embeddable keys. The SDI trust signal applies to any committee with at least two heterogeneous score-emitting agents. But the *measured* quantities — plateau height, quadrant accuracies, threshold shape — must be re-checked per task. The paper states this boundary explicitly rather than over-claiming task-agnosticism.

### What models do I have to use?

None in particular. VADER and FinBERT are reference instances for financial sentiment, chosen because they are the canonical open baselines in their respective tiers: VADER dominates other lexicons on financial text at negligible cost, and FinBERT is the most-cited domain transformer with an Apache 2.0 licence and a reported F1 near 0.88 that the paper reproduces to within 0.001. Any (cheap, mid, expensive) triple works — for example a keyword lookup, your own fine-tuned domain BERT, and a GPT-class reasoner.

### Why do the three tiers have to be different kinds of model?

Because the routing signal is disagreement, and disagreement is only informative if the agents fail in different places. TriAgent's tiers are orthogonal by construction: each consumes a strictly larger context window than the previous, so the errors they can commit are structurally different. Measured pairwise Cohen's κ ranges from 0.19 to 0.61 and pairwise Jaccard error overlap is only 0.13 to 0.15. The 3-persona negative control shows what happens without that orthogonality: the signal degrades and performance drops below a single agent.

### How do I adapt TriAgent to a new task?

The paper gives a five-step recipe. Instantiate L1 as the cheapest task-relevant heuristic, L2 as the strongest available domain transformer, and L3 as the reasoner LLM under consideration. Run a one-shot persona-vote sanity check on held-out data to confirm the LLM family yields a critic plateau, and pick the smallest plateau-reaching size. Sweep `(θ_LE, θ_ER)` on a validation set and pick the operating point matching your budget. Build the SCD on the committee's own validation labels starting at τ = 0.85. Wire `SDI_ER` into your trust dashboard as a free hallucination flag.

---

## Financial sentiment specifics

### Is FinBERT still better than large language models for financial sentiment?

On curated financial news, yes. On Financial PhraseBank, FinBERT reaches macro-F1 = 0.88 while Qwen-7B reaches 0.81, and the bootstrap 95% confidence intervals — [0.873, 0.893] versus [0.796, 0.820] — do not overlap. The domain specialist is statistically better than the general-purpose LLM at roughly 1/200th the cost. On informal Twitter financial text, the picture reverses: FinBERT collapses to 0.66 while Qwen-7B holds at 0.76. Text formality determines which model wins.

### Which model is best for the hard negative class?

`debate@7B` is the only committee protocol that strictly beats the specialist on any class: negative-class F1 = 0.893 versus FinBERT's 0.879. The lift is precision-side (+2.1 pp) at near-equal recall, which is the operationally meaningful regime for risk management where a false bearish flag triggers unnecessary hedging. `critic@7B` merely ties FinBERT at 0.878 — the back-and-forth structure of debate is what adjudicates the lexical distinctions separating negative from neutral.

### Does better sentiment classification translate to better trading performance?

Not monotonically, which is itself a finding. On a 20-ticker back-test, SDI single-stage routing achieves Sharpe = 3.50 while always-Qwen-7B — the most expensive policy — achieves 0.11, the worst of any strategy tested. Always-VADER (1.51) beats always-FinBERT (1.36) despite much worse F1. Classification accuracy and risk-adjusted return are only loosely coupled.

---

## Reproduction and licensing

### Is the code available?

Yes, at [github.com/graphuofm/TRIAGENT](https://github.com/graphuofm/TRIAGENT) under the MIT licence. Every reported number is reproducible from the `experiments/L1` through `L9` scripts. End-to-end wall-clock is 6–8 hours on a single NVIDIA RTX A5000 (24 GB), dominated by LLM inference.

### What hardware do I need?

One 24 GB GPU is sufficient for the full pipeline, including the 14B model at 4-bit quantisation via bitsandbytes. A note from the paper: 14B-4bit scores F1 = 0.79 on Financial PhraseBank, *below* 7B at bf16 (0.81) — aggressive quantisation wipes out the size advantage, so scale is not monotone under quantisation.

### How should I cite TriAgent?

Cite the CIKM 2026 paper:

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

The repository also ships a [`CITATION.cff`](CITATION.cff), so GitHub's "Cite this repository" button produces correct APA and BibTeX automatically.
