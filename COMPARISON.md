# TriAgent vs. Related Systems

How TriAgent ([CIKM 2026](https://doi.org/10.1145/3799682.3839978), [arXiv:2607.19794](https://arxiv.org/abs/2607.19794)) relates to other approaches for cutting LLM inference cost, caching LLM responses, and detecting LLM errors.

---

## At a glance

| System | Category | Routing signal | Training data needed | Caching | Cross-lingual | Trust signal | Marginal cost of the signal |
|---|---|---|---|---|---|---|---|
| **TriAgent** | Router + cache + detector | Inter-tier divergence (SDI) | **None** | Committee decisions | **Yes** — one cache serves all languages | **Yes**, AUC 0.90 | **Zero** |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) | Cascade router | Learned scorer confidence | Labelled examples | No | No | No | Scorer inference |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | Router | Learned preference model | Human preference pairs | No | No | No | Router inference |
| [LLM routing with benchmark datasets](https://arxiv.org/abs/2309.15789) | Router | Benchmark-derived model profile | Benchmark results | No | No | No | Profile lookup |
| [GPTCache](https://doi.org/10.18653/v1/2023.nlposs-1.24) | Semantic cache | — | None | Single-model outputs | No | No | Embedding |
| [SelfCheckGPT](https://doi.org/10.18653/v1/2023.emnlp-main.557) | Hallucination detector | — | None | No | No | Yes | *n* extra LLM samples |
| [Self-consistency](https://arxiv.org/abs/2203.11171) | Decoding strategy | — | None | No | No | Implicit | *n* extra LLM samples |
| [Multiagent debate](https://arxiv.org/abs/2305.14325) | Accuracy method | — | None | No | No | No | 2+ LLM calls per query |
| [MetaGPT](https://arxiv.org/abs/2308.00352) / [AutoGen](https://arxiv.org/abs/2308.08155) | Agent framework | — | None | No | No | No | Many LLM calls |

---

## Against learned routers — FrugalGPT and RouteLLM

**What they share with TriAgent.** All three accept that sending every query to the strongest model is wasteful, and all three try to identify the queries where a cheaper model suffices.

**Where they differ.** FrugalGPT trains a scorer on labelled examples; RouteLLM trains a router on human preference data. Both therefore carry a supervision cost: you must collect routing labels for your task, and you must retrain when you swap a model in or out. TriAgent's routing feature is arithmetic — the absolute difference between two agents' scores — so there is no router to train, no labelled routing set to collect, and no retraining when the model lineup changes.

**Where the difference shows up empirically.** A single-model confidence score fails in a specific, measurable way: the model can be confidently wrong. TriAgent's four-quadrant analysis quantifies it. In the *ambiguous* quadrant — where the two cheap tiers agree but the LLM dissents, 15.9% of Financial PhraseBank — the LLM's accuracy is **28%**, below chance for a three-way task. A confidence-threshold cascade escalates exactly here, because the LLM is confident. TriAgent's rule is the opposite: **when the cheap pair agrees, trust the pair and skip the LLM, even when the LLM separately dissents.** That case is invisible to any signal derived from one model.

**The honest trade-off.** A learned router can in principle exploit task structure that a fixed arithmetic rule cannot. TriAgent buys its zero-supervision property by giving that up. It also requires running two cheap models per query rather than one scorer — cheap, but not free.

---

## Against semantic caches — GPTCache

**What they share.** Both avoid recomputation by serving semantically similar queries from an embedding-keyed store, rather than requiring an exact string match.

**Two differences.** First, what gets stored: GPTCache caches single-model outputs, while the Shared Consensus Dictionary caches **committee decisions**, which are more accurate than any individual member. Second, the encoder: because the SCD keys on a *multilingual* sentence embedding, it functions as a cross-lingual canonicaliser rather than a per-language store.

**The result that follows from the second difference.** At similarity threshold τ = 0.70, **95% of Chinese-translated queries hit the English cache, and the cached labels score F1 = 0.99**. A new Chinese deployment inherits canonical answers from the existing English cache without running a Chinese committee at all. A per-language cache cannot do this; it must be populated once per language.

| τ | Chinese hit rate on the English cache | F1 |
|---|---|---|
| 0.50 | 100% | 0.983 |
| **0.70** | **94.5%** | **0.987** |
| 0.85 | 54.9% | 1.000 |

---

## Against hallucination detectors — SelfCheckGPT and self-consistency

**What they share.** All three detect likely LLM errors without ground-truth labels at inference time.

**Where they differ.** SelfCheckGPT and self-consistency re-sample the *same* model several times and measure agreement among samples, so detection cost scales linearly with the number of samples. TriAgent's `SDI_ER` compares the LLM against a *different* model that the pipeline already ran for routing, so the marginal cost is zero.

**Detection quality.** `SDI_ER` separates correct from incorrect LLM outputs at **AUC = 0.898** (Welch t = 46.8, p ≈ 6e−255). Mean divergence is 0.168 when the LLM is right and 0.709 when it is wrong.

**A result that pins down the mechanism.** Not every pairing works — only the cross-granularity one:

| SDI variant | Pairing | AUC |
|---|---|---|
| `SDI_LE` | lexicon vs. encoder | 0.620 |
| `SDI_LR` | lexicon vs. reasoner | 0.575 |
| **`SDI_ER`** | **encoder vs. reasoner** | **0.898** |

Disagreement in general is a weak signal. Disagreement *across granularities* is a strong one.

**The trade-off.** SDI requires you to already be running a second, heterogeneous model. SelfCheckGPT works on a single model in isolation. If you are not running a committee, SelfCheckGPT is applicable where SDI is not.

---

## Against multi-agent LLM systems — debate, MetaGPT, AutoGen

**What they share.** Both use multiple agents and inter-agent interaction to improve on a single model's output.

**Where they differ, and why it matters.** Debate frameworks and agent orchestration systems typically instantiate agents as *different prompts of the same model*. TriAgent stratifies agents by *model class and context window*. The paper includes a negative control isolating exactly this variable.

Three personas of the same Qwen2.5-1.5B — bull, bear, neutral — voting together:

| Configuration | Macro-F1 |
|---|---|
| persona-bull | 0.74 |
| persona-bear | 0.68 |
| persona-neutral | 0.55 |
| **3-persona vote** | **0.66** |
| single Qwen-1.5B baseline | 0.69 |
| **cross-tier critic@1.5B** | **0.87** |

The homogeneous vote scores *below a single instance of the same model*. Same parameter count, same family, three agents, worse result. What produces the +18-point lift is **granularity-stratified diversity**, not multi-agent interaction as such.

**Cost.** Debate and agent frameworks multiply LLM calls. TriAgent fires an interaction protocol on only the fraction of queries where divergence is high — 1.5% at the Balanced operating point.

---

## Against always-using-the-big-model

The baseline most production systems actually run.

| Metric | Always-GPT-4 | Always-GPT-4o-mini | Always-Qwen-7B | **TriAgent** |
|---|---|---|---|---|
| Annual cost @ 10M users | $365M | $11.0M | $1.05M | **$1.65M** |
| Macro-F1 (FPB) | — | — | 0.809 | **0.88** |
| Back-test Sharpe | — | — | 0.112 | **3.499** |

Two findings that complicate the "just use the best model" intuition. First, Qwen-7B alone (F1 = 0.809) is **statistically worse** than FinBERT alone (0.883) on Financial PhraseBank — non-overlapping bootstrap CIs — at roughly 200× the cost, which puts the always-LLM policy off the Pareto front entirely. Second, on the 20-ticker back-test, always-LLM is the *worst* strategy tested on a risk-adjusted basis (Sharpe 0.112 vs. 3.499).

---

## Where TriAgent is the wrong tool

Stated plainly, because knowing the boundary is part of the contribution.

**Your mid-tier specialist is weak or miscalibrated.** This is the documented failure mode. On TFNS, where FinBERT collapses to F1 = 0.66, the critic protocol reaches only 0.65 — *below* the 0.76 the LLM achieves alone. The scaffolding hurts. Run the one-shot critic-versus-LLM-alone check before committing.

**You are defending against adversarial input.** SDI reaches AUC ≈ 0.71 on negation insertion and ≈ 0.50 — chance — on synonym swaps, numeric flips, and typos. It is a hallucination detector, not an adversarial one.

**You run exactly one model.** The whole approach is built on cross-model divergence. With one model there is nothing to diverge from; use self-consistency or SelfCheckGPT instead.

**Your task has no cheap specialist.** In Mandarin, `finbert-tone-chinese` (0.72) *loses* to Qwen-7B (0.80). Where no strong specialist exists, the LLM is the specialist and the routing hierarchy should invert.

**You need generation, not classification or scoring.** SDI is defined over comparable scalar scores. Extending it to free-form generation — for example by comparing outputs in embedding space — is future work, not a validated result.

---

## Summary

TriAgent's distinguishing claim is not that it routes, caches, or detects hallucinations better than every specialised system at each of those tasks individually. It is that **all three fall out of a single divergence signal computed from outputs the pipeline already has**, with no learned router and no labelled routing data — and that the signal only works when the agents are stratified by contextual granularity, which the 3-persona negative control demonstrates by removing that property and watching performance drop below a single agent.
