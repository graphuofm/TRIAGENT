# Security & Bias as a First-Class Contribution

This document expands on contribution #6 from `docs/10_AGENTIC_PIVOT.md`.

## The argument in three sentences

Single large LLMs are a single point of failure: one compromised model, one
prompt injection, or one bias mode can corrupt every downstream decision.
A committee of small isolated agents removes this concentration — each agent
runs in its own process / sandbox / VM, and the same SDI signal we use for
cost-routing **also detects when one agent has been tampered with or
hallucinates**. Heterogeneous small models additionally have orthogonal
failure modes, so majority-style voting suppresses individual-agent bias.

## Why this only works at small scale

Sandboxing one GPT-4-class model already saturates a GPU. You cannot
realistically run five isolated copies, run them on different hardware
substrates, or restart any one of them on attack. Small agents (≤3B) can
each live in a separate Docker container / VM / serverless invocation
**without breaking the deployment budget**, and that's what makes per-agent
attack containment feasible.

This is the security half of the same coin as cost: the cost-routing
contribution and the security contribution are **not independent claims** —
they're two consequences of the same architectural decision (small isolated
agents instead of one big shared one). Stress this in §6.

## Concrete experiments to add (1-2 days)

### E1: SDI as adversarial-perturbation detector

For a held-out subset of FPB (say 500 sentences):

1. Apply mild text perturbations:
   - Synonym swap on key sentiment words (using WordNet)
   - Negation insertion ("X grew" → "X did not grow")
   - Numeric perturbation ("profit rose 10%" → "profit rose 0.1%")
   - Adversarial typos / character drops
2. Run the committee on both clean and perturbed versions
3. Measure SDI shift: ΔSDI = SDI(perturbed) − SDI(clean)
4. Report:
   - Mean ΔSDI per perturbation type
   - AUC of "SDI shift > τ" as a binary detector for "is this perturbed"
   - Headline: **"SDI catches X% of adversarial perturbations at 5% FPR"**

### E2: SDI as hallucination detector

Use the LLM-vs-FinBERT subcase: when LLM gives a wrong label, how often does
FinBERT disagree?

1. Stratify the FPB committee data by `llm_correct ∈ {True, False}`
2. Compute mean and distribution of SDI_ER (FinBERT-LLM divergence) per stratum
3. Report ROC of using SDI_ER > τ as a "the LLM is hallucinating" signal
4. Headline: **"When the LLM is wrong, FinBERT disagrees X% more than when
   the LLM is right (p < 0.001, AUC = Y)"**

### E3: Bias diversity quantification

For each agent (VADER, FinBERT, Qwen-{0.5B, 1.5B, 3B, 7B}):

1. Compute per-class confusion (where do the errors land?)
2. Compute pairwise overlap of error sets (intersection / union)
3. Show that error sets are mostly disjoint — different agents fail on
   different sentences
4. Quantify bias-cancellation effect: how much does ensembling reduce
   per-class FPR/FNR vs the worst single agent?
5. Headline: **"Heterogeneous agents have only X% error overlap; majority
   vote reduces minority-class FNR by Y% over the best single agent"**

### E4: Per-agent isolation cost

Mostly a deployment table, not a research experiment:

| Agent | RAM | Cold-start time | Container size | Independent restart? |
|---|---|---|---|---|
| VADER | 50 MB | <0.1s | 10 MB | ✓ |
| FinBERT | 500 MB | 2s | 500 MB | ✓ |
| Qwen-0.5B | 1.5 GB | 5s | 1 GB | ✓ |
| Qwen-7B | 16 GB | 30s | 15 GB | ✓ |
| (hypothetical GPT-4 self-hosted) | 200+ GB | minutes | 200+ GB | ✗ in practice |

This table goes in §6 as the deployment-feasibility argument for the
isolation claim.

## Paper integration

Add a new subsection §5.6 "Security & Bias Diversity" (~0.5 page) covering
E1+E2+E3 results. E4 lives in §6 (Discussion).

The Introduction should now have **three** motivations, not two:

1. Cost: agentic financial systems waste compute by always using the largest
   model
2. Capability: smaller models can recover lost accuracy via interaction
3. **Security & bias**: small isolated agents enable defense-in-depth that
   monolithic large models cannot

Make it explicit that these three follow from the same architectural choice.
That's the elegance of the framing — one design decision, three benefits.

## Falsifiable hypotheses (commit upfront)

H4: SDI rises significantly under adversarial perturbations vs clean text
    (predicted: ΔSDI Cohen's d > 0.5 for synonym/negation/numeric
    perturbations on FPB).

H5: SDI_ER (FinBERT vs LLM) is significantly higher when LLM is wrong than
    when LLM is right (predicted: AUC > 0.65 as a hallucination detector).

H6: Heterogeneous-agent error sets have <50% overlap on the negative class
    (the hardest class for VADER), so committee voting strictly improves
    minority-class recall.

If any of these are negative, the security section becomes a "limitations
of the SDI signal" subsection rather than a contribution. Either way the
paper is still publishable.
