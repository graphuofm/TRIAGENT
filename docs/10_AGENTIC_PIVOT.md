# Agentic Pivot: Minimum-Viable Multi-Agent Committees

**Status**: This doc supersedes parts of the original BRIEF for the new
direction. Original cost-routing material (docs 04-09) still applies; this
doc adds the agentic axis on top.

## What changed

The project pivots from **"cost-aware routing through a fixed 3-tier hierarchy"**
to **"minimum-viable agentic committees with disagreement-triggered interaction"**.

Two new axes are introduced:

1. **Model-size axis** — sweep the LLM stage across {0.5B, 1.5B, 3B, 7B, 14B-4bit}
   to locate the inflection point where a model becomes a useful agentic
   committee member.
2. **Interaction axis** — when committee members disagree (high SDI), trigger
   a structured interaction protocol (vote / critic / debate) instead of
   simply escalating to the most expensive model.

The headline question becomes:

> **Can N small models interacting outperform a single large model at the same
> total compute? If so, where is the threshold?**

If the answer is **yes**, we have a genuinely new agentic finding that bridges
"scaling laws" and "multi-agent systems". If the answer is **no**, the cost-
routing story still holds and the model-size sweep becomes a robustness
ablation.

## Updated paper title (working)

**"How Small Can Agents Go? Minimum-Viable Multi-Agent Committees for
Financial Sentiment"**

Alternates:
- "Disagreement-Triggered Interaction for Cost-Efficient Financial Agents"
- "Trading Parameters for Interactions: A Scaling Study of Financial NLP Committees"

## Updated contributions (covering all 3 CFP topic buckets)

1. **Three-way SDI decomposition + four-quadrant interpretation** (unchanged)
2. **NEW — Model-size scaling study**: F1 vs LLM size from 0.5B to 14B,
   identify the inflection point where the model becomes useful in a committee
3. **NEW — Disagreement-triggered interaction protocols**:
   `vote`, `critic`, `debate` — show that small-model interaction substitutes
   for parameter count
4. **Cost-Pareto frontier with size dimension** (extended from original):
   3D Pareto over (cost, F1, model-size)
5. Edge-side n-gram predictor (load-bearing — see customer-language angle below)
6. **NEW — Security & bias as a side benefit of small isolated agents**:
   - Each small agent is cheap enough to run in a separate sandbox/process,
     making per-agent compromise containment realistic (impossible with one
     large shared model)
   - SDI doubles as an anomaly-detection signal: a tampered or
     prompt-injected agent's output diverges from its peers
   - Heterogeneous small models have orthogonal failure modes ⇒ majority
     vote suppresses single-agent bias

## Three-fold motivation (use this for the §1 narrative)

This frames the work as covering all three CFP topic buckets:

| CFP bucket | Our angle |
|---|---|
| **Architectures, Agentic Workflows & Model Collaboration** | Cloud LLM + edge specialist + lexicon agent, with disagreement-triggered interaction |
| **Efficiency, Token Economics & Optimization** | Pareto frontier; "minimum-viable agent" finding shows you can push smaller than current practice without losing F1 |
| **Security, Trust & Ethics in Autonomous Finance** | Small-model isolation + SDI-based anomaly detection; bias diversity quantified across heterogeneous agents |

## Customer-language framing

The deployment target is **fragmented, conversational customer text**
(chatbot, broker support, robo-advisor inputs) — not formal sentence-level
news. Implications:

- The bigram/trigram trigger work in L4 is load-bearing, not an ablation:
  customer language is phrase-bound, not sentence-bound
- We need at least one short/conversational financial dataset (e.g.
  FiQA Task 1 or Twitter Financial News) alongside FPB to validate
  the "phrase > word" claim on real customer-style text

## Models to sweep (L1.5)

All from the Qwen2.5-Instruct family for clean apples-to-apples scaling:

| Model | Params | bf16 mem | 4bit mem | Approx tok/s on A5000 |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | ~1 GB | ~0.4 GB | ~100 |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | ~1 GB | ~80 |
| Qwen2.5-3B-Instruct | 3B | ~6 GB | ~2 GB | ~60 |
| Qwen2.5-7B-Instruct | 7B | ~14 GB | ~5 GB | ~30 |
| Qwen2.5-14B-Instruct | 14B | ~28 GB ❌ | ~9 GB ✓ | ~15 |

The 14B fits only with 4-bit quantization (bitsandbytes). 7B is the current
default. The 0.5B / 1.5B / 3B sweep is the new addition; cheap to run
(<5 min each on A5000).

## Interaction protocols (L2.5 — NEW)

### `vote` — weighted majority
All three agents vote with confidence-weighted ballots. Final label is the
arg-max. Cheapest possible interaction (just compute, no extra LLM call).

### `critic` — LLM as judge
LLM sees VADER's prediction + softmax + FinBERT's prediction + softmax,
plus the original sentence, and outputs a final classification with rationale.
One extra LLM call per disagreement.

### `debate` — two-round structured exchange
Round 1: Each model outputs label + rationale.
Round 2: LLM is shown all three rationales and asked to either:
  (a) agree with one of them, or (b) issue a tie-breaking new judgment.
Two LLM calls per debate.

Each protocol is **only triggered when SDI > threshold** — measure both
F1 and additional cost-per-disagreement.

## Reframed paper structure

Sections roughly as before; key changes:

| § | Original | New |
|---|---|---|
| §1 Intro | "asset managers waste $30K/mo on LLM API" | "agentic systems use the largest model for everything; we ask how small can each agent get" |
| §3.2 SDI | unchanged | unchanged |
| §3.3 Routing | "S0–S6 routing strategies" | "S0–S6 + I0 (no interaction), I1 (vote), I2 (critic), I3 (debate)" |
| §4 (NEW) | — | Scaling study: Figure showing F1 vs model size |
| §5.3 main result | Pareto figure | 3D Pareto: (cost, F1, model-size) |
| §5.x (NEW) | — | "Interaction substitutes for parameters" — does 3×0.5B interacting beat 1×3B? |

## Critical-path checkpoints (revised)

Before each subsequent layer, verify:

- **L1.5 checkpoint**: F1 should monotonically increase with model size for the
  Qwen family. If 0.5B already hits F1>0.75, the inflection is too low and
  the story weakens. If 7B is still <0.80, the family choice is wrong.

- **L2.5 checkpoint**: At least one interaction protocol (likely `critic` or
  `debate`) should beat plain "Always-L3" on F1 at equal or lower cost. If
  none do, the interaction story dies.

- **The killer figure**: A single chart with model-size on x-axis, F1 on y-axis,
  and **both** "single agent" and "3-agent debate" curves. If the debate
  curve dominates the single curve at small sizes, the paper writes itself.

## Falsifiable hypotheses (good to commit to upfront)

H1: There exists a model size below which agentic interaction collapses.
    (Prediction: ~1B; below this models can't even output valid JSON
    reliably.)

H2: For sizes above the threshold, N×small > 1×large at equal total compute.
    (Prediction: holds for 1.5B–3B range; breaks down at 7B+ where single
    model is already strong.)

H3: SDI-triggered interaction is more cost-efficient than always-on
    interaction.
    (Prediction: holds; ~70% of samples don't need interaction.)

If we end up rejecting H2 entirely, the paper pivots back to "minimum-viable
single agent for financial sentiment" — still publishable, less exciting.

## Implementation order (15-day plan)

See top-level BRIEF for the day-by-day schedule. In the code:

1. `src/agents/llm_agent.py` — already supports `model_name` param, no change
2. `experiments/L1p5_size_sweep.py` — NEW, sweeps Qwen sizes
3. `src/interaction/` — NEW module
   - `vote.py`
   - `critic.py`
   - `debate.py`
4. `experiments/L2p5_interaction.py` — NEW, runs all protocols
5. `src/routing/strategies.py` — extended with I0/I1/I2/I3 strategies
6. Existing L3-L5 unchanged in shape; just plot more curves

## What we keep from the original plan

Everything in `docs/04-09` still applies. The cost-routing Pareto is now one
slice of the larger picture, not the whole story. SDI, edge-predictor,
backtest — all unchanged. We are **adding** axes, not replacing the framework.
