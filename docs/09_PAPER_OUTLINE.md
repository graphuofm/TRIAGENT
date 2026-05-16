# Paper Outline (7 pages content + 2 pages refs)

> **Updated 2026-04-30** — pivoted from "cost-routing committees" to
> "minimum-viable agentic committees with disagreement-triggered interaction".
> See `docs/10_AGENTIC_PIVOT.md`. The original outline below is preserved
> for reference; substantial sections (esp. §1, §3, §5) are rewritten.

## Title (working candidates, updated)

1. **"How Small Can Agents Go? Minimum-Viable Multi-Agent Committees for
   Financial Sentiment in Customer-Facing Deployments"** ← current
2. **"Disagreement-Triggered Multi-Agent Interaction for Cost-Efficient
   Financial Sentiment Classification"**
3. **"Trading Parameters for Interactions: Edge-Friendly Agentic Committees
   for Production Financial NLP"**

(1) hits the FinLLM CFP themes most directly: agentic, scaling-down,
deployment, customer-facing.

## Abstract (~200 words)

Lead with the deployment pain (cost). Introduce SDI committee. Give headline 
numbers. Close with deployability claim.

Draft skeleton:

> Production deployment of LLM-based financial sentiment analysis faces a 
> structural challenge: 80% of queries are trivially classifiable but are 
> routinely processed by expensive reasoning models, incurring API costs that 
> can reach $30,000/month for a single asset manager. We propose **TriAgent**, 
> a three-tier committee combining a lexicon agent (VADER), a domain-specific 
> transformer (FinBERT), and a reasoning LLM (GPT-4o-mini), orchestrated by 
> a **Semantic Divergence Index (SDI)** that routes each query to the 
> minimum-cost tier yielding correct judgment. We make three contributions: 
> (1) a three-way SDI decomposition exposing four committee-behavior 
> quadrants, (2) a Pareto analysis showing SDI-routing achieves 95% of 
> GPT-4o-mini accuracy at 15% of the cost, and (3) an edge-side divergence 
> predictor leveraging unigram, bigram, and trigram trigger features 
> (AUC=0.85), enabling fully-deployable cloud-edge collaboration. End-to-end 
> backtests across 20 stocks demonstrate that SDI-routed committees achieve 
> the highest **excess return per inference dollar** among all strategies. 
> Code available at: [github link].

## §1 Introduction (1 page, ~6 paragraphs)

**Para 1 — The deployment scenario**:
- Asset managers process 100K+ queries/day through LLM agents
- Concrete dollar figure: $30K/month for GPT-4-class models
- Industry shift: agentic finance (mention OpenClaw, OPC paradigm to hit CFP)

**Para 2 — The structural waste**:
- 80% of queries are trivially classifiable (cost data + sanity numbers)
- Current "all-or-nothing" deployment: pay too much OR perform too poorly
- Empirical motivation: 60% of FPB sentences are correctly classified by VADER

**Para 3 — Our proposal**:
- Three-tier committee (L1/L2/L3) with SDI routing
- Each query gets routed to the cheapest tier that handles it correctly
- Headline preview: "We achieve X% accuracy at Y% cost vs. Always-L3"

**Para 4 — Contributions** (numbered, concrete):
1. Three-way SDI decomposition with four-quadrant interpretation
2. Token-Economic Pareto analysis across 7 routing strategies (→ main figure)
3. Edge-side divergence predictor with bigram/trigram trigger features
4. End-to-end financial backtest with cost-adjusted return metric

**Para 5 — Why now / why this matters**:
- Reference 2026 CFP themes: model collaboration, token economics, bias diversity
- Addresses real deployment friction (E Fund-style audience cue)

**Para 6 — Roadmap**:
- Standard 2-sentence "the paper is organized as follows" closer

## §2 Related Work (0.5 page)

Three sub-paragraphs:

**Domain challenges in financial NLP** — Loughran & McDonald 2011 (negative-
word miscalibration); FinBERT (Araci 2019) as the de facto domain model; 
recent FinLLM survey work (cite something from 2024-2025 if available).

**Multi-agent and committee systems** — query-by-committee (Settles 2009 
active learning); ensemble disagreement as uncertainty (Lakshminarayanan et 
al. 2017); recent multi-agent LLM frameworks (cite MENTOR, FinSphere from 
FinLLM 2025).

**Cost-aware inference and routing** — model cascading literature 
(Wang et al. mixture of cheap and expensive models); LLM routing recent work 
(RouteLLM, FrugalGPT). Position our work as the first to apply this lens 
specifically to financial sentiment with three tiers and SDI.

## §3 Framework (1.5 pages)

### §3.1 Three-Tier Committee Architecture (0.5 page)

System diagram (Figure 1) showing:
- L1: VADER (edge, $0)
- L2: FinBERT (specialist, ~$0)
- L3: GPT-4o-mini (cloud, $$$)
- Routing decision module in the middle

Table summarizing latency, cost, and use case for each tier.

### §3.2 Semantic Divergence Index — Three-Way Decomposition (0.5 page)

Define:
- SDI_LE = |s_vader − s_finbert| (Lexicon-Expert)
- SDI_LR = |s_vader − s_llm| (Lexicon-Reasoner)
- SDI_ER = |s_finbert − s_llm| (Expert-Reasoner)

Four-quadrant interpretation (referenced from L2 figures):
- Consensus / Domain Shift / Ambiguous / Mixed

Brief justification for why three-way > two-way (allows distinguishing 
"easy domain shift" from "genuinely hard").

### §3.3 Routing Strategies (0.3 page)

Catalog S0–S6 in a compact table. Special focus on S6 (two-stage) with 
pseudocode.

### §3.4 Theoretical Note (0.3 page) — minimal, just satisfies academic reviewers

One paragraph + one lemma:

> **Lemma 1**: For categorical-output agents, SDI is monotonically related 
> to the Jensen-Shannon divergence between agent output distributions.
> **Proof sketch**: [2-3 sentences in appendix]

That's it. Don't over-invest here.

## §4 Divergence Mining and Edge-Side Prediction (1 page)

### §4.1 Log-Odds Trigger Mining — Unigram + Bigram + Trigram (0.5 page)

Method description + Table 2 of top triggers per n-level.

Headline: "While unigrams like `loss` and `profit` exhibit context-
dependent polarity, bigrams like `loss_narrowed` (positive) and 
`profit_declined` (negative) resolve the ambiguity directly."

### §4.2 Edge-Side Predictor (0.5 page)

Feature description + Table 3 (the ablation: random / LR-uni / LR-uni+bi / 
LR-uni+bi+tri / XGBoost).

Highlight: "Adding bigram features lifts AUC by ~7 points, validating 
that phrase-level patterns are the natural unit of financial sentiment."

## §5 Experiments (2 pages — the meat of the paper)

### §5.1 Setup (0.2 page)

Dataset, models, hardware, code availability.

### §5.2 Baseline Performance and Bias Diversity (0.4 page)

- Table 1: single-model baselines (Acc, F1, Latency, Cost)
- Figure 2: Bias diversity (kappa heatmap + entropy distribution)
- Brief commentary on negative-class systematic mis-classification

### §5.3 Token-Economic Pareto Frontier — ⭐ MAIN RESULT (0.5 page)

- Figure 3: the Pareto plot ⭐ (THIS IS THE FIGURE)
- Table 4: three operating points (Budget / Balanced / Premium)
- Headline numbers in text: "SDI-Two-Stage Balanced achieves 95% of 
  Always-L3's F1 at 15% of the cost"

### §5.4 Edge-Side Predictor (0.4 page)

- Figure 4: ROC curves + feature importance
- Table 3 (referenced from §4) — already covered, just point to it
- Brief: predictor-routing reaches 87% of oracle-SDI-routing's F1

### §5.5 End-to-End Trading Performance (0.5 page)

- Figure 5: equity curves
- Table 5: Sharpe / MaxDD / **Excess Return per Inference Dollar**
- Headline: SDI-Two-Stage has **highest** excess return per $

## §6 Discussion and Limitations (0.5 page)

Six bullets:
1. Deployment considerations (latency, model versioning, compliance)
2. FPB-as-news-proxy limitation in L5
3. Generalization to other languages — point to §7 if included
4. Robustness across regimes (2-year backtest is short)
5. Future work: real-time news streams, OPC-style autonomous agents
6. Open release: code + lexicon

## §7 (Optional) Cross-Lingual Pilot (0.3 page)

Brief case study on 150 Chinese financial sentences. Show SDI generalizes. 
Don't make grand claims.

## §8 Conclusion (0.3 page)

Summarize:
- SDI routing + edge predictor = deployable, cost-efficient committee
- 95% F1 retention at 15% cost
- Open code

End with broader vision (agentic finance, OPC paradigm, "every Token Should 
Earn Its Place"-style closer).

## Figure & Table Budget

- Figure 1: System architecture
- Figure 2: Bias diversity (kappa + entropy)
- Figure 3: ⭐ Pareto frontier (the main figure)
- Figure 4: Predictor performance (ROC + feature importance)
- Figure 5: Equity curves
- (Optional) Figure 6: Cross-lingual

- Table 1: Single-model baselines
- Table 2: Top n-gram triggers
- Table 3: Predictor ablation
- Table 4: Operating points (Pareto)
- Table 5: Backtest summary

5 figures + 5 tables in 7 pages — tight but standard for IJCAI workshops.

## Style Notes (E Fund-friendly writing)

When drafting any section, **prime your writing tool with**:

> "This is for a finance-industry-led workshop. Lead with concrete numbers 
> and dollar amounts, not abstract novelty. Examples of good phrasing: 
> 'asset managers deploying...', 'production deployments incur...', 
> 'inference cost as a first-order operational concern'. Avoid: 'we propose 
> a novel framework that...', 'state-of-the-art performance...', overly-
> academic jargon."

Reviewers from E Fund will reward:
- Specific deployment scenarios
- Concrete dollar figures
- Pareto-frontier-style cost analyses
- Honest limitations

Reviewers from Qiang Yang's academic side will additionally reward:
- The lemma in §3.4
- Clean ablations in §4.2 and §5
- Statistical rigor (CIs, p-values for the main claims)

Both audiences will reject:
- Overclaiming ("state of the art")
- Hidden synthetic data (e.g., synthetic backtest signals — DON'T do this)
- Unfounded business jargon
