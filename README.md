# TriAgent

**Divergence-aware multi-agent committee for financial sentiment analysis.**

Companion code for our FinLLM @ IJCAI 2026 workshop submission.

📄 **Paper:** [`triagent_paper_compiled.pdf`](triagent_paper_compiled.pdf) · arXiv preprint coming soon
📦 **Overleaf bundle:** [`triagent_overleaf.zip`](triagent_overleaf.zip) (drop into Overleaf to recompile)

---

## What's in the paper

A three-tier sentiment committee stratified by **contextual granularity**:

| Tier | Agent | Granularity | Speed |
|------|-------|-------------|-------|
| L1 | VADER | word-level lexicon | ~0.03 ms |
| L2 | FinBERT | sentence-level transformer | ~1.5 ms |
| L3 | Qwen2.5 / Mistral / Phi-3.5-mini | cross-sentence reasoner | hundreds of ms |

A three-way **Semantic Divergence Index (SDI)** measures pairwise disagreement
between tiers and routes each query accordingly.

**Headline findings:**
- **Critic plateau:** critic@1.5B = critic@3B = critic@7B ≈ F1=0.87 (bootstrap CIs overlap).
  Same-size 3-persona vote drops to F1=0.66 — *granularity-stratified diversity*, not multi-agent voting per se, drives the plateau.
- **Shared Consensus Dictionary (SCD):** 95% of Chinese queries find their English cached answer at F1=0.99 — cross-border canonicalization for free.
- **SDI as hallucination detector:** AUC=0.90 on FPB with zero extra models or labels.
- **20-ticker back-test:** SDI single-stage Sharpe=3.50 (best of any strategy).
- **At 10M-user scale:** TriAgent saves ~$363M/yr vs. always-GPT-4.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY only if you want the GPT-4o-mini sanity check
```

Tested on Python 3.10 + PyTorch CUDA 12.4, one NVIDIA RTX A5000 (24 GB).

## Reproducing the paper end-to-end

The pipeline is split into **layers** (each is a stand-alone `experiments/L*.py`
script). Run them in order; each consumes the previous layer's CSV.

```bash
# L1 — three-agent data collection on FPB
python experiments/L1_data_collection.py --dataset fpb --yes

# L1.5 — multi-size LLM sweep (Qwen 0.5B/1.5B/3B/7B/14B-4bit, Mistral-7B)
for SIZE in 0.5B 1.5B 3B 14B Mistral-7B; do
    python experiments/L1p5_size_sweep.py --sizes "$SIZE"
done

# L2 — SDI + four-quadrant decomposition
python experiments/L2_sdi_analysis.py

# L2.5 — interaction protocols (vote / critic / debate, swept over sizes)
for SIZE in 1.5B 3B 7B Mistral-7B; do
    python experiments/L2p5_interaction.py --protocols critic --llm-size "$SIZE"
    python experiments/L2p5_interaction.py --protocols debate --llm-size "$SIZE"
done

# L3 — cost-Pareto routing sweep + operating points
python experiments/L3_routing_pareto.py
python experiments/L3p5_scaling.py

# L4 — three-granularity edge predictor (XGBoost AUC=0.85)
python experiments/L4_predictor.py --with-reasoning

# L5 — 20-ticker back-test
python experiments/L5_backtest.py

# L5.5 — security side-experiments (SDI as hallucination detector, adversarial probe)
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

# L9 — same-size persona vote (negative control)
python experiments/L9_same_size_multiagent.py

# Render every paper figure from results/data/*.csv
for s in paper/figures/code/make_*.py; do python "$s"; done
```

End-to-end wall-clock on a single A5000: ~6–8 h (LLM inference dominates).
The full pipeline spec is in [`paper/PIPELINE_README.md`](paper/PIPELINE_README.md).

## Repository layout

```
paper/                LaTeX source (main.tex, sections/, tables/, figures/, references.bib)
  figures/code/       matplotlib scripts that regenerate every figure from results/data/
  PIPELINE_README.md  end-to-end pipeline spec
src/                  agents, SDI, routing, SCD, predictor, backtest modules
experiments/          L1 … L9 reproducible per-layer scripts
data/                 raw datasets and mined trigger lexicons
results/              figures, tables, summary CSVs
docs/                 layer-by-layer design specs
triagent_paper_compiled.pdf   compiled paper (latest build)
triagent_overleaf.zip         Overleaf-ready LaTeX bundle
```

## Cite

Bibtex entry will be added once the arXiv preprint is live.

## License

[MIT](LICENSE). Models and datasets we depend on retain their own licenses:
VADER (MIT), FinBERT (Apache 2.0), Qwen2.5 (Apache 2.0), Mistral-7B (Apache 2.0),
Phi-3.5-mini (Apache 2.0), Financial PhraseBank (CC-BY-NC-SA),
TFNS (research use).
