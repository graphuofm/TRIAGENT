"""L1: Committee Data Collection.

Run:
    python experiments/L1_data_collection.py --sample 100  # quick test
    python experiments/L1_data_collection.py --yes         # full FPB run
    python experiments/L1_data_collection.py --dataset tfns --yes  # Twitter customer text
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR
from src.utils.data_loader import load_fpb
from src.utils.customer_data_loader import load_tfns, load_finchina_sentiment
from src.agents.vader_agent import VADERAgent
from src.agents.finbert_agent import FinBERTAgent
from src.agents.llm_agent import LLMAgent
from sklearn.metrics import accuracy_score, f1_score


DATASETS = {
    'fpb':       ('financial_phrasebank',       load_fpb,                  'committee_data.csv'),
    'tfns':      ('twitter_financial_news',     load_tfns,                 'committee_data_tfns.csv'),
    'finchina':  ('finchina_sentiment_zh',      load_finchina_sentiment,   'committee_data_finchina.csv'),
}


def main(args):
    name, loader, out_filename = DATASETS[args.dataset]
    print(f"Dataset: {name}")
    df = loader()
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"⚠  Using sample of {len(df)} rows")

    df['sentence_id'] = range(len(df))

    # VADER
    print("\n=== L1: VADER ===")
    vader = VADERAgent()
    vader_results = vader.predict_batch(df['sentence'].tolist())
    df['vader_score'] = [r.score for r in vader_results]
    df['vader_label'] = [r.label for r in vader_results]
    df['vader_confidence'] = [r.confidence for r in vader_results]
    df['vader_latency_ms'] = [r.latency_ms for r in vader_results]
    df['vader_cost_usd'] = [r.cost_usd for r in vader_results]
    df['vader_pos'] = [r.extra['pos'] for r in vader_results]
    df['vader_neg'] = [r.extra['neg'] for r in vader_results]
    df['vader_neu'] = [r.extra['neu'] for r in vader_results]

    # FinBERT
    print("\n=== L2: FinBERT ===")
    finbert = FinBERTAgent()
    finbert_results = finbert.predict_batch(df['sentence'].tolist(), batch_size=32)
    df['finbert_score'] = [r.score for r in finbert_results]
    df['finbert_label'] = [r.label for r in finbert_results]
    df['finbert_confidence'] = [r.confidence for r in finbert_results]
    df['finbert_latency_ms'] = [r.latency_ms for r in finbert_results]
    df['finbert_cost_usd'] = [r.cost_usd for r in finbert_results]
    df['finbert_prob_pos'] = [r.extra['prob_pos'] for r in finbert_results]
    df['finbert_prob_neg'] = [r.extra['prob_neg'] for r in finbert_results]
    df['finbert_prob_neu'] = [r.extra['prob_neu'] for r in finbert_results]

    # LLM
    if not args.skip_llm:
        from src.config import LOCAL_LLM_MODEL
        print(f"\n=== L3: Local LLM ({LOCAL_LLM_MODEL}) ===")
        print("⚠  Running on local GPU; first call downloads ~15GB of weights.")
        if not args.yes:
            confirm = input("Continue? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

        llm = LLMAgent()
        llm_results = llm.predict_batch(df['sentence'].tolist(), batch_size=8)
        df['llm_score'] = [r.score for r in llm_results]
        df['llm_label'] = [r.label for r in llm_results]
        df['llm_confidence'] = [r.confidence for r in llm_results]
        df['llm_latency_ms'] = [r.latency_ms for r in llm_results]
        df['llm_cost_usd'] = [r.cost_usd for r in llm_results]
        df['llm_reasoning'] = [r.extra['reasoning'] for r in llm_results]
        df['llm_input_tokens'] = [r.extra['input_tokens'] for r in llm_results]
        df['llm_output_tokens'] = [r.extra['output_tokens'] for r in llm_results]

        print(f"\n💰 Total LLM cost: ${df['llm_cost_usd'].sum():.3f}")

    # Save
    out_path = RESULTS_DATA_DIR / out_filename
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")

    # Reproduction check
    print("\n=== Reproduction Check ===")
    agents = ['vader', 'finbert'] + (['llm'] if not args.skip_llm else [])
    for agent in agents:
        acc = accuracy_score(df['label_text'], df[f'{agent}_label'])
        f1 = f1_score(df['label_text'], df[f'{agent}_label'], average='macro')
        latency = df[f'{agent}_latency_ms'].mean()
        cost = df[f'{agent}_cost_usd'].sum()
        print(f"  {agent:>8}: Acc={acc:.4f}  F1={f1:.4f}  "
              f"Latency={latency:.2f}ms  TotalCost=${cost:.4f}")

    print("\nExpected baselines:")
    print("  VADER:   Acc≈0.5433  F1≈0.4889")
    print("  FinBERT: Acc≈0.8894  F1≈0.8822")
    print("  LLM:     Acc≈0.84    F1≈0.84")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS.keys()), default='fpb',
                        help='Which corpus to run the committee on (fpb=Financial PhraseBank, '
                             'tfns=Twitter Financial News Sentiment / customer-style text)')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--yes', '-y', action='store_true')
    args = parser.parse_args()
    main(args)
