"""L1 agent: VADER lexicon-based sentiment."""
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .base import SentimentAgent, AgentOutput


class VADERAgent(SentimentAgent):
    name = "vader"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> AgentOutput:
        t0 = time.perf_counter()
        scores = self.analyzer.polarity_scores(text)
        latency = (time.perf_counter() - t0) * 1000

        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return AgentOutput(
            label=label,
            score=compound,
            confidence=abs(compound),
            latency_ms=latency,
            cost_usd=0.0,
            extra={
                'pos': scores['pos'],
                'neg': scores['neg'],
                'neu': scores['neu'],
            }
        )
