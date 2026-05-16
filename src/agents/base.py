"""Abstract base class for all sentiment agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentOutput:
    label: str
    score: float
    confidence: float
    latency_ms: float
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class SentimentAgent(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, text: str) -> AgentOutput:
        ...

    def predict_batch(self, texts: list[str]) -> list[AgentOutput]:
        return [self.predict(t) for t in texts]
