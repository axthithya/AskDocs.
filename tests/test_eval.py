from eval.run_eval import evaluate, is_hit
from eval.run_generation import refuses

RESULTS = [
    {"source": "a.pdf", "page": 1, "text": "The Reliability pillar covers recovery.", "score": 0.9},
    {"source": "b.pdf", "page": 2, "text": "Security is important.", "score": 0.8},
]


class FakeRetriever:
    def search(self, query, k):
        return RESULTS[:k]


def test_hit_requires_source_and_keyword():
    assert is_hit({"source": "a.pdf", "keyword": "reliability"}, RESULTS)
    assert not is_hit({"source": "b.pdf", "keyword": "reliability"}, RESULTS)
    assert not is_hit({"source": "a.pdf", "keyword": "cost"}, RESULTS)


def test_evaluate_counts_hits_and_respects_k():
    qs = [
        {"question": "q1", "source": "a.pdf", "keyword": "reliability"},
        {"question": "q2", "source": "b.pdf", "keyword": "security"},
    ]
    assert evaluate(FakeRetriever(), qs, 2, verbose=False) == 2
    assert evaluate(FakeRetriever(), qs, 1, verbose=False) == 1


def test_refusal_detection():
    assert refuses("I couldn’t find that in the provided documents.")
    assert not refuses("The capital is Canberra.")
