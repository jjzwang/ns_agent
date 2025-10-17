from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re
class BM25Memory:
    def __init__(self):
        self._bm = None
        self._docs = []
        self._metas = []
    def build(self, docs: List[str], metas: List[Dict[str, Any]]):
        self._docs = [self._tok(d) for d in docs]
        self._metas = metas
        self._bm = BM25Okapi(self._docs)
    def topk(self, query: str, k: int = 200) -> List[Dict[str, Any]]:
        tokens = self._tok(query)
        scores = self._bm.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"text": " ".join(self._docs[i]), "metadata": self._metas[i], "score": float(scores[i])} for i in idxs]
    @staticmethod
    def _tok(t: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", t.lower())
