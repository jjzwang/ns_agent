from typing import List, Dict, Any
def fuse(vec_hits: List[Dict[str, Any]], bm_hits: List[Dict[str, Any]], w_vec=0.6, w_bm=0.4, k=8):
    def key(hit):
        m = hit.get("metadata", {})
        return m.get("id") or m.get("source_url") or m.get("title") or hit.get("text")[:64]
    scores = {}
    for idx, h in enumerate(vec_hits):
        kk = key(h); scores.setdefault(kk, {"hit": h, "score": 0.0})
        scores[kk]["score"] += w_vec * (1.0 / (1.0 + idx))
    for idx, h in enumerate(bm_hits):
        kk = key(h); scores.setdefault(kk, {"hit": h, "score": 0.0})
        scores[kk]["score"] += w_bm * (1.0 / (1.0 + idx))
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["hit"] for r in ranked[:k]]
