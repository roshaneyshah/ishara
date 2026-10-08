"""A1: pick in-context examples from thumbs-up feedback by character n-gram similarity."""
import math
from collections import Counter


def _grams(text, n=(3, 4)):
    t = " " + " ".join((text or "").lower().split()) + " "
    c = Counter()
    for k in n:
        for i in range(len(t) - k + 1):
            c[t[i:i + k]] += 1
    return c


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def select_demos(query, signals, k, dims=()):
    """signals: DataFrame of labelled feedback. Uses thumbs-up rows only, at most one per prompt.
    Ties on similarity are broken by rubric total when present, then by signal_id.
    Returned in order of increasing similarity, so the closest example sits next to the query."""
    ups = signals[signals["up"]]
    if ups.empty or k <= 0:
        return []
    q = _grams(query)
    cands = []
    for r in ups.itertuples():
        total = sum(getattr(r, d) for d in dims if hasattr(r, d) and getattr(r, d) == getattr(r, d))
        cands.append((_cos(q, _grams(r.prompt)), total, r.signal_id, r))
    cands.sort(key=lambda x: (-x[0], -x[1], x[2]))
    chosen, seen = [], set()
    for sim, _, _, r in cands:
        if r.prompt_id in seen:
            continue
        seen.add(r.prompt_id)
        chosen.append({"signal_id": r.signal_id, "prompt_id": r.prompt_id, "prompt": r.prompt,
                       "response": r.response, "similarity": round(sim, 4)})
        if len(chosen) == k:
            break
    return list(reversed(chosen))
