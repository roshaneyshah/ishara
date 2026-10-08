"""Blind scoring sheets, score import, weighted kappa and judge agreement."""
import hashlib
import random

import numpy as np
import pandas as pd

from .util import read_jsonl, rpath

KEY_COLS = ["arm", "budget", "seed", "prompt_id", "scenario_id", "language", "context", "sample_idx"]


def item_id(rec, salt):
    s = "|".join(str(rec[k]) for k in ["arm", "budget", "seed", "prompt_id", "sample_idx"]) + "|" + salt
    return "it" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def select_items(outputs, samples_per_prompt=None, arms=None, budgets=None, seeds=None):
    out = []
    for r in outputs:
        if samples_per_prompt is not None and r["sample_idx"] >= samples_per_prompt:
            continue
        if arms and r["arm"] not in arms:
            continue
        if budgets is not None and r["arm"] != "A0" and r["budget"] not in budgets:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        out.append(r)
    return out


def teacher_items(cfg, held_ids):
    """Teacher outputs on held-out prompts, scored and tagged like any other item (RQ3)."""
    path = rpath(cfg, cfg["a4"]["teacher_heldout_file"])
    if not path.exists():
        return []
    rows = []
    for r in read_jsonl(path):
        if r["prompt_id"] not in held_ids:
            raise RuntimeError(f"Teacher held-out file has non held-out prompt {r['prompt_id']}")
        if not str(r.get("response", "")).strip():
            raise ValueError(f"Teacher held-out row {r['prompt_id']} has an empty response")
        rows.append({"arm": "TEACHER", "budget": 0, "seed": 0, "prompt_id": r["prompt_id"],
                     "scenario_id": r["scenario_id"], "language": r["language"], "context": r["context"],
                     "sample_idx": int(r.get("sample_idx", 0)), "prompt": r["prompt"],
                     "response": r["response"], "truncated": bool(r.get("truncated", False))})
    return rows


def make_sheets(items, prompts_df, dims, coders, salt, out_dir):
    """One shuffled sheet per coder. The key linking item ids to arms stays in a separate file."""
    ref = dict(zip(prompts_df["prompt_id"], prompts_df["reference"]))
    key_rows, base = [], []
    seen = set()
    for r in items:
        iid = item_id(r, salt)
        if iid in seen:
            raise RuntimeError("Item id collision. Change the salt.")
        seen.add(iid)
        key_rows.append({"item_id": iid, **{k: r[k] for k in KEY_COLS}, "truncated": r["truncated"]})
        base.append({"item_id": iid, "prompt": r["prompt"], "what_the_user_meant": ref.get(r["prompt_id"], ""),
                     "response": r["response"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key_rows).to_csv(out_dir / "KEY_do_not_open_until_scoring_is_done.csv", index=False,
                                  encoding="utf-8-sig")
    paths = []
    for ci, coder in enumerate(coders):
        rows = [dict(b) for b in base]
        random.Random(f"{salt}|{coder}|{ci}").shuffle(rows)
        for row in rows:
            for d in dims:
                row[d] = ""
            row["error_tags"] = ""
            row["notes"] = ""
        df = pd.DataFrame(rows)
        if not df["what_the_user_meant"].astype(str).str.strip().any():
            df = df.drop(columns=["what_the_user_meant"])
        p = out_dir / f"sheet_{coder}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    return paths


def load_scores(sheet_path, dims, lo=0, hi=2, allowed_tags=None):
    df = pd.read_csv(sheet_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    problems = []
    for d in dims:
        v = pd.to_numeric(df[d], errors="coerce")
        bad = df.loc[v.isna() | (v < lo) | (v > hi) | (v != v.round()), "item_id"].tolist()
        if bad:
            problems.append(f"{d}: {len(bad)} missing or invalid, for example {bad[:3]}")
        df[d] = v
    if allowed_tags is not None:
        for r in df.itertuples():
            for t in [x.strip() for x in str(r.error_tags).split("|") if x.strip()]:
                if t not in allowed_tags:
                    problems.append(f"{r.item_id}: unknown error tag {t}")
    if problems:
        raise ValueError(f"{sheet_path}:\n" + "\n".join(problems[:20]))
    df["total"] = df[dims].sum(axis=1)
    return df[["item_id"] + dims + ["total", "error_tags"]]


def weighted_kappa(a, b, categories, weights="quadratic"):
    """Cohen's weighted kappa for two raters over the given ordered categories."""
    a, b = np.asarray(a), np.asarray(b)
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) ** 2 if weights == "quadratic" else np.abs(i - j)) / (k - 1) ** (2 if weights == "quadratic" else 1)
    den = (w * exp).sum()
    if den == 0:
        return float("nan")
    return 1.0 - (w * obs).sum() / den


def agreement_table(s1, s2, dims, lo=0, hi=2):
    m = s1.merge(s2, on="item_id", suffixes=("_1", "_2"))
    rows = []
    for d in dims:
        rows.append({"measure": d, "n": len(m),
                     "weighted_kappa": weighted_kappa(m[f"{d}_1"].astype(int), m[f"{d}_2"].astype(int),
                                                      list(range(lo, hi + 1))),
                     "exact_agreement": float((m[f"{d}_1"] == m[f"{d}_2"]).mean()) if len(m) else float("nan")})
    tot_hi = hi * len(dims)
    rows.append({"measure": "total", "n": len(m),
                 "weighted_kappa": weighted_kappa(m["total_1"].astype(int), m["total_2"].astype(int),
                                                  list(range(lo * len(dims), tot_hi + 1))),
                 "exact_agreement": float((m["total_1"] == m["total_2"]).mean()) if len(m) else float("nan")})
    return pd.DataFrame(rows), m


def thumbs_agreement(human_total, judge_total, threshold):
    h = np.asarray(human_total) >= threshold
    j = np.asarray(judge_total) >= threshold
    acc = float((h == j).mean()) if len(h) else float("nan")
    kap = weighted_kappa(h.astype(int), j.astype(int), [0, 1], weights="linear")
    return {"n": int(len(h)), "accuracy": acc, "cohen_kappa": kap,
            "human_up_rate": float(h.mean()) if len(h) else float("nan"),
            "judge_up_rate": float(j.mean()) if len(j) else float("nan")}
