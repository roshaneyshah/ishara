"""Feedback signals: the labelling sheet for A0 pool outputs, and nested budget subsets."""
import random

import pandas as pd

from .util import read_jsonl, rpath


def export_label_sheet(cfg, pool_outputs_path, out_csv, dims):
    """One row per A0 output on the feedback pool, shuffled. Fill `thumbs` with up or down.
    Rubric columns are optional. Truncated outputs are kept and marked, since they are real outputs."""
    rows = read_jsonl(pool_outputs_path)
    rng = random.Random(cfg["data"]["split_seed"] + 1)
    rng.shuffle(rows)
    recs = []
    for i, r in enumerate(rows):
        rec = {"signal_id": f"sig{i:04d}", "prompt_id": r["prompt_id"], "sample_idx": r["sample_idx"],
               "prompt": r["prompt"], "response": r["response"], "truncated": r["truncated"],
               "thumbs": "", "notes": ""}
        for d in dims:
            rec[d] = ""
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return len(df)


def load_labels(cfg, pool_ids, dims):
    """Labelled signals. Only rows with thumbs filled count. Fails on rows outside the pool."""
    path = rpath(cfg, cfg["feedback"]["labels_file"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["thumbs"] = df["thumbs"].str.strip().str.lower()
    bad = sorted(set(df["thumbs"]) - {"up", "down", ""})
    if bad:
        raise ValueError(f"thumbs must be up or down. Found {bad}")
    df = df[df["thumbs"] != ""].copy()
    outside = sorted(set(df["prompt_id"]) - set(pool_ids))
    if outside:
        raise RuntimeError(f"Labelled signals from prompts outside the feedback pool: {outside}")
    df["up"] = df["thumbs"] == "up"
    for d in dims:
        if d in df.columns:
            df[d] = pd.to_numeric(df[d], errors="coerce")
    return df.reset_index(drop=True)


def budget_subset(labels, budget, seed):
    """Nested subsets: for a fixed seed, the 10-signal set is inside the 25-signal set, and so on."""
    if budget > len(labels):
        raise ValueError(f"Budget {budget} needs {budget} labelled signals. Only {len(labels)} are labelled.")
    order = list(range(len(labels)))
    random.Random(1000 + seed).shuffle(order)
    return labels.iloc[order[:budget]].reset_index(drop=True)
