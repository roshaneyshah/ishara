"""Runs each arm on the held-out set and writes outputs, manifests and compute logs.
Every run can be resumed after a Colab disconnect: finished prompts are skipped."""
import random
from pathlib import Path

from . import data as D
from .backend import build_messages
from .feedback import budget_subset, load_labels
from .icl import select_demos
from .util import append_jsonl, manifest, out_dir, read_json, read_jsonl, rpath, write_json, write_jsonl


def run_dir(cfg, arm, budget, seed, split="heldout"):
    tag = f"{split}_s{seed}" if arm == "A0" else f"b{budget}_s{seed}"
    d = out_dir(cfg) / arm / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_compute(cfg, backend, arm, budget, seed, phase, secs):
    append_jsonl(out_dir(cfg) / "compute.jsonl",
                 {"arm": arm, "budget": budget, "seed": seed, "phase": phase, "secs": secs,
                  "device": getattr(backend, "device_name", None)})


def generate_set(cfg, backend, prompts_df, arm, budget, seed, rdir, make_messages, extra=None):
    """Generate n_samples per prompt, appending to outputs.jsonl and skipping finished prompts."""
    n = cfg["generation"]["n_samples"]
    out_path = rdir / "outputs.jsonl"
    done = set()
    if out_path.exists():
        prev = read_jsonl(out_path)
        counts = {}
        for r in prev:
            counts[r["prompt_id"]] = counts.get(r["prompt_id"], 0) + 1
        done = {p for p, c in counts.items() if c == n}
        if len(done) != len(counts):
            write_jsonl(out_path, [r for r in prev if r["prompt_id"] in done])
    total = 0.0
    for r in prompts_df.itertuples():
        if r.prompt_id in done:
            continue
        messages, meta = make_messages(r)
        res, secs = backend.generate(messages, n, seed, f"{arm}|{r.prompt_id}")
        total += secs
        for i, o in enumerate(res):
            rec = {"arm": arm, "budget": budget, "seed": seed, "prompt_id": r.prompt_id,
                   "scenario_id": r.scenario_id, "language": r.language, "context": r.context,
                   "sample_idx": i, "prompt": r.prompt, **o, **meta, **(extra or {})}
            append_jsonl(out_path, rec)
    log_compute(cfg, backend, arm, budget, seed, "generate", total)
    return out_path


def _prep(cfg):
    df = D.load_prompts(cfg)
    pool, held = D.partition(cfg, df)
    return df, pool, held


def run_a0(cfg, backend, seed, split="heldout"):
    _, pool, held = _prep(cfg)
    target = held if split == "heldout" else pool
    rdir = run_dir(cfg, "A0", 0, seed, split)
    write_json(rdir / "manifest.json", manifest(cfg, {"arm": "A0", "split": split, "seed": seed}))
    return generate_set(cfg, backend, target, "A0", 0, seed, rdir,
                        lambda r: (build_messages(r.prompt), {}))


def run_a1(cfg, backend, budget, seed):
    D.check_rubric_frozen(cfg)
    _, pool, held = _prep(cfg)
    dims = cfg["rubric"]["dimensions"]
    labels = load_labels(cfg, set(pool["prompt_id"]), dims)
    sig = budget_subset(labels, budget, seed)
    rdir = run_dir(cfg, "A1", budget, seed)
    k = cfg["a1"]["k_demos"]
    n_up = int(sig["up"].sum())
    write_json(rdir / "manifest.json", manifest(cfg, {
        "arm": "A1", "budget": budget, "seed": seed, "signal_ids": sig["signal_id"].tolist(),
        "n_up": n_up, "fallback_to_base": n_up == 0}))

    def mk(r):
        demos = select_demos(r.prompt, sig, k, dims)
        return build_messages(r.prompt, demos=demos), {"demo_signal_ids": [d["signal_id"] for d in demos]}
    return generate_set(cfg, backend, held, "A1", budget, seed, rdir, mk)


def run_a2(cfg, backend, oracle, budget, seed):
    from .prompt_search import load_fragments, render, search
    D.check_rubric_frozen(cfg)
    _, pool, held = _prep(cfg)
    rdir = run_dir(cfg, "A2", budget, seed)
    best_path = rdir / "search.json"
    if best_path.exists():
        res = read_json(best_path)
        fragments = res["fragments"]
        best = tuple(res["best_genome"])
    else:
        fragments = load_fragments(rpath(cfg, cfg["a2"]["fragments_file"]))
        log = []
        best, stats = search(backend, oracle, pool, fragments, budget, seed, cfg["a2"], log)
        secs = sum(x["gen_secs"] for x in log)
        log_compute(cfg, backend, "A2", budget, seed, "search", secs)
        write_jsonl(rdir / "search_signals.jsonl", log)
        write_json(best_path, {"fragments": fragments, "best_genome": list(best), "stats": stats,
                               "signals_used": len(log), "oracle": cfg["a2"]["oracle"],
                               "oracle_parse_failures": getattr(oracle, "parse_failures", 0)})
    system = render(best, fragments)
    write_json(rdir / "manifest.json", manifest(cfg, {"arm": "A2", "budget": budget, "seed": seed,
                                                      "system_prompt": system}))
    return generate_set(cfg, backend, held, "A2", budget, seed, rdir,
                        lambda r: (build_messages(r.prompt, system), {}))


def _run_lora_arm(cfg, backend, arm, pairs, budget, seed, meta):
    _, _, held = _prep(cfg)
    rdir = run_dir(cfg, arm, budget, seed)
    adapter = rdir / "adapter"
    fallback = len(pairs) == 0
    if not fallback and not (adapter / "adapter_config.json").exists():
        secs, losses = backend.train_lora(pairs, seed, cfg["a3"], adapter)
        log_compute(cfg, backend, arm, budget, seed, "train", secs)
        write_json(rdir / "train_log.json", {"n_pairs": len(pairs), "losses": losses})
    write_json(rdir / "manifest.json", manifest(cfg, {"arm": arm, "budget": budget, "seed": seed,
                                                      "n_train_pairs": len(pairs),
                                                      "fallback_to_base": fallback, **meta}))
    if fallback:
        backend.clear_adapter()
    else:
        backend.load_adapter(adapter)
    try:
        return generate_set(cfg, backend, held, arm, budget, seed, rdir,
                            lambda r: (build_messages(r.prompt), {}))
    finally:
        backend.clear_adapter()


def run_a3(cfg, backend, budget, seed):
    D.check_rubric_frozen(cfg)
    _, pool, _ = _prep(cfg)
    labels = load_labels(cfg, set(pool["prompt_id"]), cfg["rubric"]["dimensions"])
    sig = budget_subset(labels, budget, seed)
    ups = sig[sig["up"]]
    pairs = [(build_messages(r.prompt), r.response) for r in ups.itertuples()]
    return _run_lora_arm(cfg, backend, "A3", pairs, budget, seed,
                         {"signal_ids": sig["signal_id"].tolist(), "n_up": len(ups)})


def load_synthetic(cfg, pool, held):
    """Teacher data for A4. Refuses any row that points at, or copies, a held-out prompt."""
    rows = read_jsonl(rpath(cfg, cfg["a4"]["synthetic_file"]))
    held_ids, held_text = set(held["prompt_id"]), set(held["prompt"].str.strip())
    for i, r in enumerate(rows):
        for k in ("synthetic_id", "prompt", "response"):
            if k not in r:
                raise ValueError(f"Synthetic row {i} lacks `{k}`")
        if not str(r["response"]).strip():
            raise ValueError(f"Synthetic row {r['synthetic_id']} has an empty response")
        if r.get("prompt_id") in held_ids or r["prompt"].strip() in held_text:
            raise RuntimeError(f"Synthetic row {r['synthetic_id']} uses a held-out prompt. Leakage.")
    return rows


def run_a4(cfg, backend, budget, seed):
    D.check_rubric_frozen(cfg)
    _, pool, held = _prep(cfg)
    rows = load_synthetic(cfg, pool, held)
    if budget > len(rows):
        raise ValueError(f"Budget {budget} needs {budget} synthetic rows. Found {len(rows)}.")
    order = list(range(len(rows)))
    random.Random(2000 + seed).shuffle(order)
    chosen = [rows[i] for i in order[:budget]]
    pairs = [(build_messages(r["prompt"]), r["response"]) for r in chosen]
    return _run_lora_arm(cfg, backend, "A4", pairs, budget, seed,
                         {"synthetic_ids": [r["synthetic_id"] for r in chosen]})


def collect_outputs(cfg):
    """All held-out outputs from every arm, as one list."""
    rows = []
    base = out_dir(cfg)
    for p in sorted(Path(base).glob("*/*/outputs.jsonl")):
        if p.parent.parent.name == "A0" and not p.parent.name.startswith("heldout"):
            continue
        rows.extend(read_jsonl(p))
    return rows
