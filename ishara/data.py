"""Prompt loading, design checks and the scenario split."""
import random

import pandas as pd

from .lexicon import roman_urdu_hits
from .util import read_json, rpath, sha256_file, write_json

LANGS = ("english", "roman_urdu")
CONTEXTS = ("explicit", "implicit")


def load_prompts(cfg):
    """Read the prompt file and rename columns and values to the internal names."""
    dcfg = cfg["data"]
    path = rpath(cfg, dcfg["prompts_file"])
    if str(path).endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = dcfg["columns"]
    required = ["prompt_id", "scenario_id", "language", "context", "prompt"]
    missing = [cols[k] for k in required if cols[k] not in df.columns]
    if missing:
        raise ValueError(f"Prompt file is missing columns {missing}. Found {list(df.columns)}. "
                         f"Fix data.columns in the config.")
    rename = {cols[k]: k for k in required}
    if cols.get("reference") and cols["reference"] in df.columns:
        rename[cols["reference"]] = "reference"
    df = df.rename(columns=rename)
    if "reference" not in df.columns:
        df["reference"] = ""
    lang_map = {v: k for k, v in dcfg["language_values"].items()}
    ctx_map = {v: k for k, v in dcfg["context_values"].items()}
    bad_l = sorted(set(df["language"]) - set(lang_map))
    bad_c = sorted(set(df["context"]) - set(ctx_map))
    if bad_l or bad_c:
        raise ValueError(f"Unknown language values {bad_l} or context values {bad_c}. "
                         f"Fix data.language_values and data.context_values in the config.")
    df["language"] = df["language"].map(lang_map)
    df["context"] = df["context"].map(ctx_map)
    for c in ["prompt_id", "scenario_id", "prompt", "reference"]:
        df[c] = df[c].astype(str)
    return df[["prompt_id", "scenario_id", "language", "context", "prompt", "reference"]].copy()


def validate(df):
    """Check the 2x2 design and English purity. Returns (errors, warnings) as lists of strings."""
    errors, warnings = [], []
    dup = df["prompt_id"][df["prompt_id"].duplicated()].tolist()
    if dup:
        errors.append(f"Duplicate prompt_id values: {dup}")
    dup_text = df["prompt"][df["prompt"].duplicated()].tolist()
    if dup_text:
        errors.append(f"Identical prompt text appears more than once: {dup_text[:5]}")
    empty = df.loc[df["prompt"].str.strip() == "", "prompt_id"].tolist()
    if empty:
        errors.append(f"Empty prompts: {empty}")
    for sid, g in df.groupby("scenario_id"):
        cells = sorted(zip(g["language"], g["context"]))
        want = sorted((lang, ctx) for lang in LANGS for ctx in CONTEXTS)
        if cells != want:
            errors.append(f"Scenario {sid} does not have exactly one prompt per 2x2 cell. Has {cells}")
            continue
        by = {(r.language, r.context): r.prompt for r in g.itertuples()}
        # One-dimension rule: prompts that share a language should differ, prompts that share a
        # context should differ. Word-level identity across languages cannot be checked by code.
        for lang in LANGS:
            if by[(lang, "explicit")] == by[(lang, "implicit")]:
                errors.append(f"Scenario {sid} {lang}: explicit and implicit prompts are identical")
    for r in df[df["language"] == "english"].itertuples():
        hits = roman_urdu_hits(r.prompt)
        if hits:
            warnings.append(f"English prompt {r.prompt_id} (scenario {r.scenario_id}) contains "
                            f"possible Roman Urdu words {sorted(set(hits))}. Check by hand.")
    for r in df[df["language"] == "roman_urdu"].itertuples():
        if not roman_urdu_hits(r.prompt):
            warnings.append(f"Roman Urdu prompt {r.prompt_id} has no word from the Roman Urdu list. "
                            f"Check that it is labelled correctly.")
    n_s = df["scenario_id"].nunique()
    if n_s < 40:
        warnings.append(f"{n_s} scenarios found. The scope targets roughly 40.")
    return errors, warnings


def make_split(cfg, df, force=False):
    """Split by scenario into a feedback pool and a held-out set. Written once, then frozen."""
    dcfg = cfg["data"]
    path = rpath(cfg, dcfg["split_file"])
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. The split is frozen. Pass --force only if no "
                              f"adapted run has used it yet.")
    errors, _ = validate(df)
    if errors:
        raise ValueError("Fix the prompt file before splitting:\n" + "\n".join(errors))
    scen = sorted(df["scenario_id"].unique())
    rng = random.Random(dcfg["split_seed"])
    rng.shuffle(scen)
    n_held = round(len(scen) * dcfg["heldout_fraction"])
    held = sorted(scen[:n_held])
    pool = sorted(scen[n_held:])
    split = {
        "prompts_sha256": sha256_file(rpath(cfg, dcfg["prompts_file"])),
        "split_seed": dcfg["split_seed"],
        "heldout_fraction": dcfg["heldout_fraction"],
        "feedback_pool_scenarios": pool,
        "heldout_scenarios": held,
        "n_feedback_pool_prompts": int(df["scenario_id"].isin(pool).sum()),
        "n_heldout_prompts": int(df["scenario_id"].isin(held).sum()),
    }
    write_json(path, split)
    return split


def load_split(cfg, df=None):
    """Load the frozen split and refuse to continue if the prompt file changed after splitting."""
    path = rpath(cfg, cfg["data"]["split_file"])
    if not path.exists():
        raise FileNotFoundError(f"No split at {path}. Run the `split` command first.")
    split = read_json(path)
    now = sha256_file(rpath(cfg, cfg["data"]["prompts_file"]))
    if now != split["prompts_sha256"]:
        raise RuntimeError("The prompt file changed after the split was written. Restore it, or make a "
                           "new split before any adapted run.")
    pool, held = set(split["feedback_pool_scenarios"]), set(split["heldout_scenarios"])
    if pool & held:
        raise RuntimeError(f"Scenarios in both sets: {sorted(pool & held)}")
    if df is not None:
        known = set(df["scenario_id"])
        if known != pool | held:
            raise RuntimeError("Scenario ids in the prompt file do not match the split.")
    return split


def partition(cfg, df):
    split = load_split(cfg, df)
    pool = df[df["scenario_id"].isin(split["feedback_pool_scenarios"])].reset_index(drop=True)
    held = df[df["scenario_id"].isin(split["heldout_scenarios"])].reset_index(drop=True)
    return pool, held


def freeze_rubric(cfg):
    """Record the rubric hash. Adapted runs refuse to start if the rubric changes afterwards."""
    rub = rpath(cfg, cfg["rubric"]["file"])
    text = rub.read_text(encoding="utf-8")
    if "REPLACE" in text:
        raise ValueError("rubric.md still has REPLACE placeholders. Paste the anchors from the first "
                         "study before freezing.")
    lock = rub.with_suffix(".lock.json")
    if lock.exists():
        raise FileExistsError(f"{lock} exists. The rubric is already frozen.")
    write_json(lock, {"rubric_sha256": sha256_file(rub)})
    return lock


def check_rubric_frozen(cfg):
    rub = rpath(cfg, cfg["rubric"]["file"])
    lock = rub.with_suffix(".lock.json")
    if not lock.exists():
        raise RuntimeError("Rubric is not frozen. Run `freeze-rubric` before any adapted run.")
    if read_json(lock)["rubric_sha256"] != sha256_file(rub):
        raise RuntimeError("rubric.md changed after it was frozen.")

