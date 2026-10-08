"""End to end test with the stand-in model, plus checks on the data rules.
Run: python -m pytest tests -q"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from make_fake_project import build  # noqa: E402

from ishara import data as D  # noqa: E402
from ishara.backend import FakeBackend  # noqa: E402
from ishara.judge import Judge, parse_scores  # noqa: E402
from ishara.prompt_search import load_fragments, search  # noqa: E402
from ishara.util import load_config  # noqa: E402

DIMS = ["intent_inference", "social_appropriateness", "unsupported_assumptions", "semantic_preservation"]


def run(fp, *args):
    cmd = [sys.executable, "-m", "ishara", "--config", str(fp / "configs" / "default.yaml"), *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


@pytest.fixture()
def fp(tmp_path):
    return build(tmp_path / "proj")


def test_english_purity_flag(fp):
    cfg = load_config(fp / "configs" / "default.yaml")
    df = D.load_prompts(cfg)
    i = df.index[(df.language == "english")][0]
    df.loc[i, "prompt"] = "Can you ask bhai if he is free?"
    _, warnings = D.validate(df)
    assert any("bhai" in w for w in warnings)


def test_design_errors(fp):
    cfg = load_config(fp / "configs" / "default.yaml")
    df = D.load_prompts(cfg)
    bad = df.drop(df.index[0])
    errors, _ = D.validate(bad)
    assert any("2x2" in e for e in errors)


def test_split_is_by_scenario_and_frozen(fp):
    run(fp, "split")
    cfg = load_config(fp / "configs" / "default.yaml")
    df = D.load_prompts(cfg)
    pool, held = D.partition(cfg, df)
    assert not set(pool.scenario_id) & set(held.scenario_id)
    assert len(pool) + len(held) == len(df)
    r = subprocess.run([sys.executable, "-m", "ishara", "--config", str(fp / "configs" / "default.yaml"), "split"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    p = fp / "data" / "prompts.csv"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(RuntimeError):
        D.partition(cfg, df)


def test_prompt_search_uses_exact_budget(fp):
    run(fp, "split")
    cfg = load_config(fp / "configs" / "default.yaml")
    df = D.load_prompts(cfg)
    pool, _ = D.partition(cfg, df)
    frags = load_fragments(fp / "configs" / "prompt_fragments.txt")
    judge = Judge(FakeBackend(256, judge_dims=DIMS), "rubric", DIMS, 6)
    for budget in [10, 25, 37, 200]:
        log = []
        best, stats = search(FakeBackend(), judge, pool, frags, budget, 0, cfg["a2"], log)
        assert len(log) == budget
        assert sum(v["n"] for v in stats.values()) == budget
        assert all(r["prompt_id"] in set(pool.prompt_id) for r in log)


def test_parse_scores():
    assert parse_scores('x {"a": 1, "b": 2} y', ["a", "b"]) == {"a": 1, "b": 2}
    assert parse_scores('{"a": 3, "b": 2}', ["a", "b"]) is None
    assert parse_scores("no json", ["a"]) is None


def test_synthetic_leak_is_refused(fp):
    run(fp, "split")
    cfg = load_config(fp / "configs" / "default.yaml")
    df = D.load_prompts(cfg)
    pool, held = D.partition(cfg, df)
    from ishara.runner import load_synthetic
    r = held.iloc[0]
    (fp / "data" / "synthetic_teacher.jsonl").write_text(json.dumps(
        {"synthetic_id": "x", "prompt": r.prompt, "response": "y"}) + "\n")
    with pytest.raises(RuntimeError):
        load_synthetic(cfg, pool, held)


def test_end_to_end(fp):
    run(fp, "split")
    run(fp, "freeze-rubric")
    run(fp, "--fake", "a0", "--split", "pool", "--seeds", "0")
    run(fp, "--fake", "a0", "--seeds", "0,1")
    run(fp, "label-sheet")
    lab = pd.read_csv(fp / "data" / "feedback_labels_TO_FILL.csv", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    rng = random.Random(0)
    lab["thumbs"] = [rng.choice(["up", "down"]) for _ in range(len(lab))]
    lab.to_csv(fp / "data" / "feedback_labels.csv", index=False, encoding="utf-8-sig")
    cfg = load_config(fp / "configs" / "default.yaml")
    pool, held = D.partition(cfg, D.load_prompts(cfg))
    with open(fp / "data" / "synthetic_teacher.jsonl", "w") as f:
        for i, r in enumerate(pool.itertuples()):
            f.write(json.dumps({"synthetic_id": f"s{i}", "prompt_id": r.prompt_id, "prompt": r.prompt,
                                "response": "t"}) + "\n")
    with open(fp / "data" / "teacher_heldout.jsonl", "w") as f:
        for r in held.itertuples():
            f.write(json.dumps({"prompt_id": r.prompt_id, "scenario_id": r.scenario_id, "language": r.language,
                                "context": r.context, "prompt": r.prompt, "response": "t"}) + "\n")
    for arm in ["a1", "a2", "a3", "a4"]:
        run(fp, "--fake", arm, "--budgets", "10,25", "--seeds", "0,1")
    # resume: rerunning must not duplicate outputs
    run(fp, "--fake", "a1", "--budgets", "10", "--seeds", "0")
    out = fp / "runs" / "primary" / "A1" / "b10_s0" / "outputs.jsonl"
    n = len(out.read_text().strip().splitlines())
    assert n == len(held) * cfg["generation"]["n_samples"]
    run(fp, "sheets", "--samples-per-prompt", "1", "--salt", "t")
    sd = fp / "runs" / "primary" / "scoring" / "round1"
    key = pd.read_csv(sd / "KEY_do_not_open_until_scoring_is_done.csv")
    for c in ["coder1", "coder2"]:
        s = pd.read_csv(sd / f"sheet_{c}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert "arm" not in s.columns and "budget" not in s.columns
        assert set(s.item_id) == set(key.item_id)
        for d in DIMS:
            s[d] = [str(rng.randint(0, 2)) for _ in range(len(s))]
        s.to_csv(sd / f"sheet_{c}.csv", index=False, encoding="utf-8-sig")
    run(fp, "analyze", "--scoring-dir", str(sd), "--sheets", str(sd / "sheet_coder1.csv"), str(sd / "sheet_coder2.csv"))
    res = fp / "runs" / "primary" / "results"
    t = pd.read_csv(res / "budget_table.csv")
    assert set(t.cell) >= {"A0", "A1@10", "A2@25", "A3@10", "A4@25"}
    assert (res / "results.md").exists() and (res / "budget_curve.png").exists()
