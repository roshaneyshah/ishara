"""Command line entry point. Run `python -m ishara --help`."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data as D
from .util import load_config, out_dir, rpath, write_json


def _backend(cfg, fake=False):
    if fake:
        from .backend import FakeBackend
        return FakeBackend(cfg["generation"]["max_new_tokens"])
    from .backend import HFBackend
    return HFBackend(cfg["model"], cfg["generation"])


def _judge_backend(cfg, fake=False):
    if fake:
        from .backend import FakeBackend
        return FakeBackend(256, judge_dims=cfg["rubric"]["dimensions"])
    from .backend import HFBackend
    j = cfg["judge"]
    if not j.get("hf_id") or not j.get("revision"):
        raise ValueError("Set judge.hf_id and judge.revision in the config.")
    mcfg = {"hf_id": j["hf_id"], "revision": j["revision"], "trust_remote_code": j.get("trust_remote_code", False),
            "dtype": "float16", "load_in_4bit": j.get("load_in_4bit", False), "chat_template_kwargs": {}}
    return HFBackend(mcfg, {"max_new_tokens": 256, "do_sample": False, "temperature": None, "top_p": None})


def _judge(cfg, fake=False):
    from .judge import Judge
    rubric = rpath(cfg, cfg["rubric"]["file"]).read_text(encoding="utf-8")
    return Judge(_judge_backend(cfg, fake), rubric, cfg["rubric"]["dimensions"], cfg["a2"]["judge_thumbs_threshold"])


def _ints(s):
    return [int(x) for x in s.split(",")] if s else None


def cmd_check(cfg, a):
    df = D.load_prompts(cfg)
    errors, warnings = D.validate(df)
    counts = df.groupby(["language", "context"]).size()
    print(f"{len(df)} prompts, {df.scenario_id.nunique()} scenarios")
    print(counts.to_string())
    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("No errors.")


def cmd_split(cfg, a):
    df = D.load_prompts(cfg)
    s = D.make_split(cfg, df, force=a.force)
    print(f"Feedback pool: {len(s['feedback_pool_scenarios'])} scenarios, {s['n_feedback_pool_prompts']} prompts")
    print(f"Held-out: {len(s['heldout_scenarios'])} scenarios, {s['n_heldout_prompts']} prompts")
    print(f"Written to {rpath(cfg, cfg['data']['split_file'])}. Commit this file before running anything.")


def cmd_freeze(cfg, a):
    print("Rubric frozen:", D.freeze_rubric(cfg))


def cmd_gpu(cfg, a):
    from .backend import require_gpu
    print("GPU:", require_gpu())


def cmd_a0(cfg, a):
    from .runner import run_a0
    b = _backend(cfg, a.fake)
    for s in _ints(a.seeds):
        print("wrote", run_a0(cfg, b, s, a.split))


def cmd_label_sheet(cfg, a):
    from .feedback import export_label_sheet
    from .runner import run_dir
    src = run_dir(cfg, "A0", 0, a.seed, "pool") / "outputs.jsonl"
    dst = rpath(cfg, a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = export_label_sheet(cfg, src, dst, cfg["rubric"]["dimensions"])
    print(f"{n} rows written to {dst}. Fill `thumbs` with up or down, then save it as "
          f"{cfg['feedback']['labels_file']}")


def cmd_teacher_inputs(cfg, a):
    """Prompt files for the teacher model. Fill `response` (and `teacher_model`) in each row."""
    from .util import write_jsonl
    df = D.load_prompts(cfg)
    pool, held = D.partition(cfg, df)
    n = a.samples
    pool_rows = [{"synthetic_id": f"syn_{r.prompt_id}_{i}", "prompt_id": r.prompt_id, "prompt": r.prompt,
                  "response": "", "teacher_model": ""} for r in pool.itertuples() for i in range(n)]
    held_rows = [{"prompt_id": r.prompt_id, "scenario_id": r.scenario_id, "language": r.language,
                  "context": r.context, "prompt": r.prompt, "sample_idx": 0, "response": "",
                  "teacher_model": ""} for r in held.itertuples()]
    p1 = rpath(cfg, "data/teacher_inputs_pool.jsonl")
    p2 = rpath(cfg, "data/teacher_inputs_heldout.jsonl")
    write_jsonl(p1, pool_rows)
    write_jsonl(p2, held_rows)
    print(f"{len(pool_rows)} pool rows to {p1}. When filled, save as {cfg['a4']['synthetic_file']}")
    print(f"{len(held_rows)} held-out rows to {p2}. When filled, save as {cfg['a4']['teacher_heldout_file']}")
    print("Held-out teacher outputs are only for RQ3 error tagging. They are never used for training.")


def cmd_arm(cfg, a):
    from . import runner
    budgets = _ints(a.budgets) or cfg["feedback"]["budgets"]
    seeds = _ints(a.seeds) or cfg["feedback"]["seeds"]
    b = _backend(cfg, a.fake)
    oracle = None
    if a.arm == "a2":
        if cfg["a2"]["oracle"] == "human":
            from .judge import HumanOracle
            oracle = HumanOracle()
        else:
            oracle = _judge(cfg, a.fake)
    for bud in budgets:
        for s in seeds:
            if a.arm == "a1":
                p = runner.run_a1(cfg, b, bud, s)
            elif a.arm == "a2":
                p = runner.run_a2(cfg, b, oracle, bud, s)
            elif a.arm == "a3":
                p = runner.run_a3(cfg, b, bud, s)
            else:
                p = runner.run_a4(cfg, b, bud, s)
            print("wrote", p)


def _held(cfg):
    df = D.load_prompts(cfg)
    _, held = D.partition(cfg, df)
    return df, held


def cmd_plan(cfg, a):
    from .runner import collect_outputs
    from .scoring import select_items
    outs = select_items(collect_outputs(cfg), a.samples_per_prompt, None, _ints(a.budgets), _ints(a.seeds))
    df = pd.DataFrame(outs)
    if df.empty:
        print("No outputs yet.")
        return
    t = df.groupby(["arm", "budget"]).size().rename("items").reset_index()
    print(t.to_string(index=False))
    print(f"Total items per coder: {len(df)}")


def cmd_sheets(cfg, a):
    from .runner import collect_outputs
    from .scoring import make_sheets, select_items, teacher_items
    df, held = _held(cfg)
    items = select_items(collect_outputs(cfg), a.samples_per_prompt, None, _ints(a.budgets), _ints(a.seeds))
    items += [t for t in teacher_items(cfg, set(held.prompt_id))
              if a.samples_per_prompt is None or t["sample_idx"] < a.samples_per_prompt]
    dest = out_dir(cfg) / "scoring" / a.name
    paths = make_sheets(items, df, cfg["rubric"]["dimensions"], a.coders.split(","), a.salt, dest)
    print(f"{len(items)} items. Sheets: " + ", ".join(str(p) for p in paths))
    print("Give each coder only their own sheet. Keep the KEY file closed until scoring is finished.")


def _allowed_tags(cfg):
    text = rpath(cfg, cfg["rubric"]["file"]).read_text(encoding="utf-8")
    sec = text.split("## Error tags", 1)
    if len(sec) < 2:
        return None
    tags = set()
    for ln in sec[1].splitlines():
        ln = ln.strip()
        if ln.startswith("- ") and ":" in ln:
            tags.add(ln[2:].split(":", 1)[0].strip())
    return tags or None


def cmd_kappa(cfg, a):
    from .scoring import agreement_table, load_scores
    dims = cfg["rubric"]["dimensions"]
    tags = _allowed_tags(cfg)
    s1, s2 = load_scores(a.sheet1, dims, allowed_tags=tags), load_scores(a.sheet2, dims, allowed_tags=tags)
    t, _ = agreement_table(s1, s2, dims)
    print(t.to_string(index=False))
    res = out_dir(cfg) / "results"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "interrater_kappa.csv", index=False)


def cmd_judge_score(cfg, a):
    """Run the judge on every item of a scoring sheet. Needs a GPU unless --fake."""
    j = _judge(cfg, a.fake)
    sheet = pd.read_csv(a.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = []
    for r in sheet.itertuples():
        s = j.score(r.prompt, r.response, getattr(r, "what_the_user_meant", ""), key=r.item_id)
        rows.append({"item_id": r.item_id, **(s or {d: None for d in cfg["rubric"]["dimensions"]}),
                     "parse_failed": s is None})
    out = Path(a.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Judge scores written to {out}. Parse failures: {j.parse_failures}")


def cmd_judge_agreement(cfg, a):
    from .scoring import agreement_table, load_scores, thumbs_agreement
    dims = cfg["rubric"]["dimensions"]
    human = load_scores(a.human, dims, allowed_tags=_allowed_tags(cfg))
    judge = pd.read_csv(a.judge)
    failed = int(judge["parse_failed"].sum())
    judge = judge[~judge["parse_failed"]].copy()
    for d in dims:
        judge[d] = judge[d].astype(int)
    judge["total"] = judge[dims].sum(axis=1)
    judge["error_tags"] = ""
    t, m = agreement_table(human, judge, dims)
    th = thumbs_agreement(m["total_1"], m["total_2"], cfg["a2"]["judge_thumbs_threshold"])
    res = out_dir(cfg) / "results"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "judge_vs_human_kappa.csv", index=False)
    write_json(res / "judge_vs_human_thumbs.json", {**th, "judge_parse_failures": failed})
    print(t.to_string(index=False))
    print("Thumbs agreement:", th, "Parse failures:", failed)


def cmd_analyze(cfg, a):
    from .analysis import compute_table, error_transfer, plot_budget_curve, score_table
    from .runner import collect_outputs
    from .scoring import load_scores
    from .report import write_report
    dims = cfg["rubric"]["dimensions"]
    tags = _allowed_tags(cfg)
    key = pd.read_csv(Path(a.scoring_dir) / "KEY_do_not_open_until_scoring_is_done.csv",
                      dtype={"prompt_id": str, "scenario_id": str}, encoding="utf-8-sig")
    sheets = [load_scores(p, dims, allowed_tags=tags) for p in a.sheets]
    if a.headline == "primary" or len(sheets) == 1:
        sc = sheets[0]
    else:
        allsc = pd.concat(sheets)
        sc = allsc.groupby("item_id")[dims + ["total"]].mean().reset_index()
        sc = sc.merge(sheets[0][["item_id", "error_tags"]], on="item_id", how="left")
    scored = key.merge(sc, on="item_id", how="inner")
    missing = len(key) - len(scored)
    outputs = pd.DataFrame([r for r in collect_outputs(cfg)])
    outputs["prompt_id"] = outputs["prompt_id"].astype(str)
    outputs["scenario_id"] = outputs["scenario_id"].astype(str)
    res = out_dir(cfg) / "results"
    res.mkdir(parents=True, exist_ok=True)
    table = score_table(scored, outputs, cfg["analysis"])
    table.to_csv(res / "budget_table.csv", index=False)
    comp = compute_table(out_dir(cfg) / "compute.jsonl")
    comp.to_csv(res / "compute_table.csv", index=False)
    et = error_transfer(scored, cfg["analysis"])
    et.to_csv(res / "error_transfer.csv", index=False)
    plot_budget_curve(table, res / "budget_curve.png")
    kap = None
    if len(sheets) > 1:
        from .scoring import agreement_table
        kap, _ = agreement_table(sheets[0], sheets[1], dims)
        kap.to_csv(res / "interrater_kappa.csv", index=False)
    write_report(cfg, res, table, comp, et, kap, scored, outputs, missing, a.headline)
    print(f"Results in {res}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ishara")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fake", action="store_true", help="use the stand-in model. Tests only")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate the prompt file")
    sp = sub.add_parser("split", help="write the frozen scenario split")
    sp.add_argument("--force", action="store_true")
    sub.add_parser("freeze-rubric")
    sub.add_parser("gpu", help="confirm a GPU runtime")
    sp = sub.add_parser("a0")
    sp.add_argument("--split", choices=["heldout", "pool"], default="heldout")
    sp.add_argument("--seeds", default="0,1,2")
    sp = sub.add_parser("label-sheet", help="export A0 pool outputs for thumbs labelling")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="data/feedback_labels_TO_FILL.csv")
    sp = sub.add_parser("teacher-inputs", help="export prompts for the A4 teacher model")
    sp.add_argument("--samples", type=int, default=3, help="teacher responses per pool prompt")
    for arm in ["a1", "a2", "a3", "a4"]:
        sp = sub.add_parser(arm)
        sp.add_argument("--budgets")
        sp.add_argument("--seeds")
    for name in ["plan", "sheets"]:
        sp = sub.add_parser(name)
        sp.add_argument("--samples-per-prompt", type=int, default=None)
        sp.add_argument("--budgets")
        sp.add_argument("--seeds")
        if name == "sheets":
            sp.add_argument("--coders", default="coder1,coder2")
            sp.add_argument("--salt", required=True, help="any private string. Keep it secret from coders")
            sp.add_argument("--name", default="round1")
    sp = sub.add_parser("kappa")
    sp.add_argument("sheet1")
    sp.add_argument("sheet2")
    sp = sub.add_parser("judge-score")
    sp.add_argument("sheet")
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("judge-agreement")
    sp.add_argument("--human", required=True)
    sp.add_argument("--judge", required=True)
    sp = sub.add_parser("analyze")
    sp.add_argument("--scoring-dir", required=True)
    sp.add_argument("--sheets", nargs="+", required=True, help="primary coder first")
    sp.add_argument("--headline", choices=["primary", "mean"], default="primary")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    fn = {"check": cmd_check, "split": cmd_split, "freeze-rubric": cmd_freeze, "gpu": cmd_gpu,
          "a0": cmd_a0, "label-sheet": cmd_label_sheet, "teacher-inputs": cmd_teacher_inputs, "plan": cmd_plan, "sheets": cmd_sheets,
          "kappa": cmd_kappa, "judge-score": cmd_judge_score, "judge-agreement": cmd_judge_agreement,
          "analyze": cmd_analyze}.get(a.cmd)
    if fn is None:
        a.arm = a.cmd
        fn = cmd_arm
    fn(cfg, a)


if __name__ == "__main__":
    main()
