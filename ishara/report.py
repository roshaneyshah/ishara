"""Writes results.md. Every number in it comes from the tables this script also saves."""
from pathlib import Path

from .util import out_dir, read_json


def _f(x, nd=2):
    try:
        if x != x:
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _fallbacks(cfg):
    rows = []
    for m in sorted(Path(out_dir(cfg)).glob("*/*/manifest.json")):
        j = read_json(m)
        if j.get("fallback_to_base"):
            rows.append(f"{j['arm']} budget {j['budget']} seed {j['seed']}")
    return rows


def write_report(cfg, res, table, comp, et, kap, scored, outputs, missing, headline):
    lvl = int(cfg["analysis"]["ci"] * 100)
    L = []
    L.append("# ISHARA results\n")
    L.append(f"Model: {cfg['model']['hf_id']} at revision {cfg['model']['revision']}. "
             f"max_new_tokens {cfg['generation']['max_new_tokens']}, {cfg['generation']['n_samples']} samples per prompt.\n")
    L.append(f"Held-out scenarios: {outputs['scenario_id'].nunique()}. Held-out prompts: {outputs['prompt_id'].nunique()}. "
             f"Scored items: {len(scored)}. Items in the key without a score: {missing}.\n")
    L.append(f"Intervals are {lvl} percent cluster bootstrap intervals over held-out scenarios "
             f"({cfg['analysis']['n_boot']} resamples). Headline scores use the "
             f"{'primary coder' if headline == 'primary' else 'mean of coders'}.\n")
    L.append("Budgets for A1, A2 and A3 count feedback signals. Budgets for A4 count synthetic examples.\n")
    L.append("\n## Held-out score by arm and budget\n")
    L.append(f"| Cell | n | Mean total | {lvl}% CI | Seed SD | Gain vs A0 | {lvl}% CI | Gain per signal |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        g = getattr(r, "gain_vs_A0", float("nan"))
        L.append(f"| {r.cell} | {r.n_scored} | {_f(r.mean_total)} | {_f(r.ci_low)} to {_f(r.ci_high)} | "
                 f"{_f(r.seed_sd)} | {_f(g)} | {_f(getattr(r, 'gain_ci_low', float('nan')))} to "
                 f"{_f(getattr(r, 'gain_ci_high', float('nan')))} | {_f(getattr(r, 'gain_per_signal', float('nan')), 4)} |")
    L.append("\n## Language gap (English minus Roman Urdu)\n")
    L.append(f"| Cell | English | Roman Urdu | Gap | {lvl}% CI | Change vs A0 | {lvl}% CI |")
    L.append("|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        L.append(f"| {r.cell} | {_f(r.mean_english)} | {_f(r.mean_roman_urdu)} | {_f(r.gap_en_minus_ur)} | "
                 f"{_f(r.gap_ci_low)} to {_f(r.gap_ci_high)} | {_f(getattr(r, 'gap_change_vs_A0', float('nan')))} | "
                 f"{_f(getattr(r, 'gap_change_ci_low', float('nan')))} to {_f(getattr(r, 'gap_change_ci_high', float('nan')))} |")
    L.append("\n## Truncation rate (all generated outputs, scored or not)\n")
    L.append(f"| Cell | Rate | {lvl}% CI |")
    L.append("|---|---|---|")
    for r in table.itertuples():
        L.append(f"| {r.cell} | {_f(r.truncation_rate, 3)} | {_f(r.trunc_ci_low, 3)} to {_f(r.trunc_ci_high, 3)} |")
    if len(comp):
        L.append("\n## Compute (GPU minutes)\n")
        cols = [c for c in comp.columns if c.endswith("_per_seed") or c == "total_gpu_min_all_seeds"]
        L.append("| Arm | Budget | Seeds | " + " | ".join(cols) + " |")
        L.append("|---|---|---|" + "---|" * len(cols))
        for r in comp.itertuples(index=False):
            d = r._asdict()
            L.append(f"| {d['arm']} | {d['budget']} | {d['n_seeds']} | " + " | ".join(_f(d[c]) for c in cols) + " |")
    if len(et):
        L.append("\n## Teacher error transfer (RQ3)\n")
        L.append("Share of outputs on held-out prompts where the teacher made a tagged error that repeat at "
                 "least one of the same tags. A0 is the base rate.\n")
        L.append(f"| Cell | n | Prompts | Rate | {lvl}% CI |")
        L.append("|---|---|---|---|---|")
        for r in et.itertuples():
            L.append(f"| {r.cell} | {r.n} | {r.n_prompts_with_teacher_errors} | {_f(r.transfer_rate, 3)} | "
                     f"{_f(r.ci_low, 3)} to {_f(r.ci_high, 3)} |")
    if kap is not None:
        L.append("\n## Inter-rater agreement\n")
        L.append("| Measure | n | Quadratic weighted kappa | Exact agreement |")
        L.append("|---|---|---|---|")
        for r in kap.itertuples():
            L.append(f"| {r.measure} | {r.n} | {_f(r.weighted_kappa, 3)} | {_f(r.exact_agreement, 3)} |")
    jt = Path(res) / "judge_vs_human_thumbs.json"
    if jt.exists():
        j = read_json(jt)
        L.append("\n## Judge agreement with human scores\n")
        L.append(f"Thumbs agreement on {j['n']} items: accuracy {_f(j['accuracy'], 3)}, Cohen kappa "
                 f"{_f(j['cohen_kappa'], 3)}. Judge parse failures: {j['judge_parse_failures']}. "
                 f"Per-dimension kappa is in judge_vs_human_kappa.csv.\n")
    fb = _fallbacks(cfg)
    L.append("\n## Runs with no usable training signal\n")
    L.append("These runs had no thumbs-up signal in their budget, so the arm ran the base model.\n")
    L.extend([f"- {x}" for x in fb] or ["- none"])
    L.append("\n## Limitations\n")
    L.append("- Coders were blind to arm and budget. The prompt text itself shows the language and "
             "whether context is explicit.")
    L.append("- Small sample. Read differences as effect sizes with intervals, not as significance tests.")
    (Path(res) / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
