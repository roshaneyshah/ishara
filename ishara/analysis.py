"""Budget curves, gain per signal, language gap, truncation, compute cost and error transfer.
Intervals come from a cluster bootstrap over held-out scenarios. The same resamples are used for
every cell, so differences between cells are paired."""
from pathlib import Path

import numpy as np
import pandas as pd

from .util import read_jsonl


class ScenarioBootstrap:
    def __init__(self, scenarios, n_boot, seed):
        self.scen = sorted(scenarios)
        self.idx = {s: i for i, s in enumerate(self.scen)}
        rng = np.random.default_rng(seed)
        k = len(self.scen)
        self.W = rng.multinomial(k, [1.0 / k] * k, size=n_boot).astype(float)  # (n_boot, k)

    def sums(self, df, col):
        s = np.zeros(len(self.scen))
        c = np.zeros(len(self.scen))
        for sid, g in df.groupby("scenario_id"):
            s[self.idx[sid]] += g[col].sum()
            c[self.idx[sid]] += len(g)
        return s, c

    def mean_draws(self, df, col):
        s, c = self.sums(df, col)
        den = self.W @ c
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.W @ s) / den


def ci(draws, level):
    d = np.asarray(draws)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def cell_label(arm, budget):
    return f"{arm}" if arm == "A0" else f"{arm}@{budget}"


def score_table(scored, outputs, acfg):
    """scored: one row per scored item with arm, budget, seed, scenario_id, language, total.
    outputs: every generated held-out output, for truncation rates."""
    level = acfg["ci"]
    boot = ScenarioBootstrap(outputs["scenario_id"].unique(), acfg["n_boot"], acfg["boot_seed"])
    cells = sorted({(a, b) for a, b in zip(scored["arm"], scored["budget"]) if a != "TEACHER"},
                   key=lambda x: (x[0], x[1]))
    a0 = scored[scored["arm"] == "A0"]
    a0_draws = boot.mean_draws(a0, "total") if len(a0) else None
    a0_gap_draws = None
    if len(a0):
        a0_gap_draws = (boot.mean_draws(a0[a0.language == "english"], "total")
                        - boot.mean_draws(a0[a0.language == "roman_urdu"], "total"))
    rows = []
    for arm, budget in cells:
        g = scored[(scored.arm == arm) & (scored.budget == budget)]
        o = outputs[(outputs.arm == arm) & (outputs.budget == budget)]
        draws = boot.mean_draws(g, "total")
        seed_means = g.groupby("seed")["total"].mean()
        en, ur = g[g.language == "english"], g[g.language == "roman_urdu"]
        gap_draws = boot.mean_draws(en, "total") - boot.mean_draws(ur, "total")
        tr_draws = boot.mean_draws(o.assign(tr=o["truncated"].astype(float)), "tr")
        row = {"cell": cell_label(arm, budget), "arm": arm, "budget": budget,
               "n_scored": len(g), "n_scenarios": g["scenario_id"].nunique(),
               "n_seeds": int(g["seed"].nunique()),
               "mean_total": g["total"].mean(), "ci_low": ci(draws, level)[0], "ci_high": ci(draws, level)[1],
               "seed_sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else float("nan"),
               "mean_english": en["total"].mean(), "mean_roman_urdu": ur["total"].mean(),
               "gap_en_minus_ur": en["total"].mean() - ur["total"].mean(),
               "gap_ci_low": ci(gap_draws, level)[0], "gap_ci_high": ci(gap_draws, level)[1],
               "truncation_rate": float(o["truncated"].mean()) if len(o) else float("nan"),
               "trunc_ci_low": ci(tr_draws, level)[0], "trunc_ci_high": ci(tr_draws, level)[1]}
        if arm != "A0" and a0_draws is not None:
            diff = draws - a0_draws
            row["gain_vs_A0"] = row["mean_total"] - a0["total"].mean()
            row["gain_ci_low"], row["gain_ci_high"] = ci(diff, level)
            row["gain_per_signal"] = row["gain_vs_A0"] / budget
            row["gps_ci_low"], row["gps_ci_high"] = ci(diff / budget, level)
            gd = gap_draws - a0_gap_draws
            row["gap_change_vs_A0"] = row["gap_en_minus_ur"] - (
                a0[a0.language == "english"]["total"].mean() - a0[a0.language == "roman_urdu"]["total"].mean())
            row["gap_change_ci_low"], row["gap_change_ci_high"] = ci(gd, level)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_table(compute_path):
    if not Path(compute_path).exists():
        return pd.DataFrame()
    c = pd.DataFrame(read_jsonl(compute_path))
    per_run = c.groupby(["arm", "budget", "seed", "phase"])["secs"].sum().reset_index()
    wide = per_run.pivot_table(index=["arm", "budget", "seed"], columns="phase", values="secs",
                               fill_value=0.0).reset_index()
    phases = [p for p in ["search", "train", "generate"] if p in wide.columns]
    wide["total_secs"] = wide[phases].sum(axis=1)
    agg = wide.groupby(["arm", "budget"]).agg(
        n_seeds=("seed", "nunique"),
        **{f"{p}_gpu_min_per_seed": (p, lambda x: x.mean() / 60) for p in phases},
        total_gpu_min_per_seed=("total_secs", lambda x: x.mean() / 60),
        total_gpu_min_all_seeds=("total_secs", lambda x: x.sum() / 60)).reset_index()
    return agg


def _tags(s):
    return {t.strip() for t in str(s).split("|") if t.strip()}


def error_transfer(scored, acfg, arms=("A0", "A3", "A4")):
    """For held-out prompts where the teacher made tagged errors, the share of an arm's outputs on
    those prompts that repeat at least one of the same error tags. A0 gives the base rate."""
    t = scored[scored.arm == "TEACHER"]
    if t.empty:
        return pd.DataFrame()
    teacher_tags = {}
    for r in t.itertuples():
        teacher_tags.setdefault(r.prompt_id, set()).update(_tags(r.error_tags))
    teacher_tags = {p: s for p, s in teacher_tags.items() if s}
    if not teacher_tags:
        return pd.DataFrame()
    sub = scored[scored.prompt_id.isin(teacher_tags) & scored.arm.isin(arms)].copy()
    sub["repeats_teacher"] = [float(bool(_tags(r.error_tags) & teacher_tags[r.prompt_id]))
                              for r in sub.itertuples()]
    boot = ScenarioBootstrap(scored["scenario_id"].unique(), acfg["n_boot"], acfg["boot_seed"])
    rows = []
    for (arm, budget), g in sub.groupby(["arm", "budget"]):
        d = boot.mean_draws(g, "repeats_teacher")
        lo, hi = ci(d, acfg["ci"])
        rows.append({"cell": cell_label(arm, budget), "arm": arm, "budget": budget, "n": len(g),
                     "n_prompts_with_teacher_errors": g["prompt_id"].nunique(),
                     "transfer_rate": g["repeats_teacher"].mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def plot_budget_curve(table, path, ylabel="Held-out total score (0 to 8)", col="mean_total",
                      lo="ci_low", hi="ci_high"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A1": "#1f77b4", "A2": "#ff7f0e", "A3": "#2ca02c", "A4": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    a0 = table[table.arm == "A0"]
    if len(a0):
        r = a0.iloc[0]
        ax.axhline(r[col], color="gray", lw=1.2, label="A0 baseline")
        ax.axhspan(r[lo], r[hi], color="gray", alpha=0.15, lw=0)
    for arm in ["A1", "A2", "A3", "A4"]:
        t = table[table.arm == arm].sort_values("budget")
        if t.empty:
            continue
        label = "A4 (x = synthetic examples)" if arm == "A4" else arm
        ax.plot(t.budget, t[col], marker="o", color=colors[arm], label=label)
        ax.fill_between(t.budget, t[lo], t[hi], color=colors[arm], alpha=0.15, lw=0)
    ax.set_xscale("log")
    budgets = sorted(table.loc[table.arm != "A0", "budget"].unique())
    if budgets:
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Budget (feedback signals, or synthetic examples for A4)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
