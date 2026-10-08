"""Builds a throwaway project with invented prompts so the pipeline can be tested end to end.
The prompts are placeholders, not research data."""
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def build(dst):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "configs").mkdir(parents=True)
    (dst / "data").mkdir()
    shutil.copy(ROOT / "configs" / "prompt_fragments.txt", dst / "configs")
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    cfg["project_dir"] = str(dst)
    cfg["model"]["hf_id"] = "fake/model"
    yaml.safe_dump(cfg, open(dst / "configs" / "default.yaml", "w"))
    rub = (ROOT / "rubric.md").read_text().replace("REPLACE", "test anchor")
    (dst / "rubric.md").write_text(rub)
    rows = []
    for s in range(40):
        for lang in ["english", "roman_urdu"]:
            for ctx in ["explicit", "implicit"]:
                base = f"test scenario {s} request" if lang == "english" else f"yaar scenario {s} kya karna hai"
                extra = " with background" if ctx == "explicit" else ""
                rows.append({"prompt_id": f"s{s:02d}_{lang[:2]}_{ctx[:3]}", "scenario_id": f"s{s:02d}",
                             "language": lang, "context": ctx, "prompt": base + extra,
                             "reference": f"meaning {s}"})
    pd.DataFrame(rows).to_csv(dst / "data" / "prompts.csv", index=False)
    return dst


if __name__ == "__main__":
    print(build(sys.argv[1]))
