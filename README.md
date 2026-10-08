# ISHARA

Feedback-efficient adaptation of small language models to Roman Urdu pragmatics.

This repository holds the adaptation code, the design checks and the analysis for the ISHARA project. Every number in the write-up comes from `python -m ishara analyze`.

## Setup

Colab or Kaggle with a T4 GPU. Clone the repository into your Drive, open `notebooks/ishara_colab.ipynb` and run the cells in order. On Kaggle, use `/kaggle/working/ishara` as the project folder.

Before the first run, edit `configs/default.yaml`:

- `model.hf_id`: the exact MiniCPM5-2B repository id. The revision is already pinned.
- `data.columns`: your column names, if they differ from the defaults.
- `judge.hf_id` and `judge.revision`: only needed for A2.

## Prompt file

`data/prompts.csv` with one row per prompt.

| Column | Values |
|---|---|
| prompt_id | unique id |
| scenario_id | shared by the four prompts of a scenario |
| language | english or roman_urdu |
| context | explicit or implicit |
| prompt | the text sent to the model |
| reference | optional. What the speaker meant. Shown to coders and the judge |

`python -m ishara check` enforces one prompt per cell for every scenario, and flags any English prompt that contains a word from the Roman Urdu list in `ishara/lexicon.py`. Flags are for you to check by hand.

## Workflow

| Week | Command | Output |
|---|---|---|
| 1 | `check`, then `split`, then `freeze-rubric` | `data/split.json`, `rubric.lock.json` |
| 1 | `a0 --split heldout`, `a0 --split pool --seeds 0`, `label-sheet` | baseline outputs and the sheet to label |
| 2 | `a1`, `a2` | gradient-free arms |
| 3 | `teacher-inputs`, then `a3`, `a4` | LoRA arms |
| 4 on | `plan`, `sheets`, `kappa`, `judge-score`, `judge-agreement`, `analyze` | `runs/primary/results/` |

Run any command as `python -m ishara <command>`. Add `--config configs/other_model.yaml` for the second and third models. Give each model its own `model.name` so outputs stay separate.

After `split`, the split is frozen. Every later command checks that the prompt file has not changed since. After `freeze-rubric`, arms A1 to A4 refuse to start if `rubric.md` changes.

## How each arm uses the budget

| Arm | Budget unit | What it does |
|---|---|---|
| A0 | none | base model |
| A1 | labelled signals | picks up to 3 thumbs-up examples per held-out prompt by character n-gram similarity |
| A2 | oracle calls | evolutionary search over subsets of `configs/prompt_fragments.txt`. Each output shown to the oracle costs one signal |
| A3 | labelled signals | LoRA on the thumbs-up outputs in the budget |
| A4 | synthetic examples | LoRA on teacher outputs, same hyperparameters as A3 |

Budgets are nested. For a given seed, the 10-signal set sits inside the 25-signal set, and so on up to 200.

## Decisions in the code for you to confirm

1. A1 and A3 learn from thumbs-up signals only. Thumbs-down signals count against the budget and are otherwise unused.
2. A2 cannot reuse the labelled A0 signals, since each candidate prompt produces new outputs. Its feedback comes from the judge, or from you if `a2.oracle` is `human`. Its signals differ in source from A1 and A3, and the paper should say so.
3. When a budget holds no thumbs-up signal, A1 and A3 fall back to the base model. These runs are listed in `results.md`.
4. Sampling uses temperature 0.7 and top_p 0.9 to get several samples per prompt. Change this in the config if the first study used other settings.
5. Held-out fraction is 0.4. With 40 scenarios that gives 16 held-out scenarios (64 prompts) and 24 pool scenarios (96 prompts).
6. Headline scores use the first coder. The second coder feeds the kappa. Use `--headline mean` if both coders score everything.
7. The error tags in `rubric.md` are a draft. Edit them before freezing.
8. The judge counts thumbs up at a total of 6 or more. This is `a2.judge_thumbs_threshold`.

## Scoring load

With one sample per prompt, all arms, all five budgets and three seeds, each coder scores 4,032 items (64 held-out prompts times 63 runs), plus 64 teacher items. `python -m ishara plan` prints the exact count for any subset. Use `--seeds` and `--budgets` on `sheets` to score a planned subset.

## Intervals

All intervals are cluster bootstrap intervals over held-out scenarios. The same resamples are used for every cell, so gains against A0 are paired.

## Tests

`python -m pytest tests -q` runs the full pipeline with a stand-in model, and runs the real transformers and peft code on CPU with a tiny random model. Tested with transformers 5.17.0 and peft 0.21.0. The MiniCPM5-2B path itself has not been run, since that needs the GPU.
