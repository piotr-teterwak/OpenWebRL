# WebVoyager and DeepShop evals (tillicum)

These run the same actor-only browser eval as Online-Mind2Web (`openwebrl/arm_eval.py
--mode baseline`), with each benchmark's own task file and judge. The protocol is ours, not
the published one. It is meant for comparing checkpoints against base, not against numbers
in papers (see [Caveats](#caveats)).

## Running

```bash
BENCH=webvoyager MODELS=base,rpo-matched-lr5e-5 PASS=bench1 \
  sbatch --gpus=h200:2 --cpus-per-task=16 --mem=240G --time=06:00:00 --exclude=g013,g021 \
  tillicum/eval_concurrent.sbatch
BENCH=deepshop MODELS=base,rpo-matched-lr5e-5 PASS=bench1 \
  sbatch --gpus=h200:2 --cpus-per-task=16 --mem=240G --time=03:00:00 --exclude=g013,g021 \
  tillicum/eval_concurrent.sbatch
```

- **Models.** Use one GPU per model in `MODELS`, and always include `base` in the same job:
  site behaviour changes over hours (see [blocking](#caveats)). The labels are defined in
  the `SPECS` list of `tillicum/eval_concurrent.sbatch`. Unmerged LoRA runs are merged on
  first use.
- **Smoke test.** First run `SMOKE=1 SMOKE_TASKS=0,40 PASS=smoke-bench sbatch --qos=debug
  --gpus=h200:1 --cpus-per-task=8 --mem=120G --time=00:40:00`. That is 2 tasks, base only,
  and checks the server, browsers, judge and API key.
- **Resuming.** A job that times out resumes on resubmission with the same `PASS`.
- **Judge key.** The judge needs `OPENAI_API_KEY` in `.env`.

| | WebVoyager | DeepShop |
|---|---|---|
| Tasks | `openwebrl/data/eval/webvoyager_fara.jsonl`, 595 tasks over 15 sites | `openwebrl/data/eval/deepshop.jsonl`, 150 tasks, all Amazon |
| Judge | `openwebrl/eval/reward_webvoyager.py`, GPT-4o | `openwebrl/eval/reward_deepshop.py`, GPT-4o, structured verdict |
| `arm_eval.py --protocol` | `webvoyager` | `deepshop` |
| Wall time, 1 H200 per model, 12 browsers | ~2 h | ~50 min |

Shared settings: temperature 0.7, top-p 0.9, 1,024 new tokens, 30 steps, seed 42.

DeepShop rows are flat. `run_evaluate.load_tasks_from_jsonl` passes the actor only the
`task_name`, `website` and `task_id` fields, never `evaluator_reference` or `definite_answer`.

## Output

Results go to `/gpfs/scrubbed/piotrt/owrl/eval/pass<PASS>/<label>/<bench>/`:

- `summary.json`: `success_rate_all_scheduled` (invalid episodes count as failures; this is the headline) and `success_rate_valid`
- `results/*.json`: one file per episode, with the judge verdict and reasoning in `metadata.reward`, plus the messages and final screenshot
- `manifest.json`: model, judge, sampling settings and task-file hash
- `../eval-<bench>.log`, `../actor-server-<bench>.log`: run logs

An episode is invalid if it produced no trajectory (the site failed to load), was aborted by
an environment or server error, or the judge call failed.

`/gpfs/scrubbed` is purged after 60 days of inactivity, so copy anything you need to keep.

## Results so far (2026-10-22)

Base OpenWebRL-4B-SFT vs matched-pair RPO (`rpo-matched-lr5e-5`, seed 42), in pass `bench1`.

| | Base: overall / valid only | Matched RPO: overall / valid only | Paired on tasks valid in both | Paired on tasks valid and not blocked in both |
|---|---|---|---|---|
| WebVoyager | 43.9 / 44.3 | 44.4 / 46.4 | +1.8 pts, 87 wins / 77 losses, p = 0.48 (n = 565) | +5.0 pts, 69 / 48, p = 0.064 (n = 417) |
| DeepShop | 40.0 / 40.5 | 44.0 / 45.5 | +4.2 pts, 24 / 18, p = 0.44 (n = 143) | +3.9 pts, 19 / 14, p = 0.49 (n = 128) |

All p-values are two-sided sign tests on the tasks one model solved and the other didn't.

Seed 42 is the strongest of three matched-RPO seeds on Online-Mind2Web: about +10 points
over base there, against +0–4 for seeds 1 and 2. So these transfer numbers are probably an
optimistic estimate.

## Caveats

- **Site blocking.** About 20% of WebVoyager episodes hit a block: 403 errors, CAPTCHAs, or
  "access denied". Cambridge Dictionary (~70% of its tasks) and Allrecipes (~45%) block both
  models. Coursera started returning 403 partway through `bench1` and hit the second model
  much harder (27/40 vs 7/40). The agent then ends the episode normally, so these episodes
  count as *valid failures*. Compare models on tasks unblocked in both, as in the table
  above. The block filter is a keyword match on the tool responses and final answer (403,
  blocked, captcha, access denied, unusual traffic, robot), so treat it as approximate.
  Zixian's earlier WebVoyager runs used a stealth browser (Browser Use) to reduce blocking;
  this eval does not.
- **The judge sees one screenshot.** The judges attach up to 30 screenshots, but each
  turn-level sample stores only its own, so GPT-4o judges from the final screenshot plus
  the answer. This applies to both models equally but differs from the published protocols.
- **Response cut-offs.** The 1,024-token cap cut off 57 of matched RPO's WebVoyager episodes
  against 9 for base. Most were long reasoning with no tool call.
- **Overlap with training.** All 15 WebVoyager sites appear in the C2 training pool
  (`webgym_filtered_popular`), and 3–4 WebVoyager tasks are near-duplicates of training
  tasks (word-overlap Jaccard ≥ 0.8). DeepShop has no near-duplicates (max 0.30).
- **Judges.** GPT-4o is still served. Online-Mind2Web's o4-mini judge was retired in
  2026-10 and replaced by gpt-5.6-terra (`scripts/rejudge_om2w.py` re-scores old OM2W runs).
  WebVoyager and DeepShop are unaffected.
