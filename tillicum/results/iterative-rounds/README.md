# Iterative RPO with a GPT-6 Sol teacher: SFT vs Round 1 vs Round 2 vs data-matched single round

Branch `piotr_arm`, Tillicum (UW Hyak, H200), November 2026. This README documents how the four models in the table
below were produced and evaluated: task split, data collection, labelling, pair construction, training and
evaluation, with the exact scripts and knobs. Paths under `/gpfs/scrubbed/` are scratch (60-day purge); see
[Artifacts](#artifacts) for what is backed up where.

## Result

Valid-only success = successes / valid episodes; overall = successes / all episodes (invalid counted as failure).
All four models were evaluated side by side in the same two eval passes (`s2-a`, `s2-b`; 600 OM2W + 300 DeepShop
episodes per model), T 1.0.

| model | OM2W valid-only | DeepShop valid-only | pooled valid-only | pooled overall (invalid = failure) |
|---|---|---|---|---|
| SFT | 33.5 | 30.7 | 32.5 | 29.9 |
| Round 1 | 39.6 | 36.1 | 38.4 | 34.2 |
| Round 2 | 33.8 | 37.4 | 35.1 | 30.6 |
| Data-matched single round | 42.1 | 37.4 | 40.4 | 36.2 |

Paired differences, pooled over both benchmarks, valid-only, task-clustered bootstrap (tasks resampled with all
their passes; 10k resamples; `tillicum/iter_pooled.py`, output `tillicum/results/luna-iter/iter-pooled.txt`):

| comparison | difference [95% CI] | passes |
|---|---|---|
| Round 1 − SFT | **+4.9 [+1.4, +8.4]** | 2 |
| Data-matched single round − SFT | **+7.7 [+4.1, +11.4]** | 2 |
| Round 2 − SFT | +2.3 [−1.0, +5.7] | 2 |
| Data-matched single round − Round 1 | +2.1 [−1.5, +5.6] | 2 |
| Round 2 − Round 1 | **−3.1 [−5.6, −0.5]** | 4 (`s2-a`, `s2-b`, `rs-a`, `rs-b`) |
| Round 2 − Data-matched single round | **−5.7 [−9.2, −2.1]** | 2 |

Round 1 vs SFT also reproduces on earlier, independent passes: +5.7 [+3.2, +8.2] pooled over 4 passes
(`tillicum/results/luna-iter/RESULTS.md`).

## Models at a glance

| table name | run name | start model | DPO reference | training pairs | half-B candidates from |
|---|---|---|---|---|---|
| SFT | `base` | — (OpenWebRL-4B-SFT) | — | — | — |
| Round 1 | `r2-sol-r1-all-m0.3` ("C2") | SFT | SFT | 5,304 (half A) | — |
| Round 2 | `r2-sol2-r2` | Round 1 (merged) | Round 1 | 6,250 (half B) | Round 1 |
| Data-matched single round | `r2-sol2-ctrl` | SFT | SFT | 10,690 (half A + half B) | SFT |

Round 2 and the data-matched single round see the same half-B tasks, collected the same way (Sol-steered, same
labelling and pair rules). They differ in which actor generated the half-B candidates, and in sequential vs joint
training. In total, both have seen Round 1's half-A pairs.

## Common setup

**Actor.** `OpenWebRL-4B-SFT` (Qwen3-VL-4B), mirror copy at
`/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT`.

**Task pool and split.** This is Zixian's deduplicated C2 collection pool (`c2-full-282782`, 2,091 tasks). 104 tasks
that overlap validation were dropped, leaving 1,987, which were split by host (seed 0) into half A (994) and half B
(993). No task is in both halves; 29 hosts appear in both. Report: `tillicum/results/luna-iter/split-report.json`.
Files: `/gpfs/scrubbed/piotrt/owrl/c2-collect/luna-iter/tasks-{A,B}{,-shard1,-shard2}.jsonl` (each half is in 2 shards
of about 497 tasks).

**Collection harness.** `tillicum/c2_collect.sbatch` → `openwebrl/arm_c2.py`. At each step the actor samples 5
candidates (T 0.7, top-p 0.9, 1,024 new tokens). A selector picks one **before execution**, and that pick is executed.
Episode outcomes are judged by `gpt-5.6-terra` (`JUDGE`). Each job uses 1 H200 (actor sglang server + API selector),
16 browsers, seeds 42 and 43, 2 shards per half, so 4 jobs per collection.

**Teacher.** `gpt-6-sol` through `scripts/serve_sol_selector.py`, with the pinned ARM selection prompt
(screenshot, task, URL, history, the 5 candidates). With `--probabilities` (`TEACHER_PROBS=1` / `PROBS=1`), Sol also
returns a probability per candidate. That is the "probability prompt" used for every model here.

**Pair construction.** `tillicum/build_round2_data.sbatch` → `scripts/prepare_arm_joint_data.py --relabel`.
- **Chosen:** Sol's pick at a state.
- **Rejected:** another candidate at the same state with a *different* action and the same number of tool calls. With
  `MARGIN=0.3`, the rejected candidate must be at least 0.3 below the chosen one in Sol's probability; the most
  probable such candidate is used, with ties broken by hash order.
- **States:** every valid state with ≥ 2 distinct candidate actions, from **successful and failed episodes**
  (`ALLOW_FAILED=1`). Pairs from failed episodes carry `terminal_success=false`.
- **Validation:** the validation set is `joint-data/matched`'s (658 pairs) for every dataset, so held-out metrics are
  comparable across runs.
- **Combining datasets:** `tillicum/build_combined_data.sbatch` concatenates datasets (ids are made distinct with
  `IDPREFIX`).

**Training recipe (RPO), identical for all three trained models.** `tillicum/train_joint.sbatch` →
`scripts/train_arm_joint_ddp.py`.
- **Objective:** DPO, β 0.1, plus RPO's length-normalised NLL on the chosen response, weight 1.0
  (`OBJECTIVE=dpo BETA=0.1 SFT_WEIGHT=1`). The NLL term applies only to pairs from successful episodes
  (`SFT_SUCCESS_ONLY=1`); failed-episode pairs get the DPO term only.
- **Adapter:** LoRA r16, α 32, dropout 0.05.
- **Optimisation:** lr 5e-5 (`LR=5e-5`), warmup over 512 examples, decaying to 0.5× by the end. Batch 32
  (2 H200 × 16 gradient accumulation), 2 epochs (`EPOCHS=2`), seed 42, max 32,768 tokens.
- **Reference scores:** cached in a pre-pass. `OWRL_ACTOR` is the start model; `OWRL_REFERENCE` is the DPO reference
  (defaults to the start model).
- **Frozen configs:** each run's config is frozen at first submission (hashes of trainer, data and model) in
  `/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs/<run>/frozen-config.json`. Copies are in `tillicum/results/luna-iter/`.
- **Gate:** after training, `tillicum/gate.sbatch` checks the final checkpoint's generation length-cap rate (must be
  ≤ 0.05) before evaluation.

**Evaluation.** `tillicum/eval_concurrent.sbatch` (one H200 per model, all models of a pass in one job) →
`openwebrl/arm_eval.py --mode baseline`: the model acts alone, with no selector.
- **Protocol:** EVAL.md T 1.0 (T 1.0, top-p 0.9, 1,024 new tokens, 30 steps, seed 42, 12 browsers per model, 1,800 s
  task timeout).
- **Judges:** Online-Mind2Web (300 tasks) with its canonical judge; DeepShop (150 tasks, all Amazon) with gpt-4o
  (30 screenshots).
- **Checkpoints:** final checkpoints are merged into `/gpfs/scrubbed/piotrt/owrl/eval/merged/<run>`.
- **Results:** `/gpfs/scrubbed/piotrt/owrl/eval/pass<PASS>/<label>/<bench>/results/*.json`; per-pass summaries are
  copied to `tillicum/results/luna-iter/eval.<pass>.<label>.<bench>.summary.json`.

## Round 1 (`r2-sol-r1-all-m0.3`, "C2")

Data: `/gpfs/scrubbed/piotrt/owrl/joint-data/sol-Aall-m0.3`. 5,304 train pairs: 2,044 from successful episodes and
3,260 from failed episodes.

1. **Collect half A, Luna-steered** (`tillicum/luna_iter_pipeline.sh`, jobs in `tillicum/results/luna-iter/round1-jobs.txt`):
   ```
   ACTOR=<SFT> TEACHER=luna SEED={42,43} NAME=luna-A-s{42,43}-shard{1,2} PARALLEL=16 \
     TASKS=.../luna-iter/tasks-A-shard{1,2}.jsonl LUNA_MAX_USD=25 \
     sbatch -A vlm --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 tillicum/c2_collect.sbatch
   ```
   `gpt-6-luna` (reasoning effort high) chose the executed candidate. These collections provide the states and the 5
   SFT candidates at each state.
2. **Export states.**
   - States from successful episodes come from the collections' `training.jsonl`.
   - States from failed episodes come from `scripts/export_failed_turns.py` (`luna-Afail-s{42,43}`).
   - Only states with ≥ 2 distinct candidate actions are kept:
     `/gpfs/scrubbed/piotrt/owrl/c2-collect/sol-relabel/{A,Afail}-s{42,43}-distinct.jsonl`.
3. **Relabel with Sol** (`tillicum/sol_relabel_pipeline.sh` for successful states, `tillicum/sol_relabel_c2.sh` for failed):
   ```
   MODEL=gpt-6-sol PROBS=1 DATASET=.../sol-relabel/<A|Afail>-s<seed>-distinct.jsonl OUT=.../sol-relabel/<...>/labels.jsonl \
     sbatch -A vlm tillicum/relabel_luna.sbatch
   ```
   Sol re-picks the preferred candidate at every stored state, with a probability per candidate. The labels are
   pre-execution, so they are valid at failed-episode states too.
4. **Build pairs.** For each seed, run `build_round2_data.sbatch` with `RELABEL=<Sol labels> RELABEL_NAME=gpt-6-sol
   MARGIN=0.3` (adding `ALLOW_FAILED=1` for failed states). Then combine:
   `sol-A-m0.3` (2,044) + `sol-Afail-m0.3` (3,260) → `sol-Aall-m0.3`.
5. **Train:**
   ```
   OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 \
     OWRL_ACTOR=<SFT> OWRL_JOINT_DATA=.../joint-data/sol-Aall-m0.3 RUN=r2-sol-r1-all-m0.3 \
     sbatch -A vlm --time=14:00:00 tillicum/train_joint.sbatch
   ```
   332 updates.

Note: half A was **steered by Luna and relabelled by Sol**, which differs from half B below (Sol both steered and
labelled).

## Half-B collections (shared by Round 2 and the data-matched single round)

`tillicum/sol_round2.sh` (run with `FORCE_RECIPE=prob`, i.e. Round 1's probability prompt + margin 0.3). Two
collections on the same half-B tasks, identical apart from the actor:

```
ACTOR=<actor> TEACHER=sol TEACHER_PROBS=1 SKIP_IDENTICAL=1 SEED={42,43} NAME=<prefix>-s{42,43}-shard{1,2} PARALLEL=16 \
  TASKS=.../luna-iter/tasks-B-shard{1,2}.jsonl LUNA_MAX_USD=70 \
  sbatch -A vlm --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 tillicum/c2_collect.sbatch
```

| collection | actor generating candidates | pairs |
|---|---|---|
| `sol2-Br1` | Round 1 (merged `r2-sol-r1-all-m0.3`) | 6,250 (2,254 successful / 3,996 failed) |
| `sol2-Bsft` | SFT | 5,386 (2,172 successful / 3,214 failed) |

- **Steering:** Sol steers the episodes. Its executed pre-execution pick *is* the label, so there's no separate
  relabelling. `scripts/teacher_labels_from_selections.py --executed` extracts the picks with their probabilities.
- **Skipped steps:** `SKIP_IDENTICAL=1` skips the Sol call when all 5 candidates share one action; such states give no
  pairs.
- **States:** all valid states, successful and failed episodes (`scripts/export_failed_turns.py --include-successful`).
- **Pairs:** `build_round2_data.sbatch` with `RELABEL=<executed labels> RELABEL_NAME=gpt-6-sol ALLOW_FAILED=1
  MARGIN=0.3`, then the two seeds combined.

## Round 2 (`r2-sol2-r2`)

Data: `joint-data/sol2-Br1`, 6,250 pairs (half B, Round 1's candidates).

```
OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 \
  OWRL_ACTOR=/gpfs/scrubbed/piotrt/owrl/eval/merged/r2-sol-r1-all-m0.3 \
  OWRL_REFERENCE=/gpfs/scrubbed/piotrt/owrl/eval/merged/r2-sol-r1-all-m0.3 \
  OWRL_JOINT_DATA=.../joint-data/sol2-Br1 RUN=r2-sol2-r2 sbatch -A vlm --time=14:00:00 tillicum/train_joint.sbatch
```

- **Setup:** starts from merged Round 1, with a fresh LoRA adapter and a fresh optimizer. The learning-rate schedule
  restarts (warmup back to 5e-5). The reference is Round 1. 391 updates.
- **No replay:** half-A pairs are not seen again.

## Data-matched single round (`r2-sol2-ctrl`)

Data: `joint-data/sol2-ctrl` = `sol-Aall-m0.3` (Round 1's 5,304 pairs) + `sol2-Bsft` (5,386) = 10,690 pairs
(4,216 successful / 6,474 failed), via `build_combined_data.sbatch`.

```
OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 \
  OWRL_ACTOR=<SFT> OWRL_REFERENCE=<SFT> OWRL_JOINT_DATA=.../joint-data/sol2-ctrl RUN=r2-sol2-ctrl \
  sbatch -A vlm --time=14:00:00 tillicum/train_joint.sbatch
```

One round from SFT on both halves mixed; 669 updates. Total compute is similar to Round 1 + Round 2 (332 + 391 = 723
updates).

## Evaluation of the table

```
MODELS=base,r2-sol-r1-all-m0.3,r2-sol2-r2,r2-sol2-ctrl PASS={s2-a,s2-b} BENCH={online-mind2web,deepshop} TEMP=1.0 \
  sbatch -A vlm --gpus=h200:4 --cpus-per-task=32 --mem=256G --time=02:30:00 tillicum/eval_concurrent.sbatch
```

- **Analysis:**
  - `tillicum/round2_result.py` gives within-set success rates and paired differences
    (`tillicum/results/luna-iter/round2-final.txt`).
  - `tillicum/iter_pooled.py` gives the task-clustered bootstrap across all passes
    (`tillicum/results/luna-iter/iter-pooled.txt`).
- **Follow-up:** passes `rs-a` / `rs-b` (Nov 14) re-evaluated Round 1 and Round 2, plus Round 2 with an SFT reference
  (`r2-sol2-r2-refsft`, `tillicum/r2_refsft.sh`) and epoch-1 checkpoints (`tillicum/r2_epoch1_evals.sh`). Neither
  variant recovered Round 2: −5.4 and about −4.5 vs Round 1.

## Caveats

- **Pass-to-pass noise.** T 1.0 sampling isn't reproducible across runs (sglang batching), and the sites change.
  Between identical reruns, 20–24% of OM2W and 27–34% of DeepShop task outcomes flip, and one pass flipped the sign of
  Round 2 − Round 1. Use multiple passes and task-clustered CIs; single-pass differences of about 5 points are not
  reliable.
- **DeepShop in multi-model passes.** With 4 models × 12 browsers on one node, Amazon serves its "Continue shopping"
  check page in about 63–68% of episodes (about 20–30% in single-model runs), and success drops. Comparisons within a
  pass stay fair, but absolute DeepShop numbers here are about 13 points below single-model evals, and the gaps are
  compressed. OM2W is the cleaner benchmark for these comparisons.
- **Half A vs half B collection.** Half A was steered by Luna and relabelled by Sol; half B was steered by Sol.
- **Only 2 passes for SFT and the data-matched single round.** Data-matched − Round 1 (+2.1) is not significant.

## Artifacts

| what | where | backed up |
|---|---|---|
| code, configs, eval summaries, analyses | this repo (`tillicum/`, `scripts/`, `tillicum/results/`) | git (`piotr_arm`) |
| Round 1 adapter + training metadata | `/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs/r2-sol-r1-all-m0.3/student/update-000332` | HF `PTeterwak/openwebrl-4b-r2-sol-r1-all-m0.3` (private) |
| Round 1 pairs | `/gpfs/scrubbed/piotrt/owrl/joint-data/sol-Aall-m0.3` | HF dataset `PTeterwak/openwebrl-luna-iter` (private) |
| Round 2, data-matched adapters | `/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs/{r2-sol2-r2,r2-sol2-ctrl}/student/update-{000391,000669}` | **not backed up**: the HF upload failed when the private storage quota was reached |
| half-B pairs (`sol2-*`) | `/gpfs/scrubbed/piotrt/owrl/joint-data/sol2-{Br1,Bsft,ctrl}` | **not backed up** (same reason) |
| collections (states, screenshots, Sol selections) | `/gpfs/scrubbed/piotrt/owrl/c2-collect/{luna-A-*,sol2-B*}` | no |
| per-episode eval results | `/gpfs/scrubbed/piotrt/owrl/eval/pass{s2-a,s2-b,rs-a,rs-b}/` | summaries only, in git |

Scratch is purged after 60 days of inactivity. Copy anything you want to keep to group storage or Kopah S3.
