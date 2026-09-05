# klone handoff — selector evals, throughput, tooling

Operational state as of **2026-09-05**. Read `README.md` for cluster/env setup
and `SELECTION_EVAL_README.md` for what the selectors *are*; this file is what
has been run, what was measured, and what will bite you.

Everything below was measured on klone, not inherited from SCC.

---

## 1. Where the numbers stand

### Selector evals (Online-Mind2Web, 300 tasks, GPT-4.1 outcome judge)

| selector | job | v2 | training | aborts | tasks |
|---|---|---|---|---|---|
| `arm` | 39447295 | **41.6%** | 36.1% | 37 | 300/300 |
| `zs` | 39429421 | **37.9%** | 32.0% | 29 | 300/300 |
| `scalar` | 39619163 | *running (started 2026-09-05, g3114, ~9h expected)* | | | |
| n=1 nucleus | 39383327 | 32.1% | 23.9% | 87 | 300/300 |
| n=1 greedy | 39323119 | 34.1% | 28.5% | 74 | 300/300 |

SCC reference ordering: n=1 33.8 < zs 41.5 < scalar 46.3 < ARM 51.1.

**The ordering reproduces (n=1 < zs < arm); absolute values sit 7-10 pts lower.**
Do not compare the absolutes to SCC: different outcome judge (GPT-4.1 vs
o4-mini), different denominator (300 vs a 272-task intersection), different
live-web conditions. The ordering is the portable claim.

Two score columns, both from `compute_eval_success_rate_woaborted_v2.py`:
- **v2** — a blocked-looking task leaves the denominator only if the *active tab
  at the final step* independently corroborates a genuine block (Cloudflare,
  CAPTCHA, "Access Denied", outage). Unconfirmed candidates are recounted as
  failures. Requires `eval_samples` saving to have been on.
- **training** (`--abort-mode training`) — only system failures are excluded;
  aborts count as failures.

### What the judges actually do (from `arbiter_logs/` sidecars)

| | picks 1st | picks 2nd-4th | picks 5th |
|---|---|---|---|
| `arm` | 59.4% | 34.3% | 6.4% |
| `zs` | 44.8% | 23.7% | **31.5%** |

4,719 selection steps for arm, 4,364 for zs. **The zero-shot judge has a severe
last-position bias** — 31.5% on the final candidate against 7-9% for each middle
slot is positional, not content-driven. ARM decays monotonically (59/16/11/8/6),
which is what a judge discriminating on content looks like. This is a concrete
mechanism for the 3.7-pt ARM > zs gap.

Caveat on effect size: **only ~76% of steps have >1 distinct candidate.** On the
rest all five samples collapse and selection is a no-op, so the per-*decision*
effect is larger than the headline.

Note also that `frontier_arbiter.py` falls back to candidate 1 on a parse
failure, so a judge whose output stops parsing degrades *silently* toward n=1.
The position histogram above is the check that this did not happen — always run
it before believing a selector result.

### RL throughput (one iteration, `NUM_ROLLOUT=1`, warm-up included)

| config | min/step | samples | s/sample | projected days |
|---|---|---|---|---|
| 8xA40 TP=4 DP=2, 26 browsers | 132.1 | 640 | 12.38 | 21.7 |
| 4xA100 TP=4 DP=1, 12 browsers | 94.3 | 448 | 12.62 | 22.1 |
| **8xL40S TP=2 DP=4, 96 browsers** | **37.6** | **704** | **3.20** | **5.6** |

`probe_l40s8.sbatch` is the winner and it is not close — ~3.9x better per sample
than either alternative, on *weaker* GPUs. **The bottleneck was browser
concurrency, not GPU class, not parallelism.** A100 is not worth chasing; more
GPUs is not worth chasing. Raise `BROWSER_SLOTS` first.

Its internals still show `wait_time_ratio 0.78` — the trainer idles 78% of the
step waiting on rollout — so there is likely more headroom above 96 slots.

Both numbers include first-iteration warm-up. **A `NUM_ROLLOUT=2` run for a
steady-state number has not been done** and should be, before anyone plans
around 5.6 days.

### Context budget

Measured, not assumed: a 1280x1000 screenshot is **1,240 tokens**
(`image_grid_thw [1,62,80]`, patch 16 / merge 2). Against
`rollout_max_context_len: 32768`:

| `--context-num-screenshots` | hit the limit |
|---|---|
| K=1 (shipped default) | **0/25** trajectories |
| K=20 | **17/25**, median step 19 |

Overflow is **not** graceful — `generate_browser.py:1777` ends the episode and
marks every turn sample `TRUNCATED` / `generation_length_limit`. It does not
evict old turns. So K>1 is not a free knob: past ~step 19 it kills the rollout,
right where the long-horizon tasks live. K=20 needs a bigger window to be
testable at all.

---

## 2. Running things

```bash
# selector eval, full 300 tasks. ALWAYS ARM=nucleus (greedy makes selection a no-op)
SELECTOR=scalar ARM=nucleus sbatch klone/sel_l40s.sbatch    # non-preemptible, use this
SELECTOR=arm    ARM=nucleus sbatch klone/sel_ckpt.sbatch    # ckpt: faster to start, gets preempted

# scoring (point at the eval dirs)
klone/score_selectors.sh
klone/score_intersection.sh      # common-task subset across selectors

# throughput probe
sbatch klone/probe_l40s8.sbatch

# trajectory report -> self-contained HTML, big annotated screenshots + prompt anatomy
python openwebrl/visualize_turn_samples.py \
    --eval-dir <eval_dir> --out report.html \
    --num-examples 12 --max-steps-per-example 8 \
    --context-num-screenshots 1 --compare-k 20 \
    --tokenizer /gscratch/cse/piotrt/models/OpenWebRL/OpenWebRL-4B-SFT

# single trajectory in Jupyter (same overlays, bigger screenshots)
python openwebrl/visualize_eval_trajectory.py --eval-dir <dir> --task-id <id>
```

Run scripts with the env python: `/gscratch/cse/piotrt/envs/slime/bin/python`.
Import off GPFS is slow (~2 min for transformers); that is normal, not a hang.

**Do not run `python -m openwebrl.<mod>`** for the visualizers — the package
`__init__` pulls the training stack. Invoke the file path directly.

---

## 3. Gotchas that actually cost time

Each of these burned at least one run.

1. **`SLURM_RESTART_COUNT` counts preemptions.** `sel_ckpt.sbatch` budgeted 5
   restarts for chromium retries by reading it — but Slurm bumps it on *every*
   restart, so on `ckpt-all` the scheduler silently spends the browser budget.
   Job 39450598 then declared chromium "unusable after 9 requeues" and quit in
   2m48s on a node that had passed preflight the run before. **Fixed in
   `sel_l40s.sbatch`** with its own job-ID-keyed counter file. If you copy the
   ckpt variant, copy the fix.

2. **`--task-indices` is comma-separated only.** `run_evaluate.py:507` does
   `int(x) for x in ...split(",")`. `TASK_INDICES="0-119"` raises `ValueError`
   ~15 min in, after the judge has finished loading. Pass `0,1,2,...` or nothing.

3. **Do not run 8-hour evals on `ckpt-all`.** scalar needs ~9h at ~99 s/task and
   was preempted at 26/300 while running perfectly. arm got lucky with a clean
   3h window; that is luck, not a plan.

4. **`pip install peft` downgrades cuDNN.** It silently pulled
   `nvidia-cudnn-cu12` 9.16.0.29 -> 9.10.2.21 and broke three jobs. Install with
   `--no-deps`, and after *any* pip operation verify the `nvidia-*` pins, not
   just torch/sglang/transformers.

5. **peft is incompatible with the pinned torchao 0.9.0** (sglang pins it).
   `klone/scalar_server.py` bypasses `is_torchao_available` rather than
   upgrading — do not "fix" this by bumping torchao.

6. **Fixed ports collide.** Two selector jobs on one node both bound
   `SGLANG_PORT=9000` and would have cross-wired results silently. Both sbatches
   now derive ports from `SLURM_JOB_ID`. Keep that.

7. **The ARM/ScalarRM configs were exported under transformers 5.x**, the klone
   stack is 4.57.1. A local patched copy lives at
   `/gscratch/cse/piotrt/models/PTeterwak/OpenWebRL-4B-SelectionARM`
   (`rope_parameters`->`rope_scaling`, extracted `rope_theta`, vision
   `model_type` back to `qwen3_vl`, `extra_special_tokens` dropped) and is used
   via `ARM_HF`. **Re-exporting the models under 4.57 would retire this.**

8. **Chromium launch is a node lottery** (3s..213s cold, measured). Preflight is
   8-way to match `N_PARALLEL=8` — a 1-way check passes on nodes where the real
   workload fails. `PREFLIGHT_COLD_MAX_S` is 90 (45 rejected a node at 46s).

9. **Browser subprocesses must not inherit `LD_LIBRARY_PATH`.** With
   `$CONDA_PREFIX/lib` on it chromium resolves 35 libs off GPFS (~40s cold, 0/8
   under 8-way). `browser_python.sh` clears it; route browser launches through it.

10. **The Conv3d patch is load-bearing for RL.** `Qwen3VLVisionPatchEmbed`
    hits `slow_conv_dilated3d_cuda`, a fallback that loops over the batch;
    rewriting it as a matmul gave **33.7x on log_probs**. Lives in
    `slime/backends/megatron_utils/__init__.py` on the RL branch, disable with
    `SLIME_DISABLE_PATCH_EMBED_MATMUL=1`. Related to pytorch#168167 but distinct.
    Untested against cuDNN >= 9.15, where the upstream fix may make it redundant.

---

## 4. Open threads

- **scalar has never completed.** 4 attempts: 1 preemption, 1 genuinely bad node
  (g3127), 2 bugs of ours (#1, #2 above). The scalar path itself works — attempt
  39447296 was at 26/300 with `abnormal=0` when killed. **39619163 is the fixed
  run and is live** (g3114, non-preemptible, all 300 tasks, both servers up,
  sidecars showing `candidates=5` — i.e. genuine best-of-5, not a silent n=1
  fallback). Fill in its row above when it lands.

  **Verify any selector run this way before trusting it.** "Job is RUNNING" is
  not enough: check `arbiter_logs/` is growing and that a sidecar shows 5
  candidates. `frontier_arbiter.py` falls back to candidate 1 on parse failure
  and `OWRL_ARBITER` not reaching the eval process degrades to n=1 — both look
  like a healthy run from the outside.
- `NUM_ROLLOUT=2` throughput run for a steady-state (non-warm-up) number.
- Push `BROWSER_SLOTS` above 96 — 78% trainer idle says there is room.
- Test `SLIME_DISABLE_PATCH_EMBED_MATMUL=1` on cuDNN >= 9.15 to see whether the
  upstream Conv3d fix makes our patch unnecessary.
- Re-export ARM/ScalarRM under transformers 4.57 (retires gotcha #7).
- Queued and probably worth cancelling: `39404760` (a100 probe), `39424540`
  (2-node l40s). Both test hypotheses the l40s8 result already answered.

## 5. Files added here

| file | what |
|---|---|
| `sel_l40s.sbatch` | selector eval, non-preemptible, fixed requeue counter — **use this** |
| `sel_ckpt.sbatch` | ckpt variant; starts sooner, gets preempted, has gotcha #1 |
| `chromium_preflight.sh` | 8-way browser preflight with warm-up + 3 retries |
| `score_selectors.sh` | v2 + training success rates per selector |
| `score_intersection.sh` | same, restricted to tasks all selectors completed |
| `probe_l40s8.sbatch` | the 3.20 s/sample throughput config |
| `../openwebrl/visualize_turn_samples.py` | batch HTML trajectory report + prompt anatomy |
| `../openwebrl/visualize_eval_trajectory.py` | Jupyter viewer, now with the same overlays |
