# Evaluation protocols for Piotr's experiments (Tillicum, Nov 2026)

The protocols behind the results in [BASELINES.md](BASELINES.md) and [ITERATIVE_DPO.md](ITERATIVE_DPO.md),
in the same form as Zixian's [`openwebrl/docs/EVAL.md`](https://github.com/zixianma/OpenWebRL/blob/arm/openwebrl/docs/EVAL.md)
on `arm`. There are four set-ups:

| Set-up | Actor | Selectors | Benchmarks | Results |
| --- | --- | --- | --- | --- |
| A. External selectors on our actor | OpenWebRL-4B-SFT | Best-of-Q, PRO-CUA PRM, WebDreamer | OM2W, WebVoyager, DeepShop | BASELINES §3 |
| B. GPT-4o on WebArena-Lite | gpt-4o-2024-08-06 | none, WebArbiter-7B, our ADM | WebArena-Lite (4 sites) | BASELINES §1 |
| C. GPT-4o on Online-Mind2Web | gpt-4o-2024-08-06 | none, WebArbiter-7B, our ADM | OM2W (live) | BASELINES §2 |
| D. Iterative DPO actors | DPO checkpoints | none | OM2W (WebVoyager re-run) | ITERATIVE_DPO |

"Our ADM" is SelectionARM, `PTeterwak/OpenWebRL-4B-SelectionARM` revision
`81b452d800d9f859687074f82680dd5257e02d89`, served by `scripts/serve_arm.py --mode selection`.

## What you need

- **Code.** Branch `piotr_arm` of `piotr-teterwak/OpenWebRL` (commit `c379321` or later). Set-ups A and D use the
  repository's `openwebrl.arm_eval`; B and C use `webarena_repro/` with Zixian's `vendor/native_episode.py` and the
  released WebArbiter viewer (`YaoZ720/WebArbiterCode` at `764ec26`, checked out under
  `/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26`).
- **Software.**
  - Model servers (actor, selectors): Zixian's `openwebrl-runtime/venv` with sglang 0.5.6.post2, torch 2.9.1+cu129,
    transformers 4.57.1.
  - Evaluation: `openwebrl-runtime/arm-reproduction/venv` (A, D) and the WebArbiter browser environment
    `webprm-baselines-20261009/browser-env` (Python 3.12, Playwright; B, C).
- **Models** (all on the job's GPU):

| Model | Revision | Used in |
| --- | --- | --- |
| OpenWebRL-4B-SFT | `/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT` (HF `OpenWebRL/OpenWebRL-4B-SFT`) | A, D, ADM base |
| SelectionARM (our ADM) | `81b452d8` | A (reference), B, C |
| WebArbiter-7B (`ZYao720/WebArbiter-7B`) | `6518479b` | B, C |
| Dreamer-7B (`osunlp/Dreamer-7B`) | `73192d5f` | A |
| Qwen3-VL-4B-Instruct | `/gpfs/scrubbed/weihanx/models/Qwen3-VL-4B-Instruct` | A (PRO-CUA PRM, WebDreamer value model) |
| Qwen2.5-VL-3B-Instruct | `66285546` | A (Best-of-Q embedder) |

- **API.** An OpenAI key with gpt-4o-2024-08-06, gpt-4o-mini, gpt-4.1-2025-04-14 and gpt-5.6-terra. Jobs read
  `OPENAI_API_KEY` from Zixian's `.env` at run time; the key is never written to a log.
- **Hardware.** One H200 per job (8 CPUs, 120 GB). A MIG slice (2 CPUs, 32 GB, about a quarter of an H200's
  billing) is enough for GPT-4o with no selector in C, where nothing runs on the GPU.

## A. External selectors on our actor

| | Online-Mind2Web | WebVoyager | DeepShop |
| --- | --- | --- | --- |
| Tasks | all 300 | all 595 | all 150 |
| Task file (sha256 prefix) | `openwebrl/data/eval/online-mind2web.jsonl` (`8343c23b`) | `openwebrl/data/eval/webvoyager_fara.jsonl` (`ac925331`) | `openwebrl/data/eval/deepshop.jsonl` (`479d35f1`) |
| Judge | gpt-5.6-terra, AgentTrek prompt, final 3 screenshots | gpt-4o, FARA prompt, up to 30 screenshots | gpt-4o, MolmoWeb prompt, up to 30 screenshots |
| Slurm limit used | 8–10 h | 10–12 h (WebDreamer: two shards) | 8 h |

- **Actor.** OpenWebRL-4B-SFT on sglang (mem fraction 0.45, context 32,768); temperature 1.0, top-p 0.9, 1,024 tokens
  per step, at most 30 steps, seed 42, 12 browsers per job, 30-minute task limit (90 minutes for the WebDreamer
  timeout reruns).
- **Selection.** `arm_eval --mode scalar`: 5 candidates per step, the selector server returns one score per
  candidate, the argmax executes (ties to the earlier sample).

| Selector | Server | Settings |
| --- | --- | --- |
| Best-of-Q | `best_of_q/serve_q_selector.py` | final-round Q-function `q5` from the branch-state simulation (`best_of_q/branch_sim.py`); Qwen2.5-VL-3B embeddings |
| PRO-CUA PRM | `pro_cua/serve_procua_selector.py` | PRO-CUA `PRM_PROMPT` on Qwen3-VL-4B-Instruct (mem 0.30); screenshot annotated with their `actions_visual`; score P(is_correct = true), greedy, 512 tokens |
| WebDreamer | `webdreamer/serve_webdreamer_selector.py` | Dreamer-7B world model (bf16, fa3, mem 0.6 of what is free); description = tool call in words + actor reasoning; one imagined step; value model Qwen3-VL-4B-Instruct (mem 0.25), WebDreamer's value prompt, n = 10 or 1 samples, at most 4 value requests in flight at n = 10 (40 at n = 1); a watchdog stops the eval if any server dies |
| Our ADM | Zixian's harness, condition `piotr5` | her `controls-sameday-20261103` batch, 3 runs |

- **Controls.** Our single-sample base actor (`base_t1`) ran in the same days; the ADM row is from Nov 3–4. Compare
  across days with care (see Zixian's same-day rule).

## B. GPT-4o on WebArena-Lite

| | Shopping | CMS (shopping_admin) | Reddit | GitLab |
| --- | --- | --- | --- | --- |
| Tasks | 45 | 35 | 19 | 30 |
| Jobs per condition | 2 | 2 | 1 | 2 |
| Slurm limit | 10 h | 6 h | 8 h | 6 h |

- **Tasks.** The released WebArena-Lite configs (`viewer/config_files/wa/test_webarena_lite.raw.json`,
  `895c2116`), 129 tasks on four sites; Map is not hosted. Per-site task ids are in the job logs listed in
  `/gpfs/scrubbed/piotrt/owrl/webarena/jobs/selectionarm_jobs.txt` (each `cells-*.out` starts with them).
- **Sites.** The original WebArena images, hosted per task from a fresh copy (`webarena_repro/wa_site.py`) inside a
  network namespace; model calls leave through `wa_relay.py` to the cluster proxy. Shopping is unpacked once per job
  to node-local disk from `sandbox/shopping.tar.zst`.
- **Evaluation.** WebArena's programmatic evaluators; the fuzzy-match judge is gpt-4.1-2025-04-14 (the released
  gpt-4-1106-preview is retired); three released-evaluator bugs fixed as documented in `wa_episode.py`.
- **Policy (A18).** gpt-4o-2024-08-06 with the released WebArbiter policy prompt and parser, full action space
  (`WA_POLICY_STOP=full`), no terminal detector (`WA_DETECTOR=off`), gpt-4o-mini helpers, at most 30 steps.
  - No selection: one sample per step.
  - Selectors: Web-Shepherd sampling (`WA_SAMPLING=shepherd`): 20 samples in one call at T 1.0 / top-p 0.9, the five
    most frequent kept.
- **Selectors.** WebArbiter-7B in the released knockout tournament (sglang, greedy, mem 0.6); our ADM through
  `wa_selectionarm.py`, which renders each element action as the SFT actor's tool call (`point_2d` = element centre /
  viewport × 1000).
- **Viewport.** `WA_VIEWPORT=1280x720` (matched), or `1024x2490` (WebArbiter's released service). At 1024×2490 our ADM
  sees a 1024×576 window placed to cover the most candidate targets (`WA_ARM_CROP=576 WA_ARM_WINDOW=cover`).

## C. GPT-4o on Online-Mind2Web

- **Tasks.** All 300 of `online-mind2web.jsonl` (`8343c23b`), live sites, 1280×720.
- **Policy and selectors.** As in B (A18), via `webarena_repro/om2w_episode.py`, which runs the same harness on live
  sites: any origin allowed through the cluster proxy (`tillicum-dtn01:3128`), desktop Chrome user agent,
  screenshots through CDP (no web-font wait), navigation waits for DOM ready.
- **Judge.** The OM2W/AgentTrek prompt from `openwebrl/eval/reward_online_mind2web.py` on gpt-5.6-terra, seed 42, with
  the executed actions and the final viewport screenshot.
- **Jobs.** No selection in 2 MIG shards; each selector in 4 H200 shards of ~73 tasks, 14 h. Per-task limit 50
  minutes; tasks with no result were rerun once with a 20-minute limit (`TASK_TIMEOUT=1200`), and a second hang is
  recorded as invalid.

## D. Iterative DPO actors

- **Eval.** `tillicum/eval_concurrent.sbatch`: `arm_eval --mode baseline` (one sample per step), T 0.7, top-p 0.9,
  1,024 tokens, 30 steps, 12 browsers, all 300 OM2W tasks, judge gpt-5.6-terra (o4-mini before its retirement on
  Oct 23). LoRA checkpoints are merged once into `eval/merged/<run>`.
- **Controls.** Every round-2 model ran in the same job as the round-1 model `onpol-dpo-b0.3-s1`; compare only within
  a pass. Clean WebVoyager re-runs (`passwvclean*`) put base and round 1 in one job.

## Metrics

- **Valid episode.** The judge returned a verdict and the episode was not aborted, timed out or killed.
- **Valid-only success.** Successes / valid episodes. **Overall success.** Successes / all tasks (invalid = failure).
- **Means.** Unweighted over benchmarks (A) or sites (B).
- **Differences.** Paired over tasks valid in both rows (or all tasks for overall), 4,000 task resamples, seed 0,
  95% percentile intervals; site-stratified on WebArena-Lite. Zixian's tables pool her three runs (successes / valid
  over runs) and use 10,000 resamples; we use her pooled numbers for her rows.

## Running it

```bash
# A. one benchmark per job; SELECTOR server is started by the sbatch
BENCH=deepshop PASS=procua-t1 TEMP=1.0 sbatch pro_cua/eval_procua_sel.sbatch
for B in deepshop online-mind2web; do BENCH=$B PASS=webdreamer2 TEMP=1.0 sbatch --time=10:00:00 webdreamer/eval_webdreamer.sbatch; done
for R in 0-297 298-594; do BENCH=webvoyager PASS=webdreamer2 TEMP=1.0 RANGE=$R sbatch --time=10:00:00 webdreamer/eval_webdreamer.sbatch; done
#    NSAMP=1 VALUE_CONC=40 for one value sample; INDICES=<comma list> TASK_TIMEOUT=5400 to rerun timeouts

# B. one site shard per job (copy webarena_repro/jobs/run_cells.sbatch next to the sandbox; METHOD = no_search | arbiter | selectionarm)
SITE=gitlab TASKS="9 25 34 ..." METHOD=selectionarm TAG=S18-r1 WA_POLICY_STOP=full WA_DETECTOR=off \
  WA_SAMPLING=shepherd WA_VIEWPORT=1280x720 sbatch -A vlm -p gpu-h200 --gpus=1 --cpus-per-task=8 --mem=120G run_cells.sbatch
#    1024x2490 ADM: WA_VIEWPORT=1024x2490 WA_ARM_CROP=576 WA_ARM_WINDOW=cover

# C. one shard of OM2W task indices per job
METHOD=arbiter INDICES="10 11 ..." TAG=A18v720-full WA_POLICY_STOP=full WA_DETECTOR=off WA_SAMPLING=shepherd \
  WA_VIEWPORT=1280x720 sbatch -A vlm -p gpu-h200 --gpus=1 --cpus-per-task=8 --mem=120G --time=14:00:00 om2w_cells.sbatch

# Tables
python3 webarena_repro/baseline_tables.py        # A, B, C
python3 tillicum/iterative_tables.py             # D
```

## Code

| File | Role |
| --- | --- |
| `openwebrl/arm_eval.py` | Episode runner for A and D (modes `baseline`, `selection`, `scalar`), judges, protocols |
| `best_of_q/`, `pro_cua/`, `webdreamer/` | Selector servers and per-benchmark job scripts for A |
| `webarena_repro/wa_episode.py`, `wa_protocol.py`, `vendor/native_episode.py` | WebArbiter run path for B (A18 knobs as `WA_*` variables) |
| `webarena_repro/wa_site.py`, `wa_relay.py` | Per-task site hosting and model egress for B |
| `webarena_repro/om2w_episode.py` | Live-site OM2W runner for C (judge included) |
| `webarena_repro/wa_selectionarm.py` | Our ADM in B and C: element actions → tool calls, viewport windows |
| `webarena_repro/jobs/run_cells.sbatch`, `om2w_cells.sbatch` | Job scripts for B and C (selector servers started inside) |
| `webarena_repro/PROTOCOL.md` | WebArbiter reproduction decisions D1–D15 and ablations A1–A20 |
| `webarena_repro/baseline_tables.py`, `tillicum/iterative_tables.py` | Every table, from the raw per-task results |

- **Machine-specific paths.** Model, environment, sandbox and output paths are Tillicum paths at the top of each job
  script and in `wa_episode.py` (`TASKS`), `om2w_episode.py` (`TASKS`, `JUDGE_SOURCE`) and `wa_relay.py` (proxy).
- **Outputs.** A: `/gpfs/scrubbed/piotrt/owrl/eval/pass<PASS>/base+<selector>/<benchmark>/results/`; B:
  `.../webarena/runs/cells/<site>-<method>-<tag>-<job>/task-*/summary.json`; C: `.../webarena/runs/om2w/<method>-<tag>-<job>/`.
  `/gpfs/scrubbed` is purged after 60 days without access.
