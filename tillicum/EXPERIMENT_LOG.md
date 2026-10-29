# ARM → actor experiments on tillicum: log and porting guide (Oct 2026)

What was run on UW tillicum (H200, Slurm) between 2026-10-19 and 2026-10-28, what came out, and what you
need to rerun it on another cluster. The other docs (`README_RPO_MATCHED.md`, `README_BENCHMARKS.md`)
describe single pieces; this is the overview.

## 1. Setup in one paragraph

Actor: **OpenWebRL-4B-SFT** (Qwen3-VL-4B, HF `OpenWebRL/OpenWebRL-4B-SFT`). Selector ("ARM"/ADM):
**SelectionARM** (`PTeterwak/OpenWebRL-4B-SelectionARM`, commit `81b452d8`). Each browser step the actor
samples 5 candidates (T 0.7, top-p 0.9, 1,024 new tokens); the ARM picks one. Preference pairs: chosen =
ARM pick in a **successful** episode, rejected = another candidate at that state, with ambiguous pairs
removed and **equal tool-call counts** required. The actor is fine-tuned on the pairs with DPO (LoRA),
gated on generation length, then evaluated in the live browser.

## 2. Headline results

All "overall" numbers count invalid episodes as failures. Online-Mind2Web (OM2W, 300 tasks) uses the
gpt-5.6-terra judge (o4-mini was retired 2026-10); WebVoyager (595) and DeepShop (150) use GPT-4o.
Paired tests compare tasks valid in both runs (sign test).

**On-policy DPO is the method that works.** β sweep, 3 seeds each, mean gain over a same-session base:

| β | OM2W | WebVoyager* | DeepShop |
|---|---|---|---|
| 0.1 | −0.4 | +5.0 | −4.0 |
| **0.3** | **+9.4** (3/3 seeds p ≤ 0.007) | +7.9* | **+7.7** (0/3 seeds significant, n≈139) |
| 0.5 | +8.3 (2/3) | +4.7* | +6.9 |
| 0.7 | +3.5 | +5.2* | +6.6 |
| 1.0 | +2.3 | +4.4* | +5.8 |

\* **WebVoyager numbers from 10-27 are unreliable.** That base run (34.5%) was depressed by site blocking
(Coursera CloudFront blocks, Hugging Face 429s), most likely from running ~25 eval/collection jobs at
once. A normal-day base is ~44%. A clean re-run is queued (§6).

Best single model: **on-policy DPO β 0.3, seed 1**, 36.3% / 37.0% OM2W in two sessions (base 26.7% / 28.3%),
uploaded as `PTeterwak/OpenWebRL-4B-onpolicy-DPO-b0.3` (private, merged bf16).

**Ablations (OM2W unless noted):**

| Question | Result |
|---|---|
| SFT vs DPO on the **same** on-policy data | Filtered SFT ≈ base (+0.4 to +2.0 vs a normal base); DPO beats it by 6–9 pts, p ≤ 0.04 on all 3 seeds. The contrast with rejected actions is what helps. |
| Off-policy (matched joint data) DPO/RPO | Seed-dependent: one seed of matched RPO gained, two did not; matched DPO at any β ≈ base. |
| Action-count filter | Required: without it DPO learns "more actions wins" and ~80% of episodes run into the token cap. |
| ARM at inference (best-of-5) | +4.6 pts on base (all 3 eval seeds significant); **+0.8 on top of the DPO actor (n.s.)**. The DPO actor alone (35.2%) ≥ base + ARM (32.7%) at 1/5 the generations. |
| Iteration (round 2: collect with the round-1 actor + ARM, DPO again) | Beats base (+4 to +9) but **not** round 1 (−1.6 to −6.8). Cause: length drift — 14–19% of episodes cut off at 1,024 tokens vs 4% for round 1; round-2 data is 17% longer. Adding the SFT term (RPO) only reduces cut-offs to 8–14%. |
| Likelihood displacement | Held-out per-token log-prob of chosen: SFT +0.025, DPO −0.043 (100% of pairs down), RPO −0.011, while RPO keeps most of DPO's margin. Displacement tracks browser cut-offs. |

## 3. Pipeline and scripts (all in `tillicum/` and `scripts/`)

| Step | Script | Notes |
|---|---|---|
| Collect rollouts | `c2_collect.sbatch` (`openwebrl/arm_c2.py collect`) | `ACTOR=`, `TEACHER=selection` (ARM) or `baseline` (no ARM), `SEED=`, `TASKS=`, `NAME=`, `PARALLEL=`; judge `JUDGE` (default gpt-5.6-terra). Resumable. Shard big pools into 1-GPU jobs. |
| Build pairs | `build_round2_data.sbatch` (wraps `scripts/prepare_arm_joint_data.py`) | Merges shards, builds pairs, keeps equal-action-count C2 pairs, drops `joint-data/matched` validation task groups, renames ids `C2r2:`, verifies. `build_combined_data.sbatch` concatenates two pair sets. |
| Train | `train_joint.sbatch` (`scripts/train_arm_joint_ddp.py`) | `OBJECTIVE=dpo|sft|dpo_action`, `LR`, `BETA`, `SFT_WEIGHT` (RPO), `SEED`, `NGPU=1|2`, `OWRL_ACTOR` (start model = DPO reference), `OWRL_JOINT_DATA`, `RUN`. Config is frozen on first submit and keyed to code/data hashes; resubmit to resume. |
| Gate | `gate.sbatch` | Merges the LoRA, samples 5 responses on 200 stored states, reports `length_cap_rate`. |
| Evaluate | `eval_concurrent.sbatch` | `MODELS=base,<run>`, `PASS=`, `BENCH=online-mind2web|webvoyager|deepshop`. One GPU per model; run base in the same job. |
| ARM composition | `eval_arm_compose.sbatch` | `TUNED="s1=<merged> ..."`, `ARMS="base+arm s1+arm@2 ..."` (`@N` = eval seed). |
| ARM data scaling | `arm_scaling_{data,train,score}.sbatch` | Data built; training **on hold**. Scorer fails on the released ARM's tokenizer config (`extra_special_tokens` list) — pass the base tokenizer before resuming. |
| Re-judge stored evals | `rejudge.sbatch` / `scripts/rejudge_om2w.py` | Used for the o4-mini → terra migration. |

Recipe of the best run: `OBJECTIVE=dpo LR=5e-5 BETA=0.3 SFT_WEIGHT=0 SEED=1 OWRL_JOINT_DATA=<onpolicy pairs>`,
1 epoch (163 updates), batch 32, LoRA r16/α32/0.05, AdamW (0.9, 0.95) wd 0.01, warmup 512 examples then
half-cosine to 0.5×.

## 4. Data and models

| What | Where |
|---|---|
| Base actor | HF `OpenWebRL/OpenWebRL-4B-SFT` |
| SelectionARM | HF `PTeterwak/OpenWebRL-4B-SelectionARM` @ `81b452d8` |
| On-policy pairs (round 1, 5,206) | HF `davidan03/openwebrl-arm-onpolicy-pairs` (images included, relative paths) |
| Matched joint pairs (off-policy, 4,078) | HF `PTeterwak/openwebrl-arm-matched-pairs` (JSONL only; image paths are tillicum-absolute) |
| Best round-1 actor | HF `PTeterwak/OpenWebRL-4B-onpolicy-DPO-b0.3` (private) |
| Round-2 pairs (2,752), combined (7,958) | tillicum `/gpfs/scrubbed/piotrt/owrl/joint-data/{round2-b0.3s1,onpol-plus-r2}` (not uploaded; image paths absolute) |
| C2 task pool (2,091 tasks) | tillicum `/gpfs/scrubbed/zixianma/.../c2-full-282782-20260908T075414Z/tasks.jsonl` |
| Eval task files | `openwebrl/data/eval/{online-mind2web,webvoyager_fara,deepshop}.jsonl` (in repo) |

`/gpfs/scrubbed` is purged after 60 days of inactivity: copy anything you need off tillicum.

## 5. Porting to another cluster

**Hardcoded tillicum specifics to change:** `--account=zixianma`, partition `gpu-h200`, the repo path
`/gpfs/home/piotrt/OpenWebRL`, output roots under `/gpfs/scrubbed/piotrt/owrl`, Zixian's Python envs
(`/gpfs/scrubbed/zixianma/openwebrl-runtime/{venv,arm-reproduction/venv}`), the Playwright/Chromium bundle,
model paths, and the `--exclude=` node lists. `.env` must provide `OPENAI_API_KEY` (currently it reads
Zixian's key — use your own).

**Resources (measured):**

| Job | GPU memory | Host RAM | Time (H200) |
|---|---|---|---|
| DPO training, 1 GPU | ~38 GB in use (plan ≥ 48 GB; 80 GB safe — long 32k-token examples) | ~31 GB | ~1.2–1.8 min/update; 163 updates ≈ 3–4 h incl. reference pass |
| Eval / gate (SGLang + 12 browsers) | 9 GB weights + KV; 24–32 GB enough with `--mem-fraction-static` ≈ 0.2 | ~6–7 GB per browser | OM2W ≈ 1.2 h, WebVoyager ≈ 2 h, DeepShop ≈ 0.8 h |
| Collection, 1 GPU (actor + ARM), 16 browsers | ~70 GB | ~120 GB | ~7 h per 523-task shard |

`NGPU=1` training is equivalent to the 2-GPU run (same 32-example mean gradient), just slower.

**Judge cost (OpenAI, list prices):** a 300-task OM2W eval ≈ 300 terra calls ≈ $8–15; a 2,091-task
collection pass ≈ $60–100; WebVoyager ≈ 600 GPT-4o calls.

## 6. Things that cost us runs (read before launching)

- **Concurrency causes site blocking.** Many evals walking the same task list at once get Coursera/Hugging
  Face/Google blocked, and blocked episodes count as *valid failures*. Keep ≤ ~4 eval jobs at a time, no
  collection in parallel, and always run base in the same job as the model.
- **Detect blocks from observations, not text.** Keyword-matching the agent's answer ("blocked", "403") is
  biased between models; use the page URL/title (see Dan's `bench_n3.py`).
- **Gate threshold.** A per-response cut-off rate *p* compounds to ≈ 1−(1−p)^14 of episodes. 1.7% → ~22%
  of episodes lost. Treat > ~1% as a red flag (`gate.sbatch` default 5% is too loose).
- **Iterative rounds drift in length.** See §2; untested fixes in priority order: gate intermediate
  checkpoints and stop early; DPO reference = base rather than the previous round; action-only DPO
  (`OBJECTIVE=dpo_action`); length-matched pairs or a length-normalised loss.
- **Slow nodes:** g011, g013, g015, g021 ran 2.5–3.5× slower; exclude them.
- **Fixed ports collide** when 1-GPU jobs share a node; `eval_concurrent.sbatch` now probes for a free block.
- **o4-mini is retired.** Old numbers judged by o4-mini are ~2–3 pts more lenient than terra; compare
  like with like (re-judged copies live under `<eval>/rejudge-gpt-5.6-terra/`).
- **Seed variance is large** (one matched-RPO seed at +10, two at base). Always run 3 seeds.

## 7. Status at 2026-10-28 19:30 PT

| Experiment | State |
|---|---|
| Round 2 + RPO (from round 1, 3 seeds) | Seeds 42/1 evals finishing: beat base, not round 1; seed 2 training |
| Combined round-1 + round-2 data, DPO from base (3 seeds) | Training, update ~110/249; results ~midnight |
| Clean WebVoyager re-run (3 same-job base + β 0.3 pairs) | Queued after the above |
| No-ARM collection (4 × 2,091 episodes, base actor) | Done; outcome-filtered SFT / trajectory-DPO baselines not yet built |
| ARM data scaling (9 points) | Data built, training on hold |
