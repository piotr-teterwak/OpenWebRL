# OpenWebRL (Xunmei-liu fork) on UW Hyak klone

> **Running this now?** See [`HANDOFF.md`](HANDOFF.md) for current results,
> working commands, and the gotchas that have each cost a run.


## Layout
| Path | What |
|---|---|
| `/gscratch/krishna/piotrt/openwebrl` | the fork (`git@github.com:Xunmei-liu/openwebrl.git`) |
| `/gscratch/krishna/piotrt/openwebrl/.env` | secrets + paths, chmod 600, gitignored |
| `/gscratch/cse/piotrt/envs/slime` | micromamba env (python 3.12, cuda 12.9) |
| `/gscratch/cse/piotrt/openwebrl_build` | sglang + Megatron-LM source checkouts |
| `/gscratch/cse/piotrt/models` | HF weights (`OpenWebRL/OpenWebRL-4B-SFT`) |
| `/gscratch/cse/piotrt/ms-playwright` | chromium bundle (staged to /scr per job) |
| `/gscratch/scrubbed/piotrt/openwebrl_outputs` | checkpoints, rollout logs (21-day scrub) |
| `/gscratch/cse/piotrt/logs` | slurm job logs |

Env is on `cse` because `krishna` is at 97% of its inode quota.

## Jobs
- `build_env.sbatch` — builds the micromamba env (adapted from `build_conda.sh`, which assumes root + `/root`).
- `fetch_model.sbatch` — downloads OpenWebRL-4B-SFT.
- `check_chromium.sbatch` — ldd + live page load; catches missing system libs.
- `smoke_2gpu.sbatch` — `scripts/run_browser_smoke_2gpu.sh` on 2x A100 80GB.

## Deviations from the fork's Beaker setup
1. **weka paths were hardcoded** in the Ray runtime-env JSON of the three
   launchers (`PYTHONPATH` -> their Megatron-LM, `LD_LIBRARY_PATH` -> their conda
   env). Made configurable via `SLIME_MEGATRON_PATH` / `SLIME_ENV_LIB_PATHS`,
   defaulting to the original weka values so Beaker behaviour is unchanged.
2. **No Docker/Kubernetes on klone** -> `SLIME_BROWSER_ENV_MODE=local_process`,
   not Orchard `sandbox`. Upstream measured a higher anti-bot block rate for
   local_process (25.7% vs 17.7%). Expect noisier rewards than the paper.
3. **No apt/root** -> chromium system libs (nss, gbm, atk, ...) come from
   conda-forge instead of `apt-get install`.
4. **`micromamba install ... -c nvidia/label/cuda-12.9.1`** needed `-c nvidia
   -c conda-forge` added; `nccl` and `cuda-nvtx-dev` are not in that label alone.
5. **A100 80GB, not B200 180GB** -> `SGLANG_MEM_FRACTION_STATIC` lowered to 0.45.
6. `TORCH_CUDA_ARCH_LIST="8.0;8.9"` so the env runs on A100, A40 and L40S.
7. **`nvidia-resiliency-ext` is uninstallable on klone.** Megatron-Bridge lists
   it as a dep, but every published release needs glibc >= 2.31 (0.5.0+ need
   2.39) and klone/RHEL8 has glibc 2.28. Installed Megatron-Bridge with
   `--no-deps` + its real deps instead. Verified safe: the fork only touches
   `megatron.bridge.AutoBridge` and `megatron.bridge.models.qwen_vl.*`; the nvrx
   imports on that path are lazy (`fault_tolerance.py` imports inside a function)
   or `safe_import`-guarded (`checkpointing.py` -> `HAVE_RESIL`). The single
   unguarded top-level import is in `bridge/training/pretrain.py`, which nothing
   in slime/openwebrl imports. If a future change pulls in the bridge *training*
   loop, this breaks and the fix is a container with newer glibc.
8. `set -u` cannot be used in the build script: conda's own
   `~cuda-nvcc_activate.sh` reads `NVCC_PREPEND_FLAGS` unguarded and aborts.

9. **scipy vs the `numpy<2` pin.** `build_conda.sh` ends with `numpy<2` (Megatron
   needs it) but never constrains scipy, so pip leaves scipy 1.18.1 which is
   built against the numpy 2.x ABI. transformers lazily imports scipy, that
   import dies, and the lazy loader reports it as
   `cannot import name 'AutoProcessor' from 'transformers'` -- a completely
   misleading symptom. Fixed with `scipy==1.13.1`.
10. **Two cuDNNs in one env.** `build_conda.sh` installs conda `cudnn` (9.10.2)
   AND pip `nvidia-cudnn-cu12==9.16.0.29`. TransformerEngine dlopens pip's
   `libcudnn_cnn.so.9` by absolute path, but the linker resolves its
   `libcudnn_graph` dependency through LD_LIBRARY_PATH. With `$CONDA_PREFIX/lib`
   first you get 9.16 cnn against 9.10 graph ->
   `undefined symbol: _ZTVN5cudnn7backend20SdpaForwardOperationE`.
   Fix is ordering only: pip's nvidia dirs must precede `$CONDA_PREFIX/lib`
   (see `SLIME_ENV_LIB_PATHS` in smoke_2gpu.sbatch). No reinstall needed.

Both 9 and 10 are latent in the fork's own `build_conda.sh`, not artifacts of
the klone port -- worth reporting upstream.

11. **flashinfer runtime JIT needs a build environment.** `set_up.sh` uninstalls
   `flashinfer-jit-cache`, so sglang JIT-compiles attention kernels during
   cuda-graph capture. That compile needs CUDA headers on `CPATH` and libs on
   `LIBRARY_PATH`; without them the link fails with `cannot find -lcuda` and
   every SGLangEngine dies in its scheduler. The Beaker entry sets this up and
   it must be mirrored. Prefer the node's real `/usr/lib64/libcuda.so`; the CUDA
   stub is a fallback and belongs on `LIBRARY_PATH` only, never
   `LD_LIBRARY_PATH` (the stub would then load at runtime instead of the driver).
   JIT cache moved off `$HOME` (10GB quota) to `$PROJ/.cache/flashinfer`.
12. **env_server startup blows the 60s default on GPFS.** Measured on klone for
   a single server, no contention:

   | module | cold |
   |---|---|
   | `slime.utils.processing_utils` | 152.9s |
   | `torch` | 115.4s |
   | `transformers` | 105.1s |
   | `slime.rollout.sglang_rollout` | 49.9s |
   | `sglang` | 29.9s |
   | `openwebrl.docker.env_server` | 6.3s |

   ~460s cold, **87s warm**, against a 60s default -> every episode aborts with
   `env_server was not healthy within 60.0s` and the rollout yields
   `aborted_ratio 1.0`, `effective_batch_size 0`. Raised to 600s startup /
   1800s acquire / 2400s task.

   The real cost is not env_server (6.3s): `openwebrl/__init__.py` eagerly runs
   `from openwebrl.generate_browser import ...`, which pulls torch + transformers
   + `slime.rollout.sglang_rollout`, and importing `openwebrl.env.web_env`
   triggers that parent `__init__`. Since an env_server is spawned **per
   episode** (240 per rollout at full scale), this is a large permanent
   throughput tax on any cluster whose filesystem is slower than weka.
   Fix worth making upstream: make those two names lazy in
   `openwebrl/__init__.py` via PEP 562 `__getattr__`. `from openwebrl import
   generate_turn_sample` keeps working, and slime's
   `--custom-generate-function-path openwebrl.generate_browser.generate_turn_sample`
   imports the submodule directly so it is unaffected. Expected ~87s -> ~7s.
13. env_server logs must go to GPFS, not node-local `/scr` -- they are the only
   thing that explains a rollout abort, and `/scr` is unreadable once the
   allocation ends.

14. `inference_step_timeout_secs` in `openwebrl/browser_training_config.yaml`
   raised 30.0 -> 180.0 (original backed up as
   `/gscratch/cse/piotrt/browser_training_config.yaml.orig`). At 30s every
   episode died with `SGLang /generate timed out after 30.0s`: multimodal
   prefill of a 1280x1000 screenshot plus long browser context on A100 80GB
   exceeds it. `run_evaluate.py` already defaults the same knob to 60.0.
15. `SMOKE_EXIT=1` is cosmetic. `scripts/clean_processes.sh` ends with
   `pkill -9 redis`; with no redis running pkill returns 1 and the launcher's
   `set -e` propagates it. Check `Job ... succeeded` and
   `latest_checkpointed_iteration.txt`, not the exit code.

## STATUS: smoke run PASSES (job 38896851, 2026-08-26)

Full loop verified: browser rollout on live sites -> gpt-4.1 judge -> GRPO step
-> checkpoint. `aborted_ratio 0.0`, `effective_batch_ratio 1.0`,
`format_reward_mean 1.0`, `response_len` mean 249.7. Checkpoint written to
`iter_0000000/` (distcp shards). Rewards are 0.0 because the smoke sets
`BROWSER_MAX_STEPS=6` -- far too few for real WebVoyager tasks -- so advantages
are 0 and there is no learning signal. That is expected for a smoke test.

env_server startup fell 102s -> 33s once the GPFS page cache was warm.

## PERFORMANCE WARNING for a real run
One rollout of 8 episodes + one train step took `perf/step_time` 3539s (59 min):
`log_probs_time` 1608s, `actor_train_time` 1685s, at
`actor_train_tok_per_s` 124 and `actor_train_tflops` 1.75 on 2x A100 -- very low.
Full training is 48 prompts x 5 samples = 240 episodes x 100 rollouts. At this
rate that is not feasible. Investigate before committing GPU time:
`--optimizer-cpu-offload`, `--micro-batch-size 1`, full activation recompute,
and the per-episode env_server import tax (item 12) are the prime suspects.

16. **Ray port collision on shared nodes -- silently runs your job in someone
   else's cluster.** The launchers hardcode `ray job submit
   --address="http://127.0.0.1:8265"`. On Beaker each job has its own network
   namespace so 8265 is always yours; klone shares nodes between users on one
   loopback (g3083 had 6 jobs from 5 users). If another user's Ray head owns
   8265, our `ray start --head` falls back to 8266 but `ray job submit` still
   targets 8265 -- so the training job is submitted into THEIR cluster.
   Observed 2026-08-27, job 39049971: entrypoint failed with
   `can't open file '/gscratch/scrubbed/hamishiv/tmax-srun-training-mini/train.py'`.
   It failed harmlessly, but the failure mode is nasty: no error at submit time,
   the job just runs somewhere else.
   Fixed by parameterising `--port` / `--dashboard-port` in
   `run_browser_smoke_2gpu.sh` and `..._tuned.sh` (defaults unchanged at
   6379/8265) and deriving per-job values from `SLURM_JOB_ID` in the sbatch,
   with an `ss` preflight that aborts on collision. **Worth reporting upstream
   -- it is a correctness bug on any shared-node cluster.**

## THROUGHPUT EXPERIMENT (2026-08-27)

Baseline job 38896851 vs tuned job 39057083, both 2x A100 80GB PCIe, GBS=8.
Tuned changed only mathematically-neutral knobs: TP 2->1 (drops
`--sequence-parallel`, which requires TP>1), micro-batch 1->4, sglang
mem-fraction 0.6->0.5. recompute and `--optimizer-cpu-offload` kept ON.

| metric | baseline | tuned | change |
|---|---|---|---|
| per turn sample | 82.3s | 40.6s | 2.03x |
| actor_train_tok_per_s | 124.4 | 242.9 | 1.95x |
| actor_train_tflops | 1.75 | 3.43 | 1.96x |
| log_probs_time | 1609s | 901s | 1.79x |
| actor_train_time | 1685s | 1050s | 1.61x |
| rollout_time | 229s | 74s | 3.1x |
| step_time | 3539s | 2040s | 1.73x |
| env_server startup | 33s | 9.6s | 3.4x |

**The hypothesis was wrong.** Predicted 5-10x on the theory that per-layer TP
collectives over PCIe (NVLINK_COUNT=0) plus per-micro-batch overhead dominated.
The falsifying test was whether `log_probs_time` (forward only) would separate
from `actor_train_time` (fwd+bwd) toward the ~1:3 ratio compute-bound work
implies. It did not:

- baseline 1609 vs 1685 -> ratio 1.05
- tuned    901 vs 1050 -> ratio 1.17

TP collectives were removed entirely and micro-batches cut 8x, and forward-only
still costs ~85% of forward+backward. So those were secondary: they account for
the 2x, not the remaining ~100x (still only ~0.5% of the pair's 624 TFLOPS peak).

**Next hypothesis — the multimodal path.** Whatever dominates is paid identically
by both passes and scales with samples, not FLOPs. Prime suspects: the H2D copy
of `pixel_values` (CHANGES.md fix #1 deliberately keeps rollout data on pinned
CPU and moves each micro-batch to GPU inside `get_batch`, so screenshots cross
PCIe on every pass), and the vision-tower forward (~1200 patches per 1280x1000
screenshot). Do not sweep more config: profile. A torch profiler trace over a
few micro-batches separates `aten::copy_` (H2D) from vision encoder time.
Note if it IS H2D over PCIe, H200 will not help much -- its bandwidth advantage
is HBM, not the host link.

**Full-run projection at tuned speed:** 1520 samples x 40.6s = 17.1h train
+ ~1.5h rollout = ~18.6h/step x 100 rollouts = ~77 days. Down from ~165, still
not viable.

## H200 RESULT (2026-08-28, job 39277587) -- hardware is NOT the bottleneck

Same tuned config (TP=1, MBS=4, GBS=8, recompute+offload on) on 2x H200 141GB
with NVLINK_COUNT=2, vs 2x A100 80GB PCIe with NVLINK_COUNT=0:

| metric | A100 (39057083) | A100 (39265632) | H200 (39277587) |
|---|---|---|---|
| actor_train_tok_per_s | 242.9 | 248.7 | **265.8** |
| actor_train_tflops | 3.43 | 3.50 | 3.74 |
| log_probs_time | 901s | 789s | 912s |
| actor_train_time | 1050s | 840s | 945s |
| step_time | 2040s | 1738s | 1960s |

**H200 is only ~7% faster** despite ~3x peak FLOPS, ~2.5x memory bandwidth, and
NVLink instead of PCIe. The log_probs/actor_train symmetry -- forward-only
costing as much as forward+backward -- survives on H200 (ratio 1.04, vs 1.17 on
A100), i.e. it is unchanged by interconnect, bandwidth or compute.

Conclusion: every hardware hypothesis is now falsified by measurement (PCIe TP
collectives, memory bandwidth, raw FLOPs, H2D pixel_values, vision tower).
The remaining cost is hardware-independent -- CPU-side, serialised, or
per-micro-batch inside the training loop. **Do not pursue H200 capacity for
throughput**; the ~78-day full-run projection barely moves. The sm_90 rebuild
was still worth doing to establish this, and the env now runs on both.

Next and only remaining lead: the phase profiler (model.py `_phase_profiled`
on forward_only / train_one_step), whose key_averages tables will name the
actual ops.

## ROOT CAUSE FOUND (2026-08-29): WRONG SGLANG VERSION

The ~0.9% eval was caused by **sglang 0.5.18 instead of the pinned 0.5.6.post2**.

`requirements.txt` pins:
    sgl-kernel==0.3.19
    sglang==0.5.6.post2
    sglang-kernel==0.4.0

The klone build script stripped those pins, reasoning that build_conda.sh already
builds sglang from source at $SGLANG_COMMIT and the pypi pins would clobber the
editable install. That gave 0.5.18 (and no sglang-kernel at all). dan29's working
env has exactly the pinned versions, installed as plain wheels, unpatched.

**Symptom:** sglang loads the model as Qwen3VLForConditionalGeneration and accepts
`image_data` without error, but the screenshot never becomes usable vision
embeddings. The model receives a `<|image_pad|>` placeholder with nothing behind
it and confabulates -- "blank or placeholder page with repeated '1' elements" --
while reasoning coherently about a page it cannot see. Every task fails; reward ~0.

**Confirmed by measurement:** after installing the pinned versions (plus re-pinning
nvidia-cudnn-cu12==9.16.0.29, which sglang 0.5.6 drags back to 9.10.2), the same
eval went from **2 successes in 300** to **~27-32% success**, matching dan29's
26-37% comparators.

This also invalidates the training smoke runs' rewards: `format_reward 1.0` with
`raw_reward 0.0` was the model being blind, not max_steps=6 being too few.

**Rule: follow requirements.txt for the sglang trio. Do not substitute the
build_conda.sh source build.** Re-pin cuDNN 9.16.0.29 afterwards.

## VERIFIED EVAL RESULT (2026-08-29, corrected stack)

OpenWebRL-4B-SFT, Online-Mind2Web 300 tasks, greedy (T=0.0, top_p=1.0),
gpt-4.1 judge, max_steps 30. Two independent runs on different hardware:

| | A100 (39323119) | L40S (39323111) | mean |
|---|---|---|---|
| **v2 success_rate** | **34.1%** (72/211) | **33.5%** (71/212) | **33.8%** |
| --abort-mode training | 28.5% (72/253) | 27.4% (71/259) | 28.0% |
| aborted | 74 | 64 | |
| confirmed anti-bot blocks | 15 | 24 | |

The two runs agree to 0.6 points -> reproducible, and inside dan29's 26-37%
comparator range. Compare with the broken stack: 0.9% (2 successes in 300).

Still to do: the nucleus arm (T=0.7/top_p=0.8 + --sampling-backend pytorch)
against this same corrected stack, and the phase profiler for the training
throughput question.

## SUPERSEDED: eval result with the broken stack

OpenWebRL-4B-SFT on Online-Mind2Web, 300 tasks, gpt-4.1 judge, max_steps 30.
greedy (T=0.0, top_p=1.0, job 39282125) and nucleus (T=0.7, top_p=0.8 +
--sampling-backend pytorch, job 39282214).

| | greedy (T=0.0) | nucleus (T=0.7, p=0.8) |
|---|---|---|
| tasks | 300/300 | 300/300 |
| aborted | 58 (19%) | 65 (22%) |
| v2 success_rate | **0.9%** (2/218) | **0.5%** (1/204) |
| training-abort-mode | 0.7% (2/270) | 0.4% (1/259) |
| confirmed anti-bot blocks | 24 | 31 |

The confirmed-block counts are a real, reusable result: 24 and 31 tasks were
independently confirmed blocked by v2's active-tab forensics, ~11-15% of the
scored denominator. That matches upstream's local_process figure (25.7% vs
17.7% sandboxed) and validates dan29's v2 methodology on our data.

**Do not cite these.** dan29's comparators on the same benchmark and model
family are 26-37%; we are ~30x below that, and greedy/nucleus agree, so it is
systematic rather than noise.

Ruled out by inspection:
- judge parsing: rewards are cleanly binary (298x0.0, 2x1.0), judge runs
- patch size: model patch_size 16 x merge 2 = 32, matches image_patch_size 32;
  and resize_scale 1000 takes precedence anyway
- sampling: both arms agree
- model competence: narration is coherent and self-aware, e.g. "The previous
  click action was incorrect again. I'm still clicking on the 'Top Stories'
  section rather than the 'Health' dropdown menu."

That last point is the lead: **the model reasons well but its actions land in
the wrong place.** Something between predicted coordinates and the actual click,
or between the expected observation/prompt format and what this SFT checkpoint
was trained on.

Next steps, in order:
1. **Ask dan29** -- they have working comparators at 26-37% and a known-good
   config; a diff against that beats more solo debugging.
2. Compare one trajectory's screenshot + predicted coords vs where the click
   landed.
3. Field-by-field diff of our eval env vs theirs. We left several at defaults
   they may set explicitly: CONTEXT_NUM_SCREENSHOTS (ours 3, training uses 1),
   SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING, the <think> tags,
   BROWSER_RESPONSE_FORMAT_MODE, TURN_HISTORY_REASONING_MODE.

Eval infrastructure itself is sound and reusable: 300/300 completed, judge
working, aborts ~20% and correctly excluded by both scorers.

## Open question
`CHANGES.md` claims `--lr 1e-6` (paper Table 7) as a deliberate divergence from
upstream's `5e-7`, but that only landed in `run_browser_full.sh`
(`--lr "${LR:-1e-6}"`). Both `run_browser_smoke_2gpu.sh` and
`run_browser_realrun_2gpu.sh` still hardcode `--lr 5e-7`. Decide which is
intended before a real training run; `run_browser_full.sh` respects `$LR`.

## Run
    source /gscratch/krishna/piotrt/openwebrl/.env
    sbatch /gscratch/cse/piotrt/smoke_2gpu.sbatch
