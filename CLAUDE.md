# CLAUDE.md — OpenWebRL on UW Hyak klone

> **This describes klone.** The project has also been run on **tillicum**, a
> separate Hyak cluster with its own filesystem, Slurm and accounts. If you are
> not on klone, read [`CLUSTERS.md`](CLUSTERS.md) first — the paths, accounts and
> the whole `klone/` directory below do not apply there.

Read this first. It is short on purpose; it says where to look and what will
bite you, not how the code works.

## What this repo is

Fork of `git@github.com:Xunmei-liu/openwebrl.git`. Browser-agent RL
(Megatron-LM + slime + SGLang + Ray + Playwright/Chromium) on Qwen3-VL-4B.
Two active branches:

- `piotr-rl-code` — RL training work
- `piotr-selection-eval` — best-of-5 action selection evals **(this branch)**

## Read these before doing anything

| doc | what it answers |
|---|---|
| **`klone/HANDOFF.md`** | **Start here.** Measured results, working commands, and 10 gotchas that have each cost a run. |
| `klone/README.md` | cluster/env setup, paths, deviations from the Beaker config |
| `klone/SELECTION_EVAL_README.md` | what `zs`/`arm`/`scalar` are and how the arbiter is wired |

## Environment

Never use the system python.

```bash
PY=/gscratch/cse/piotrt/envs/slime/bin/python     # micromamba env, py3.12, cuda 12.9
export PYTHONPATH=/gscratch/krishna/piotrt/openwebrl-eval:/gscratch/cse/piotrt/openwebrl_build/Megatron-LM
```

Imports off GPFS take ~2 minutes (transformers especially). **That is normal,
not a hang** — do not kill it and do not add a timeout under 300s.

Do not run `python -m openwebrl.<module>` for the visualizers; the package
`__init__` pulls the training stack. Invoke the file path directly.

## Secrets

`.env` (chmod 600, gitignored, **never commit**) holds the OpenAI key used by
the GPT-4.1 outcome judge and `HF_TOKEN` for the private model repos. Jobs get
it via `source $REPO/.env`. `HF_HOME` is redirected there too, so a token in
`~/.cache/huggingface` alone is not visible to jobs.

**Before every push, grep the diff** for `sk-proj-`, `sk-[A-Za-z0-9]{20,}`,
`OPENAI_API_KEY=`, `hf_[A-Za-z0-9]{20,}`, `ghp_`, and private-key headers.

## Slurm

Shared accounts, so pending is normal and often long.

| account / partition | use for |
|---|---|
| `gpu-l40s-krishna` / `gpu-l40s` | **default.** Non-preemptible, `MaxTime=UNLIMITED`. |
| `ckpt-krishna` / `ckpt-all` (`--qos=ckpt-gpu`) | starts much sooner, **gets preempted** — short jobs only |

Set `--time` generously up front: a running job's limit **cannot** be raised
(`scontrol update TimeLimit` → "Access/permission denied"). An over-long limit
costs nothing on `gpu-l40s`.

## Things that look fine but are not

These have each burned a real run. `klone/HANDOFF.md` has the full list.

1. **"Job is RUNNING" is not health.** A selector eval whose arbiter never
   engaged silently degrades to an n=1 baseline and looks normal. Check
   `arbiter_logs/` is growing and a sidecar shows `candidates=5`.
2. **Slurm state is not health either.** The throughput probes exit 0 even when
   the inner run fails — check that `perf/step_time` actually appears.
3. **Per-task tracebacks in the logs are usually fine.** A dead website raises
   inside a WARNING line and the task is marked ABORTED; the run continues.
   Only line-start tracebacks and `[eval] FATAL` are job failures.
4. **Check batch divisibility before submitting.** Megatron asserts
   `GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) == 0` at startup. A 6-GPU
   TP=2 job gives DP=3, and 64 % 3 != 0 — that probe queued 9 days to fail
   instantly on arithmetic.
5. **Size memory from `MaxRSS` of a comparable run**, not intuition. Measured:
   ~6.7 GiB per concurrent browser trajectory.
6. **`pip install` can silently downgrade CUDA libs.** `peft` pulled cuDNN
   9.16 → 9.10 and broke three jobs. Use `--no-deps` and re-check the
   `nvidia-*` pins afterwards, not just torch/sglang/transformers.
7. **Env vars get shadowed.** Several sbatches set a knob twice; the last
   assignment wins. Grep for all occurrences before believing one.

## Working style that has paid off here

- **Verify past the status line.** Nearly every failure in this project looked
  healthy from the outside.
- **Measure rather than reason about contention.** Predictions about PCIe,
  oversubscription and GPU class were each refuted by a probe. The measured
  win came from a config that the reasoning said should be worse.
- **Record the surprise, not the fix.** `HANDOFF.md` exists because the
  expensive knowledge is which assumptions were wrong, and that is exactly what
  does not survive in the code.
