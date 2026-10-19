# Which cluster is this, and what has to change

This repo has been run on **two different UW Hyak clusters**. They share nothing
but the code: separate filesystems, separate Slurm, separate accounts, separate
hardware. If you are a new session starting work here, read this before trusting
any path or job script.

| | tillicum | klone |
|---|---|---|
| whose | Zixian's | Piotr's |
| partition | `gpu-h200` | `gpu-l40s`, `gpu-a40`, `gpu-a100`, `ckpt-all` |
| account | `zixianma` (personal) | `gpu-l40s-{cse,krishna}`, `gpu-a40-krishna`, `gpu-a100-{cse,krishna}`, `ckpt-{cse,krishna}` (shared with labmates) |
| GPUs | H200, 141 GB | L40S/A40 48 GB, A100 80 GB; H200 only via preemptible `ckpt-all` |
| filesystem | `/gpfs/scrubbed/zixianma/...` | `/gscratch/{cse,krishna,scrubbed}/piotrt/...` |
| job scripts | `scripts/*.sbatch` (69 of them) | `klone/*.sbatch` |
| entry doc | — | `CLAUDE.md`, `klone/HANDOFF.md` |

**`CLAUDE.md` describes klone only.** On tillicum, treat its paths, accounts and
the whole `klone/` directory as inapplicable; the reasoning in
`klone/OFFLINE_TRAINING.md` and `klone/HANDOFF.md` still holds, only the
mechanics change.

## Porting a klone job script to tillicum

1. `#SBATCH --partition=gpu-h200`, `--account=<yours>`, `--qos=normal`.
   Drop `--constraint=` (klone-specific GPU allowlist) and any `ckpt` directives.
2. Repoint the roots. On klone these are `$PROJ=/gscratch/cse/piotrt` (env,
   models, playwright bundle, logs), `$REPO=/gscratch/krishna/piotrt/...`,
   `$SLIME_OUTPUT_ROOT=/gscratch/scrubbed/piotrt/...`. Find the tillicum
   equivalents; do not assume `/gscratch` exists there.
3. The conda env, model weights and playwright chromium bundle are **per-cluster
   artifacts**. They are not in the repo and do not transfer — rebuild or
   re-download on the new machine.
4. Sizing differs. klone L40S nodes are 128 CPU / 8 GPU; A40 and A100 nodes are
   52 CPU / 8 GPU. Zixian's tillicum scripts ask for 16-32 CPUs and 120-960 GB.
   Measure rather than copy: a completed klone eval used a mean of 5.3 busy cores
   against 20 allocated (27%).

## Already portable

These were made cluster-agnostic and need no translation:

- `scripts/train_arm_joint_{sft,ddp}.py` read `OWRL_RUNTIME`, `OWRL_JOINT_DATA`
  and `OWRL_ACTOR`, defaulting to the tillicum paths, so unset env = her
  original behaviour.
- `openwebrl/run_evaluate.py --resume` reuses the newest matching output
  directory and skips tasks that already have a usable record. Zero-length and
  truncated records are re-run deliberately — a run killed by a full filesystem
  leaves exactly those.
- `openwebrl/arm_c2.py` and `scripts/prepare_arm_joint_data.py` take their paths
  as arguments.

## Things that cost real time on klone, likely to recur anywhere

- Job state is not health. An eval whose arbiter never engaged degrades silently
  to an n=1 baseline and looks fine; check `arbiter_logs/` is growing and a
  sidecar shows `candidates=5`.
- `Errno 122` can mean the **shared volume** is full rather than your quota, and
  it can clear on its own before you finish diagnosing it.
- Over-requesting CPUs depresses scheduling priority and buys nothing.
- `prepare_arm_joint_data.py` sets its own `RLIMIT_AS` 1 GiB / `RLIMIT_CPU` 240 s,
  sized on tillicum. On a slower filesystem expect SIGXCPU.

The full list is in `klone/HANDOFF.md`, which is worth reading on either cluster.
