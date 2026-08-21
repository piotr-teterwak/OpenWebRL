# Changes on top of upstream OpenWebRL

This fork is upstream `github.com/OpenWebRL/OpenWebRL` at `05f8ed4` plus the
commits on `main`. Everything is in the tree — `git log 05f8ed4..main` for
granular history, or the compare view on GitHub. Short version: memory fixes so
training fits on 2×80 GB GPUs (the paper used 8× large-memory cards), crash
guards for the turn-level GRPO path, and rollout observability.

## A. Four bugs that blocked training on 2×80 GB

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `slime/backends/megatron_utils/actor.py`, `data.py` | The whole rollout (tokens + `pixel_values`) was moved to GPU up front; image-heavy rollouts OOM'd before the first step | Keep bulk data on pinned CPU memory; move each micro-batch to GPU in `get_batch` right before its forward |
| 2 | `.../loss.py`, `model.py` | Full `[T, V]` fp32 logits materialized up to 3× per micro-batch (Float16Module upcast + temperature copy) | `fp32_output=False`; upcast + temperature per response chunk. Numerics identical — cast/divide commute with slicing |
| 3 | `.../model_provider.py` | `--recompute-granularity` never reached the bridge provider, so activation recompute silently never ran (~47 GiB of retained activations per micro-batch) | Forward the three recompute args like the parallelism args |
| 4 | `.../megatron_utils/__init__.py` | With recompute ON, Megatron's checkpoint wrapper crashed: `save_for_backward can only save variables, but argument 6 is of type list` (`deepstack_visual_embeds`) | Wrapper flattens sequence args into tensors and reassembles them in the recomputed forward; gradients verified identical against a non-checkpointed reference |

Result: steady ~25 GiB / peak ~39 GiB per 80 GB GPU.

## B. Robustness (turn-level GRPO data path)

- Aborted trajectories arrive with `reward=None`, empty tokens, or zero-length
  responses and crashed Megatron's `get_batch`; now dropped and padded back to
  the original count with zero-loss-mask clones so batch invariants hold
  (`slime/ray/rollout.py`), with `None`-safe metric aggregation
  (`megatron_utils/data.py`).
- `RolloutManager` restart bug: the manager is created with `max_restarts=-1`
  but its `train_parallel_config` was pushed exactly once at startup, so after
  any Ray actor restart the next `generate()` died with `AttributeError` — after
  a full rollout. The config is now persisted next to the checkpoints and
  reloaded lazily (`slime/ray/rollout.py`).
- Host auto-blacklist for sites that hard-drop connections at TCP level (e.g.
  refusing datacenter IPs) never fired in `local_process` mode due to an
  error-string mismatch; fixed (`openwebrl/generate_browser.py`).

## C. Rollout observability

- `sample_log.jsonl`: one row per episode — reward, terminate reason,
  dynamic-filter verdict, and anti-bot markers scanned from the model's own
  narration (`slime/rollout/sglang_rollout.py`). Motivation: the mixed-reward
  group filter does NOT remove anti-bot noise (measured ~16% of failures inside
  kept groups touch a block), and auditing that needs per-sample data.
- Sampled trajectory dumps: a few episodes per rollout (block-suspected ones
  prioritized) saved as self-contained `trajectory.html` with screenshots.
- Both are cheap (`--dump-details`-style .pt dumps include image tensors and are
  not practical). Toggles: `SLIME_SAMPLE_LOG`, `SLIME_TRAJ_DUMP_PER_ROLLOUT`,
  `SLIME_TRAJ_DUMP_BLOCKED`.

## D. Launch & eval tooling (new files)

- `scripts/run_browser_full.sh` — parameterised launcher derived from upstream's
  `run_browser_Qwen3VL_4B_Instruct.sh`. Batch arithmetic unchanged (48 prompts ×
  5 samples, global batch 256, ppo epochs 2, same dynamic-sampling filter).
  Deliberate divergences: `--lr 1e-6` (paper Table 7; upstream script says 5e-7),
  TP configurable (we run 1–2 GPUs), `--save-interval 1`, recompute on (fix #3),
  in-training eval disabled (we evaluate separately).
- `scripts/run_staged_eval.sh` + `tools/convert_torch_dist_to_hf_bridge.py`
  usage — convert distcp → HF, then serve + evaluate several checkpoints and the
  SFT base back-to-back in one job.
- `beaker/` — launchers for our cluster (AI2 beaker), incl. checkpoint rotation
  (`prune_ckpts.sh`: keeps newest 3 + every 10th + anything with an
  `hf_converted/`). Cluster-specific; ignore unless you run on beaker.
