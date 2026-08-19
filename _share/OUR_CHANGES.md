# Our changes to OpenWebRL

Fork lives at `/weka/oe-training-default/zixianm/xunmei/OpenWebRL`
(clone of `github.com/OpenWebRL/OpenWebRL`, upstream HEAD `05f8ed4`).

Everything below is `slime/` (the training backend) plus new launch scripts.
`openwebrl/*` edits already in the working tree predate our work (dated 08-04,
staged in the index we inherited) — we did not touch the browser env or reward code.

Apply with:  `git apply _share/openwebrl_slime_fixes.patch`

## A. Four bugs that blocked training on 2×80GB

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `megatron_utils/actor.py`, `data.py` | The whole rollout (tokens + `pixel_values`) was moved to GPU up front; image-heavy rollouts OOM'd before the first step | Keep bulk data on CPU (pinned, with a pageable fallback); move each micro-batch to GPU in `get_batch` right before its forward |
| 2 | `megatron_utils/loss.py` + `model.py` | Full `[T, V]` fp32 logits were materialized up to 3× per micro-batch (Float16Module upcast + temperature copy) | Set `fp32_output=False`; upcast and apply temperature per response chunk only. Numerics identical — cast/divide commute with slicing |
| 3 | `megatron_utils/model_provider.py` | `--recompute-granularity` never reached the bridge provider, so activation recompute silently never ran: ~47GiB of stored activations per ~6.7k-token micro-batch | Forward `recompute_granularity` / `_method` / `_num_layers` like the parallelism args (3 lines) |
| 4 | `megatron_utils/__init__.py` | With recompute ON, Megatron's checkpoint wrapper crashed: `save_for_backward can only save variables, but argument 6 is of type list` (`deepstack_visual_embeds` is a list) | Monkey-patch that flattens list/tuple args into individual tensors before `save_for_backward` and reassembles them inside `_rebuilding_forward`. Verified against a mock CheckpointFunction: reproduces the crash, then gradients match |

Result: steady 24.9 GiB / peak 38.9 GiB on 80GB cards, ~110 min per iteration at TP1.

## B. Robustness fixes for preemptible clusters

- `slime/ray/rollout.py` — aborted trajectories arrived with `reward=None`, empty
  tokens, or zero-length responses and crashed Megatron's `get_batch` (negative pad).
  Now: `None` rewards → 0.0; degenerate samples dropped, then padded back to the
  original count with zero-loss-mask clones so the batch invariants
  (`num_local_samples // num_local_gbs`) still hold.
- `megatron_utils/data.py` — metric reduction crashed on `None` entries from
  aborted/padded samples; they are now excluded from the mean.
- `slime/ray/rollout.py` — **actor-restart bug (found 08-19).** `RolloutManager` is
  created with `max_restarts=-1`, but the train actor pushes `train_parallel_config`
  exactly once (at `set_rollout_manager` time). When Ray restarted the manager
  mid-run, the fresh instance had no such attribute and the *next* `generate()` died
  with `AttributeError: 'RolloutManager' object has no attribute
  'train_parallel_config'` — at the very last line, after a full ~55-minute rollout,
  throwing the whole iteration away and burning a cluster retry. Fix: persist the
  config to `$SLIME_SAVE_DIR/train_parallel_config.json` when it is set, and reload
  it lazily if the attribute is missing after a restart.

## C. Launch / ops (new files, not upstream)

- `beaker/launch_train.sh`, `beaker/train_entry.sh` — gantry launcher + on-node entry.
  Resume is automatic: if `$SLIME_SAVE_DIR/latest_checkpointed_iteration.txt` exists it
  resumes from it, so preemption costs at most one iteration.
- `scripts/run_browser_full.sh` — our parameterised copy of
  `run_browser_Qwen3VL_4B_Instruct.sh`. **Batch arithmetic is unchanged from upstream**
  (`rollout_batch_size 48`, `n_samples_per_prompt 5`, `global_batch_size 256`,
  `micro_batch_size 1`, `ppo_epochs 2`, same dynamic-sampling filter).
  Deliberate divergences: `--lr 1e-6` (paper Table 7; the repo script has 5e-7),
  `--num-rollout 90` (paper stage 1), TP 4→1/2 (we have 2 GPUs not 8),
  `--save-interval 1` (survive preemption), `--recompute-granularity full` (fix #3),
  eval-during-training disabled (we evaluate separately).
- `scripts/convert_ckpt_to_hf.sh` — distcp → HF export + verification.
  Verified: 713/714 tensors (`lm_head.weight` absent is benign — origin has
  `tie_word_embeddings: True` with embed==lm_head bitwise-identical), and weights
  do differ from base.
- `scripts/run_staged_eval.sh` — convert + serve + evaluate several checkpoints in one
  job, so mid-run evals cost one queue slot instead of one per arm.
