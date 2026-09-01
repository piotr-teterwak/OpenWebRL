# Best-of-5 action selection for the OM2W eval (branch `piotr-selection-eval`)

Context for whoever (human or Claude) runs this on klone. Ported from Piotr's
SCC selection stack on 2026-09-01. Questions → piotr.teterwak@gmail.com.

## What this is

At every browser step the policy samples **n=5 candidates at temp 0.7** and a
**judge picks one**; the winner's tokens/logprobs flow downstream unchanged.
Three judges ("selectors") are available:

| SELECTOR | judge | what it is |
|---|---|---|
| `zs` | `OpenWebRL/OpenWebRL-4B-SFT` | untrained zero-shot judge (the policy judging itself) |
| `arm` | `PTeterwak/OpenWebRL-4B-SelectionARM` | selection ARM: LoRA-trained on 40k GPT-5.5 selection labels, merged |
| `scalar` | `PTeterwak/OpenWebRL-4B-ScalarRM-LoRA` | Bradley–Terry value-head RM from the same labels; scores each candidate, argmax |

SCC reference results (o4-mini official judge, 272-task intersection, local
browsers): n=1 33.8 < zs 41.5 < scalar 46.3 < GPT-5.5-select 47.1 < ARM 51.1.
Under GPT-4.1 w/o-aborted: 41.0 / 50.5 / 54.7 / 51.3 / 57.0. Expect klone's
absolute numbers to differ (env/judge); orderings should hold.

## How to run

```bash
# once: HF token that can read Piotr's PRIVATE model repos
huggingface-cli login   # or export HF_TOKEN=...
# once, scalar only:
pip install peft

SELECTOR=arm    ARM=nucleus sbatch klone/eval_om2w_selector.sbatch
SELECTOR=zs     ARM=nucleus sbatch klone/eval_om2w_selector.sbatch
SELECTOR=scalar ARM=nucleus sbatch klone/eval_om2w_selector.sbatch
```

**Always `ARM=nucleus`.** Under `ARM=greedy` the 5 samples are near-identical
and selection is a no-op by construction (measured on SCC: 72% of steps have
>=2 genuinely distinct candidates at temp 0.7; greedy collapses that).

## How it's wired (if you need to debug)

- `openwebrl/generate_browser.py`: `_maybe_arbitered_inference()` wraps the
  two `_run_inference_step` call sites. **Env-gated**: with `OWRL_ARBITER`
  unset, behavior is byte-identical to the plain eval — the baseline sbatch
  (`eval_om2w.sbatch`) is unaffected by this branch.
- `openwebrl/frontier_arbiter.py`: builds the selection prompt (task, history,
  current screenshot, numbered candidates), calls the judge's OpenAI-compatible
  chat API, parses `{"selection": N}` (falls back to candidate 1 on parse
  failure), and writes per-step sidecars. If `OWRL_SCALAR_URL` is set it
  instead POSTs candidates to the scalar server's `/score` and argmaxes.
- `klone/eval_om2w_selector.sbatch`: 2 GPUs. GPU0 = policy sglang via the
  untouched `run_evaluation_local.sh`; GPU1 = judge (sglang server for
  arm/zs, `klone/scalar_server.py` for scalar), health-gated before the eval
  starts, killed on exit. Judge log: `$RUN/judge_server.log`.
- Env knobs: `OWRL_ARBITER_N` (5), `OWRL_ARBITER_TEMP` (0.7),
  `OWRL_ARBITER_COORDS=normalized` (the ARM was trained on normalized coords —
  don't change), `ARM_HF` to swap the arm-mode judge model.

## Debugging tips (paid for in blood on SCC)

- **Sidecars first**: `$RUN/arbiter_logs/` has one JSON per step with all 5
  candidates, the pick, and the judge's reasoning. If results look like n=1,
  check sidecars exist at all (if not, `OWRL_ARBITER` didn't reach the eval
  process) and that `selected_index` varies (if always 0, the judge's output
  isn't parsing — look at `reasoning`).
- Steps are ~5-6x slower than n=1 (5 policy samples + 1 judge call). A full
  300-task run fits a 12h allocation with N_PARALLEL=8, barely. `TASK_TIMEOUT_SECS`
  is already 9000 in the base sbatch; don't lower it.
- The judge server takes several minutes to load on first run (HF download);
  the sbatch waits up to 20 min before declaring it dead.
- scalar_server loads base + LoRA + value_head; first request is slow
  (compile); it scores 5 candidates per call in one batched forward.
- If chromium preflight fails, resubmit — it's a known bad-node lottery
  (see comments in `eval_om2w.sbatch`).
- Scoring/judging of the finished trajectories is unchanged from the baseline
  eval (this branch only changes HOW ACTIONS ARE CHOSEN), so numbers are
  directly comparable to the `eval_om2w.sbatch` baseline at the same ARM=
  decoding mode.
