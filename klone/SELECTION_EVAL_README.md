# Best-of-5 action selection for the OM2W eval (branch `piotr-selection-eval`)

> Measured klone results, judge-behaviour stats, and operational gotchas:
> [`HANDOFF.md`](HANDOFF.md).


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

