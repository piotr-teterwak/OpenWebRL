# Reproducing RPO on matched pairs (tillicum)

This is the best actor so far: OpenWebRL-4B-SFT fine-tuned with RPO (DPO + SFT on the
chosen response) on joint ARM preference pairs, **keeping only pairs whose chosen and
rejected responses make the same number of tool calls**.

| Online-Mind2Web, 300 tasks, o4-mini judge | Overall (invalid = fail) | Valid only |
|---|---|---|
| Base OpenWebRL-4B-SFT (3 runs) | 31.3–33.3 | 34.7–37.5 |
| RPO, original pairs, lr 5e-5 | ~5 (runs away; eval cancelled) | – |
| **RPO, matched pairs, lr 5e-5** | **40.0** | **49.6** |

Against the same-night base run: 45 tasks won and 19 lost (sign test p = 0.0016). This is
a single seed. 58 of its episodes were invalid, against about 30 in the other runs,
mostly on tasks whose sites block the browser (Best Buy, UPS, JCPenney).

## Why matching matters

In the original 5,540 pairs, chosen makes more tool calls than rejected in 19% of pairs
and fewer in 7%. At lr 5e-5, DPO learns "more actions wins": the actor stops ending its
turn, and about 80% of episodes hit the 1,024-token response cap. Dropping every pair
with unequal action counts (5,540 → 4,078) removes that signal. Raising β (0.3, 0.5,
1.0) or the SFT weight (α = 2) only partly fixed it.

## Steps

All job scripts are in `tillicum/` and hardcode `--account=zixianma`, the repo at
`/gpfs/home/piotrt/OpenWebRL` and outputs under `/gpfs/scrubbed/piotrt/owrl`. Change
those to your own account and paths first. The Python env, base model and C2 data are
Zixian's, read-only:

- venv: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv`
- base actor: `/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT` (HF: `OpenWebRL/OpenWebRL-4B-SFT`)
- source pairs: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910`

`/gpfs/scrubbed` is purged after 60 days of inactivity, so check these still exist.

### 1. Data

Download the matched data from HF ([`PTeterwak/openwebrl-arm-matched-pairs`](https://huggingface.co/datasets/PTeterwak/openwebrl-arm-matched-pairs); JSONL only, screenshots stay on tillicum) or rebuild it from joint-data-v2
on a compute node:

```bash
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
OUT=/gpfs/scrubbed/$USER/owrl/joint-data/matched
$PY scripts/filter_joint_pairs.py --src /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910 --out $OUT
```

You should get 4,078 training pairs (`filter-report.json`). Validation (658 pairs) is
copied unchanged. One pair is dropped so the final batch splits evenly across 2 GPUs.
The JSONL references screenshots by absolute tillicum paths in Zixian's scratch.

`verify_arm_joint_data.py` expects `<data>/../joint-data-inputs-0d83b48`. Link it next
to your output:
`ln -s /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48 $(dirname $OUT)/`.

### 2. Train (2 × H200, about 2.3 h including the reference pass)

```bash
OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 OWRL_JOINT_DATA=$OUT RUN=rpo-matched-lr5e-5 \
  sbatch --time=05:00:00 --exclude=g013,g021 tillicum/train_joint.sbatch
```

Settings: β 0.1, α (SFT weight) 1, LoRA rank 16 / alpha 32 / dropout 0.05 on the
language-model projections, batch 32, 1 epoch = 128 updates. Resubmitting the same
command resumes from the latest checkpoint. The config is frozen on first submission
and keyed to code and data hashes.

### 3. Gate (1 × H200, about 30 min)

```bash
LABEL=rpo-matched-lr5e-5 CKPT=student/update-000128 sbatch tillicum/gate.sbatch
```

This merges the LoRA, samples 5 responses on 200 stored states and reports
`length_cap_rate`. Expected: about 0.006 (base 0.005). Runaway models score 0.27. The
script's 5% threshold is too loose; treat anything above about 1.5% as suspect.

### 4. Browser eval (1 × H200 per model, about 2.3 h)

```bash
MODELS=base,rpo-matched-lr5e-5 PASS=repro1 sbatch --gpus=h200:2 --cpus-per-task=16 --mem=240G \
  --time=04:00:00 --exclude=g013,g021 tillicum/eval_concurrent.sbatch
```

Always run base in the same job: base varies by about 2 points between runs. The result
is `<eval>/pass<PASS>/<label>/<bench>/summary.json`. For WebVoyager and DeepShop, see [README_BENCHMARKS.md](README_BENCHMARKS.md); `BENCH=webvoyager` or
`BENCH=deepshop` switches benchmark (GPT-4o judge, 30 screenshots).

The judge needs `OPENAI_API_KEY` in `.env` (chmod 600, gitignored).

## Things that cost us runs

- **Slow nodes:** g013 and g021 run about 3× slower. Exclude them.
- **Proxy:** jobs inherit an HTTP proxy. The eval script adds `127.0.0.1` to `no_proxy`; otherwise sglang's self-check fails with a 503.
- **Browser port locks:** locks in the node-shared `/tmp` that other users left behind make every port look taken, and every task fails within seconds. The script uses a job-private lock directory.
- **Odd final batch:** the trainer needs an even final batch for 2 ranks. `filter_joint_pairs.py` handles this; other data builds may not.
- **Offline metrics:** held-out preference accuracy rises with lr even as browser success collapses. Gate on generation length, not on preference accuracy.
