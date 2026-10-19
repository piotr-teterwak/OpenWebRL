# Reproducing the offline-training comparison on klone

Runs the filtered-SFT / DPO table with **our own training runs**, not her
checkpoints — those live on tillicum and are not in this repo.

## What the experiment is

| arm | data | objective |
|---|---|---|
| starting actor | — | none (control) |
| 1A filtered SFT | C2 only, 8,394 turns | SFT on chosen |
| Joint SFT | J = 3,464 C2 + 2,076 Piotr | SFT on chosen |
| Joint DPO | **same J** | DPO on chosen/rejected |

C2 is self-generated (your actor rolled out with the ARM selecting, successful
trajectories kept). Piotr is GPT-5.5 choosing among candidates, published as
[`PTeterwak/action-reward-models-data`](https://huggingface.co/datasets/PTeterwak/action-reward-models-data).
Joint SFT and Joint DPO share data, order, batch, updates and seed and differ
**only** in the objective — the one clean contrast in that table.

Note the C2 contribution differs between arms: 1A uses 8,394 *turns*, the joint
set uses 3,464 *states* re-filtered to those with a usable chosen/rejected pair
(DPO needs one, and both joint recipes had to share a corpus). Joint SFT is
therefore **not** "1A plus Piotr data".

## The actual goal: does a second round add anything?

Round 1 above is the control, not the experiment. The open question — the one
that has not worked for anyone yet — is **iteration**: train the actor on
ARM-selected data, then collect *fresh* data with the improved actor and train
again. In theory each round should compound. In practice it has not.

```
round 1   base actor --collect--> C2_1 --+--> train --> actor_1
                                  Piotr -+
round 2   actor_1    --collect--> C2_2 ------> train --> actor_2
```

Round 2 is the same pipeline with two substitutions: collect using `actor_1`
instead of the base actor, and train from `actor_1`. Evaluate all three actors
(base, actor_1, actor_2) **concurrently**.

### Two reasons it may fail, and how to tell them apart

Both are measurable from data the eval already writes, so neither needs a
dedicated run.

**1. Diversity collapse.** Selection needs the five candidates to differ. SFT on
selected actions sharpens the policy, so each round makes the samples more alike
and leaves the selector less to choose between — training away the thing the
gain depends on. Klone baseline: **62.6%** of steps had >1 distinct tool call
among five candidates (76% by full response). Compute it per round from
`arbiter_logs/`; a fall toward 1 distinct is fatal, a drift from 5 toward 3 may
not be — in the lookahead study the selector's advantage was concentrated
exactly in the 1–3 distinct states.

**2. The selector goes off-distribution.** The ARM was trained to judge
candidates drawn from the *base* actor. `actor_1` samples from a shifted
distribution, so even with diversity intact the ARM is judging inputs it was not
trained on. This degrades selection independently of collapse.

They are separable: if round-2 diversity holds but the gain still evaporates,
it is the selector, not the collapse.

### The control nobody has run

Round 1 cannot distinguish "ARM filtering adds signal" from "more SFT on
in-domain data adds signal", because no unfiltered arm exists. Same turn budget,
turns chosen **at random** from the same trajectory pool, trained identically:

| arm | selection | purpose |
|---|---|---|
| filtered SFT | ARM-selected turns | the published recipe |
| **random SFT** | **random turns, same budget** | **does filtering do anything?** |

If random matches filtered, there is no quality gradient for iteration to climb,
and that explains the failure without needing either hypothesis above. One
training run plus one eval — far cheaper than another iteration attempt, and it
should come first.

## Pipeline

```bash
# 0. Piotr half -- already staged at /gscratch/scrubbed/piotrt/arm_data
#    (4.7 G, pinned revision 0d83b48c, counts verified against the audit).
#    To re-fetch: python scripts/fetch_arm_joint_inputs.py --output <dir>

# 1. C2 half -- no downloadable artifact, must be regenerated (~15 h, ~30 GPU-h)
sbatch klone/c2_collect.sbatch
#    -> $SLIME_OUTPUT_ROOT/c2_<jobid>

# 2. combine into the joint dataset (CPU only, no GPU)
C2_DIR=<step 1 output> sbatch klone/build_joint_data.sbatch
#    -> $SLIME_OUTPUT_ROOT/joint_data_<jobid>

# 3. train -- one job per objective
OBJECTIVE=sft OWRL_JOINT_DATA=<step 2 output> sbatch klone/train_joint.sbatch
OBJECTIVE=dpo OWRL_JOINT_DATA=<step 2 output> sbatch klone/train_joint.sbatch

# 4. evaluate each checkpoint AND the starting actor, submitted together
RESUME=1 RUN_NAME=<name> ARM=nucleus sbatch klone/eval_om2w.sbatch
```

**Step 4 concurrency is the point.** Every significant cell in the published
offline table compares against a *historical* baseline run on a different day;
her own doc says the fix is "a fresh concurrent base-versus-1A evaluation".
Submit the baseline alongside each trained checkpoint, and report **overall
with invalid = failure**, not valid-only — DPO's apparent valid-only advantage
came largely from excluding 16 more episodes than SFT.

## Cost

~56 GPU-h per round: collection 29, training 10, evals 17. Budget ~150 GPU-h
for two rounds including the 30–50% overhead that failed and preempted attempts
have historically cost.

## Gaps and traps

- **No unfiltered-SFT control exists anywhere.** `ARM_FILTERED_SFT_ABLATIONS` is
  a proposal: "annotation, training and evaluation have not been launched". So
  nothing tests whether ARM *filtering* does anything versus training on the same
  volume of unselected turns. Given nothing in that table is significant, this is
  arguably the highest-value single run available.
- `prepare_arm_joint_data.py` sets its own `RLIMIT_AS` 1 GiB / `RLIMIT_CPU` 240 s.
  Those were sized on tillicum; GPFS here is slower, so SIGXCPU is the likely
  first failure. Raise it in the script, visibly, rather than in the sbatch.
- The trainer's roots were tillicum absolutes. They now read `OWRL_RUNTIME` /
  `OWRL_JOINT_DATA` / `OWRL_ACTOR`, defaulting to her originals, so her runs are
  unaffected when the env is unset.
- Selection depends on actor output diversity, which SFT on selected actions can
  collapse. Baseline on klone: 62.6% of steps had >1 distinct tool call among the
  five candidates (76% by full response). Track it per round from `arbiter_logs/`
  — if it falls, selection gain shrinks mechanically.
- **Nothing here has been executed yet.** Syntax-checked, env plumbing tested;
  no stage has run on klone.
