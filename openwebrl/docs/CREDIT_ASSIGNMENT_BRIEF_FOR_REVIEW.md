# Per-step credit assignment for a web-agent RL run: full brief, and a request for review

Xunmei Liu · 2026-09-24 · written for an outside reader with no prior knowledge of this project

You are being asked to review a line of experiments and suggest what to do next. This document is
self-contained: it explains the system, the motivation, everything we have tried, what we measured,
what we believe and how confident we are, and the options we are weighing. The companion file
[`CREDIT_ASSIGNMENT_SUMMARY.md`](CREDIT_ASSIGNMENT_SUMMARY.md) is the terse results summary; this
one carries the reasoning, the design decisions and their justifications, the negative space (what
we did *not* do and why), and the open questions.

**The short version.** We are training a 4B vision-language web agent with GRPO. The reward is one
binary judgment per episode, broadcast unchanged to all ~10 turns. We tried twice to break that
into per-step credit using a pretrained action reward model. Both attempts produced no gain; the
second one is measurably (if not significantly) harmful, and we understand why. We are now deciding
whether to fix the second attempt, change what signal the reward model contributes, abandon the
pretrained model for an outcome-trained one, or abandon per-step credit entirely and spend the
compute elsewhere. **Our questions are in §10.** Read §1–§3 for the system, §4–§7 for what we did,
§8–§9 for what we now believe and what we are considering.

---

## 1. The system, from zero

### 1.1 The task and the environment

The agent operates a real Chromium browser on **live public websites**. It is given a natural
language task ("Find the general operating hours for the Yuma BMX track", "Compare the price of X
and Y on site Z") and a starting URL. At each turn it receives a screenshot (1280×1000, downscaled
to a 0–1000 coordinate space) plus its own history, and emits a thought followed by one or more
tool calls:

```
click(point_2d=[x, y])   write(message="...")   press_keys(keys=["Enter"])
scroll(direction, amount)   goto_url(url)   go_back()   new_tab()   switch_tab(i)
wait(seconds)   done(response="...")
```

An episode ends when the agent calls `done`, hits the step cap, or the environment aborts (browser
crash, page timeout, anti-bot block, network failure). A turn can contain several tool calls
executed in sequence, e.g. `click(search box) > write("query") > press_keys(["Enter"])`.

Practical consequences that matter for everything below:

- **The environment is non-stationary and adversarial.** Sites change, rate-limit, serve CAPTCHAs,
  and go down. Roughly 20–25% of evaluation episodes abort for reasons unrelated to the policy.
- **You cannot reset to a mid-episode state.** There is no way to branch: "what would have happened
  if the agent had clicked the other link at step 4" is not answerable by re-running, because the
  live page will have changed and the browser cannot be cloned. This rules out the standard
  counterfactual-rollout approach to credit assignment.
- **Episodes are short**: 8–14 steps typically, with a 15-step cap during training.

### 1.2 The policy

`OpenWebRL-4B-SFT`: a 4B vision-language model (Qwen3-VL family) supervised-fine-tuned on web agent
trajectories. This is the starting point for all RL runs. It is what the paper
([OpenWebRL, arXiv 2606.02031](https://arxiv.org/abs/2606.02031)) calls the SFT checkpoint.

### 1.3 The reward: one judge call per episode

At the end of an episode a **GPT-4.1 judge** is shown:

1. the task text,
2. the full textual action history — every turn's tool calls plus the environment's feedback
   (`click: Succeed: 'click' on <a> "Middle East" at (737, 135) executed. Page navigated to ...`),
3. the **last 3 screenshots** of the episode (not all of them — this matters later),
4. the agent's final `done` response.

It returns SUCCESS or NOT SUCCESS. That is the entire reward signal: `R ∈ {0, 1}`, once per
episode. Judge temperature 0.

### 1.4 The RL algorithm: turn-level GRPO

Per iteration:

1. Sample **48 tasks**, and for each, **5 independent rollouts** (the "group"). Rollout temperature
   0.8.
2. Judge all 240 episodes → `R ∈ {0,1}` each.
3. Compute a group-normalized advantage per *trajectory*:
   `A_i = (R_i − mean_group R) / (std_group R + ε)`, std with ddof = 1.
4. **Dynamic filter**: groups where all 5 rollouts agree (5 successes or 5 failures) have zero
   variance, contribute no gradient, and are discarded. The sampler keeps drawing tasks until it
   has 48 groups with mixed outcomes.
5. **Broadcast**: each trajectory is split into its T turns, and every turn becomes an independent
   training sample `(prompt_t, response_t, A_t)` with **`A_t = A_i` for all t**. These turn samples
   from all groups are shuffled into one batch.
6. PPO update: global batch 256, 2 epochs, lr 1e-6, constant.

Two structural facts to keep in mind:

- Because rewards are binary and groups have 5 members, `A_i` takes only the values
  `±0.447, ±0.730, ±1.095, ±1.789` (for 4/1, 3/2, 2/3, 1/4 splits). **`min |A| = 0.447`.** This
  number appears throughout our design.
- The loss is a sum over turn samples, each weighted by its own `A_t`. Nothing in the objective
  distinguishes a turn that caused the outcome from one that was incidental.

### 1.5 Evaluation, and why it is the hard part

**Protocol.** 300 Online-Mind2Web (OM2W) tasks, greedy decoding (T = 0, top-p 1), 30-step cap,
the same GPT-4.1 judge. We report `success / (tasks that were neither aborted nor confirmed
anti-bot blocks)`, which we call the **v2 scorer**. (A related metric, "valid-only", drops only
aborts and runs 1–3 points lower. Different collaborators use different ones; we always say which.)

**Noise.** This is the single most important practical constraint in the project. Re-evaluating
the *same checkpoint* three days apart **flips the verdict on 24.5% of tasks**. That is about
**±2.3 pp on a single rate** and **≈3.3 pp on a difference between two rates**. Consequences:

- A single checkpoint pair can only reveal differences of roughly **5 pp or more**.
- Comparing arms evaluated in different sweeps is close to meaningless for small effects.
- Every real comparison must be **same-window, all-300-tasks, paired at the task level**
  (McNemar). We learned this the hard way: our first two headline numbers (−10.4 pp and −9.8 pp)
  were both artifacts — one from arms having completed different numbers of tasks (different
  denominators), one from cross-sweep comparison.

**Baseline.** Our own reproduction of the paper's RL stage, v2 scorer, one sweep on 30 Aug:

| base (SFT) | iter 38 | iter 40 | iter 50 | iter 60 | iter 70 | iter 90 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32.2 | 45.7 | 47.5 | 46.9 | 46.6 | 49.4 | 48.6 |

The paper's curve on the same metric is ≈36.5 → ≈50.5. We reproduce the shape and the size of the
gain, sitting 2–4 points lower throughout. **Essentially all of the gain arrives by iteration 38**;
from there to 140 every checkpoint sits in a 43–53 band, i.e. within noise of each other. A second
training stage with a 30-step horizon (iters 100–140) adds nothing measurable (p = 0.60).

---

## 2. The motivation: why per-step credit

The gradient says "this 11-step episode failed, so push down all 11 turns equally". That is
obviously lossy:

- A failed episode often contains several correct steps and one decisive mistake. All are punished
  identically.
- A successful episode often contains wasted turns (scrolling past the answer, clicking something
  inert, going back and forth between tabs). All are reinforced identically.

The standard fixes are unavailable or expensive here:

- **A learned critic / GAE.** The trainer is actor-only; adding a value head is a real change to
  the training stack, and the value target is a single binary label at the end of a ~10-step
  episode with an enormous observation space (screenshots of arbitrary websites).
- **Monte Carlo counterfactuals.** Impossible: cannot clone a live browser mid-episode (§1.1).
- **Process supervision from a human or a frontier model.** Expensive per step, and we would be
  distilling a preference, not a causal contribution.

What we *did* have was a pretrained action reward model that already scores `(state, action)`
pairs. Using it seemed like the cheapest possible route to per-step credit: one extra model call
per turn, one reward post-processing hook, no change to the trainer.

**The research question, stated precisely:** *can a pretrained action reward model supply per-step
credit inside outcome-driven GRPO, such that the resulting checkpoint beats the same run without
it, on held-out tasks, under a paired evaluation?*

---

## 3. The tool: Piotr Teterwak's action reward models

Two models, trained from the same labels, on the same 4B backbone (LoRA r = 32).
Repo: https://github.com/piotr-teterwak/action-reward-models

### 3.1 How the labels were made

1. Take ~3k states from SFT trajectories.
2. At each state, sample **5 candidate actions** from the SFT actor (T = 0.7, top-p 0.9).
3. Show the state (screenshot, task, history) and all 5 candidates to **GPT-5.5**, and ask it to
   pick the best one. ~40k labels, ~$150.

**Nothing is executed.** The alternatives are never run in the browser, and **no task outcome ever
enters the training data**. This is a model of a teacher's local preference among sampled actions.

### 3.2 The two models

- **Selection ARM** (generative): input = screenshot, task, history, *all 5 candidates*; output =
  `{"selection": N}`. Trained by SFT on GPT-5.5's picks.
- **Scalar RM** (Bradley–Terry value head): input = same context + *one* candidate; output = a
  scalar. Trained on 76.7k (chosen, rejected) pairs with
  `L = −log σ(r_chosen − r_rejected)`. Held-out pairwise accuracy **75.8%**.

We work with the Scalar RM because its interface — one action, one number — is the shape a per-step
credit term needs. **A property of the BT loss that drives this entire document: the loss only ever
sees differences between candidates at the same state, so the per-state offset of the score
function is completely unconstrained.** Adding any constant to all scores at one state leaves the
loss unchanged.

Measured on our own data: **between-state sd 35.1, within-state sd 16.0 — 82.9% of all score
variance lives in the unconstrained offset.** Raw scores range roughly −114 to +110.

### 3.3 What the model demonstrably does well

Used as it was trained — choosing among candidates at one state — it works. Sample 5 actions per
step at T = 0.7 and execute its argmax:

| actor · benchmark | n = 1 | Scalar RM | Selection ARM |
| --- | ---: | ---: | ---: |
| OpenWebRL-4B-SFT · OM2W (Piotr's report) | 33.8 | 46.3 | 51.1 |
| same, our re-run on our cluster (v2, nucleus) | 31.7 | 35.5 (paired p = 0.014) | 40.1 (p = 0.0001) |
| same, partner cluster, 121 paired tasks | 48.8 | 55.4 (+6.6 pp) | — |

Absolute numbers differ by cluster and judge; the **ordering reproduces everywhere**. This is a
real, replicated effect, and it is the reason we thought the model had usable step-level knowledge.

**Note the asymmetry of requirements.** As a selector the model only needs the *ranking* to be
better than chance, and the policy is not trained on its output — a wrong pick costs one suboptimal
action, which the environment and the outcome reward then correct. As a *reward* its scores enter
the gradient directly, its errors are baked into the policy, and the *magnitude* of score
differences starts to matter even though the BT loss never constrained magnitudes beyond the point
where σ saturates. We under-weighted this asymmetry at the start.

---

## 4. Four ways the model has been used, and what each established

| use | mechanism | result | status |
| --- | --- | --- | --- |
| **Selector at inference** | sample 5, execute argmax | +6.6 to +17 pp over n=1, replicated on 3 clusters | **works** |
| **Expert iteration** | 1 of 5 rollouts in each GRPO group executes best-of-5 by the RM; train on the resulting data as usual | inside training the expert rollout beats its 4 plain siblings 59.7% vs 49.2%, +10.4 pp, p < 0.0001, 2,331 trajectories | **selection effect established; checkpoint effect NOT.** On the 92 tasks every arm completed: +2.2 to +5.4 pp over SFT (all p ≥ 0.33), −3.3 to −6.5 pp against plain RL |
| **exp A: cross-step credit** | `A_t = A + β·z_t`, z = the executed action's score standardized across the steps of one trajectory | paired at iter 38: −1.5 pp, p = 0.87 | **null** |
| **exp B: within-state credit** | `A_t = A + β·c_t`, c from comparing the executed action against 4 counterfactual samples at the same state | paired at iters 38/40: −1.7 pp (p = 0.64), −5.3 pp (p = 0.085) | **no gain, leaning harmful** |

A caution we want to flag explicitly, because it burned us: **the expert-iteration +10.4 pp is a
measurement of the selection procedure, not of the trained checkpoint.** It compares an
RM-guided rollout against plain rollouts *within the same iteration and weights*. It says best-of-5
picks better actions. It says nothing about whether training on those rollouts produces a better
model. We labelled it "works" in an early write-up and had to retract that.

---

## 5. exp A — cross-step credit

### 5.1 Design

```
A_t = A + β · z_t
z_t = ( s_t − mean_t s ) / ( std_t s + 1e-6 ),    z_t = 0 if T < 3
```

`s_t` = the Scalar RM's score for the action the policy actually executed at turn t, standardized
across the steps of that trajectory. β = 0.6. Trained from the SFT checkpoint to iteration 56 on
8×H100. Implemented as a single reward post-processing hook; β = 0 reproduces upstream advantages
bit-for-bit (verified).

**The design property we were proud of:** `Σ_t z_t = 0` by construction, so a trajectory's total
credit is exactly what the judge gave it. The policy cannot raise its return by pleasing the reward
model; it can only redistribute credit between its own steps. We believed this closed the
reward-hacking door.

### 5.2 Result and diagnostics

Null. Paired at iteration 38 (the only same-day comparison available): −1.5 pp, sign-test p = 0.87.

The mechanism was engaged, so "too weak a dose" is not the explanation: mean `|βz| / |A|` = 0.60,
`|βz|` exceeded `|A|` on 27.6% of turns, and the advantage's **sign was flipped on 13.8% of turns**.

### 5.3 Why it could not have worked, in hindsight

1. **Cross-state comparison of a within-state model.** `z_t` compares scores at 15 *different*
   states. 82.9% of score variance is the unconstrained per-state offset (§3.2), so the z-score is
   mostly comparing arbitrary offsets. A reasonable click on a busy page can receive the lowest z
   of its episode simply because that page's offset is low.
2. **`Σz = 0` is a liability, not just a safeguard.** In a successful episode where every step was
   fine, roughly half the steps are *still* pushed negative, by construction. Real example: an
   episode that succeeded, where the scroll that revealed the answer got z = −1.59 and its
   advantage went +0.45 → −0.50.
3. **The terminal `done` step is out of distribution.** The RM was trained on mid-episode states
   where candidates are clicks and scrolls; a plain-text answer is rare in its training data. It
   scores `done` systematically low (mean z −0.87 even in successful episodes).

There is also a measurement trap worth recording. If you ask "does trajectory-mean z predict task
success?" you get AUROC 0.680, which looks encouraging. It is an artifact: z is zero-sum over all
steps, the terminal `done` is strongly negative and gets excluded from the mean, so the residual
mean is positive by construction and larger when spread over fewer steps — and successful episodes
are shorter. The whole 0.68 is episode length.

---

## 6. exp B — within-state credit

### 6.1 Design, and the reasoning behind each choice

Compute the credit the way the model was trained: compare actions *at the same state*.

At a labeled turn, sample **K = 5 responses from the policy with identical prompt and sampling
parameters**, score all five with the Scalar RM, and **execute candidate 0** — the first sample,
chosen with no reference to the scores:

```
d_k = 0                    if candidate k canonicalizes to the executed action
    = r_exec − r_k         otherwise                       (k = the other K−1 candidates)
c_t = clip( mean_k d_k / τ , −1, +1 )
A_t = A + β · c_t
```

| choice | value | why |
| --- | --- | --- |
| execute candidate 0, never the argmax | — | keeps the trajectory **on-policy**; the scores cannot influence behaviour. Because the five are i.i.d. draws from π at the same state, whichever one we call "executed" is arbitrary, so `E[c_t │ state] = 0` — a leave-one-out baseline at the action level. This replaces exp A's zero-sum-per-trajectory property |
| divide by a **global** τ, not the state's own std | τ = 20.0 | a per-state std rescales every state to the same spread, so noise-level differences are blown up to full scale. **37% of states have five candidates with the identical action** (only the thought differs) and 40% have a within-state sd below 2; a within-state z-score manufactures \|z\| ≈ 1–2 out of nothing there. τ is the **median \|r_i − r_j\| over same-state pairs whose actions genuinely differ**, measured from data and then frozen |
| clip at ±1 | — | once σ(Δ) ≈ 1 the BT loss has no gradient, so the model was never trained to make a 60-point gap mean more than a 30-point gap. Unclipped, 1% of states carry 24% of the total squared signal and the max is 9τ |
| β = 0.4 | — | the **largest dose at which the sign can never flip**: `\|β·c\| ≤ 0.4 < 0.447 = min\|A\|`. The judge's outcome always sets the sign of `A_t`; the RM only modulates magnitude. Deliberately conservative for a first corrected run — a null at the maximum safe dose is informative, a null at half that would be ambiguous |
| canonical action matching | clicks within 20/1000, key-name case folded | see §6.3 — a large fraction of "different actions" were not different |
| `done` candidates dropped from the comparison | — | they score ~50 points below everything else, so whenever one appears among the alternatives it inflates c by +0.52 and drives 60% of those states to clip at +1 |
| terminal `done` step masked to c = 0 | — | out of distribution (§5.3). **This turned out to be the fatal decision — see §6.5** |
| 2 of each group's 5 rollouts labeled | — | cost: ≈2.6× generation, browser time unchanged |

### 6.2 Mechanism verification (all passed)

Over 45 training iterations: sign-flip rate exactly 0; fallback rate 0.01%; running τ 19.9 ± 1.6
against the frozen 20.0; `rms(β·c) / rms(A)` = 0.13; and the designed reward-hacking probe,
**"how often is the executed action the RM's argmax", sat at 20.3% ± 1.2% — exactly chance for
K = 5 — and did not drift over the whole run.** The policy was not learning to make its first
sample the RM's favourite.

### 6.3 A measurement bug worth reporting on its own

**28.5% of the pairs feeding the credit were not actually different actions.** Under exact string
matching, `press_keys(["enter"])` and `press_keys(["Enter"])` counted as different, as did clicks
20/1000 apart (i.e. ~25 px on a 1280-wide screenshot, well inside the same button). Folding these
in raises the median gap between genuinely different actions from **11.0 to 20.0** — so the τ we
had originally measured was off by ~1.8×, and every real gap was being overstated.

Concrete examples from traced steps:

- `press_keys(["enter"])` scored 46.75 while three siblings using `["Enter"]` scored 56.50 / 64.00 /
  62.25 → c = −0.789, i.e. an advantage penalty of −0.32 for a capitalization.
- `wait(5)` scored −45.50 and `wait(4)` −63.75 — 18 points for one second.
- Five `done` candidates giving the same correct answer in different words spanned 11.5 points.

This is worth flagging to anyone using a BT reward model on tool-call actions: **the scores are
sensitive to surface form at the same scale as the real signal.**

### 6.4 Results

**Offline, before training** (300-task shadow evaluation: K = 5 scored at every step but candidate 0
executed, so every state is on-policy and every task carries a judge verdict; 4,504 states,
274 joinable tasks). Does trajectory-mean c predict success?

| predictor | AUROC |
| --- | ---: |
| within-state credit c | **0.531** |
| best repaired variant (drop done, tolerant clicks, key case, kind de-bias) | 0.566 (t = 1.77, n.s.) |
| reference: −T, i.e. "the episode was shorter" | 0.702–0.776 |
| reference: mean state score `mean_k r(s, a_k)` | 0.859 |

Controlling for episode length, c's Wald z falls from +2.07 to +1.18.

**Online, paired inside GRPO groups.** Inside mixed groups, do labeled successful rollouts carry
higher mean c than their failed siblings on the same task at the same iteration?
**+0.034**, per-iteration t = 2.7 (p = 0.007). We independently recomputed this from the raw log and
get t = 1.8 aggregating by group rather than by iteration, and +0.024 (t = 1.5) after controlling
for the within-group difference in episode length. So: real, small, and the significance depends on
how you aggregate. It is also partly confounded — β = 0.4 was actively pushing the policy toward
high-c actions during the run, so c and success can co-move without c being causally useful.

**Checkpoint evaluation — the arbiter.** Six arms, one node, one window, 300 tasks each, greedy,
abort counted as failure, paired McNemar:

| arm | v2 rate | mean steps |
| --- | ---: | ---: |
| base (SFT) | 32.7 | 13.7 |
| plain RL iter 38 | 43.6 | 11.0 |
| exp B iter 38 | 41.4 | 7.9 |
| plain RL iter 40 | 45.8 | 11.1 |
| exp B iter 40 | 37.4 | 8.6 |
| exp B iter 20 | 39.3 | 11.1 |

| comparison | difference | p |
| --- | ---: | ---: |
| exp B iter 38 vs plain RL iter 38 | −1.7 pp | 0.64 |
| exp B iter 40 vs plain RL iter 40 | −5.3 pp | 0.085 |
| exp B iter 38 vs SFT | +6.0 pp | 0.039 |
| plain RL iter 38 vs SFT | +7.7 pp | 0.009 |
| plain RL iter 40 vs SFT | +9.0 pp | 0.002 |

Both arms learn. The credit arm learns less, in the same direction at both matched iterations,
never reaching significance, with iter 40 at the edge.

### 6.5 The mechanism of the regression: it learned to give up early

| arm | mean steps | episodes ending ≤ 5 steps | of those, agent called `done` and failed |
| --- | ---: | ---: | ---: |
| base (SFT) | 13.7 | 99 | 37 |
| plain RL iter 38 | 11.0 | 74 | **7** |
| exp B iter 38 | 7.9 | 131 | **38** |
| plain RL iter 40 | 11.1 | 74 | **8** |
| exp B iter 40 | 8.6 | 120 | **33** |

Plain RL **learned not to declare victory early** (37 → 7). exp B did not (37 → 38), and it ends
within five steps about 60% more often. Plain RL also keeps working ~2 turns longer on tasks it is
going to fail (9.4 → 11.6 steps); exp B spends the same length whether it wins or loses (7.8 vs
8.0), i.e. it stops at a roughly fixed short horizon regardless of task state.

**The cause is our own design decision.** We masked the terminal `done` to c = 0 because the RM is
out of distribution there. That makes `done` **the only action that can never receive negative
credit**, while every action that continues the episode can. Measured on the run's own log
(3,198 labeled trajectories with non-zero credit):

| | mean Σ_t c_t over the trajectory | n |
| --- | ---: | ---: |
| episode ends with `done` | **+0.193** | 3,035 |
| episode does not end with `done` | **−0.465** | 163 |

So terminating was worth **+0.263 of advantage** before the judge said anything. The same asymmetry
shows up per action kind, in advantage units: `back` −0.177, `other` −0.164, `write` −0.142,
`key` +0.103, `goto` +0.078, `done` exactly 0. A `back` step loses 40% of the smallest possible
advantage *simply for being a back*, however good that particular back was.

**Our designed reward-hacking probe could not see this.** `executed-is-argmax` detects "the policy
is learning to pick the RM's favourite candidate at a state". It cannot detect "the policy is
shifting its action-kind mix toward kinds that carry a favourable fixed offset". The `done` share
climbed from 9.7% to 14.8% over the run and we read it as neutral at the time.

We think this is the most transferable lesson of the whole project: **when you add a per-step term,
audit its marginal distribution over action types, not just its within-state behaviour.** Any
per-step reward that treats one action class differently — by masking it, or simply because the
model scores that class differently on average — creates a systematic incentive that has nothing
to do with the per-step judgement you intended to add.

---

## 7. What is in the raw data (available to whoever picks this up)

- **exp A credit log**: 13,355 trajectories — per-step RM scores, z, A, T, status, action kinds.
- **exp B credit log**: 9,740 trajectories (3,190 labeled) — per turn: the action, all 5 candidate
  scores, same-action flags, c, A_t. Plus 45 iterations × ~45 training metrics.
- **Step traces**: 5% of labeled trajectories with every step's screenshot, all five candidates
  with their thoughts and scores, and the credit arithmetic. Rendered as browsable HTML.
- **Selector arbiter logs**: ~5,200 states with 5 candidates and 5 scores each, from the
  inference-time selector evaluation.
- **Evaluation directories** with task-level verdicts and per-step screenshots for every arm.
- A resume-capable exp B checkpoint at iteration 45.

Every number in §5 and §6 is recomputable from these with scripts we have.

---

## 8. What we now believe about the reward model

Stated with our confidence level attached.

1. **(high confidence)** The model is a *preference ranker*, not a value estimator. It was trained
   on a teacher's choice among 5 sampled actions, never on outcomes. That is enough for best-of-5
   selection, where only the ranking matters and the policy is not trained on its output. As a
   per-step reward its ranking has to align with outcomes, and it barely does: the same scores give
   +10 pp as a selector and +0.03 on a ±1 scale as a credit signal.

2. **(high confidence)** Its scores are sensitive to action surface form (coordinates a few pixels
   apart, key-name capitalization, `wait` duration) at roughly the same magnitude as the real
   signal.

3. **(medium confidence, and the most interesting finding)** **The untrained half of the score is
   the informative half.** The per-state offset is unconstrained by the BT loss and we had
   therefore dismissed it as noise. But tested on exp A's own training log — 13,355 trajectories
   with group structure, so comparisons are within one task at one iteration and task difficulty
   cancels — over 2,781 mixed groups:

   | statistic, successful rollouts minus failed siblings in the same group | mean diff | t |
   | --- | ---: | ---: |
   | mean executed-action **score level** | +4.38 | 13.78 |
   | minimum score along the trajectory | +6.01 | 11.47 |
   | score excluding the last step | +3.75 | 11.21 |
   | last-step score | +0.21 | 0.26 |
   | **Φ(s′) − Φ(s), the "progress" form** | **−1.58** | **−1.69** |
   | number of steps T | −2.32 | −30.22 |

   In a logistic model over 13,342 trajectories, the score **level** survives controlling for
   episode length at **z = +10.06**, while the within-state **contrast** — the part the BT loss
   actually trains — collapses to **z = +1.18** under the same control. *Not trained* is not the
   same as *not informative*.

   Two warnings attached. Across tasks the raw level is a **difficulty detector** (AUROC 0.859 for
   task success), so using it un-normalized would reward the policy for drawing an easy task. And
   the difference form `Φ(s′) − Φ(s)` is **negatively** associated with success, which kills
   potential-based shaping with this model — see the open question in §10.

4. **(medium confidence)** Episode length confounds nearly every per-step statistic here.
   Successful episodes are 1.6–2.3 turns shorter than their failed siblings. Any candidate signal
   must be evaluated *after* controlling for T, and several of our early positive readings did not
   survive that control.

5. **(high confidence, about method)** AUROC on a trajectory summary is a weak proxy for what we
   actually want. The credit's job is not to predict the outcome; it is to reallocate a fixed A
   across steps. A form can fail the first and succeed at the second. The paired within-group
   comparison is closer to right, and only the checkpoint evaluation settles it.

---

## 9. The options we are weighing

### Option 1 — fix the asymmetry in exp B and rerun

`c_t ← c_t − mean(c over all turns of the same action kind)`, estimated within each iteration's
batch, with `done` **un-masked**. Tested offline against the existing log:

| scheme | mean Σc | termination bonus | the within-group gate |
| --- | ---: | ---: | --- |
| current | +0.160 | +0.263 | +0.0341, t = 2.27 |
| zero-mean per trajectory | 0.000 | 0.000 | +0.0000, t = 1.29 — **signal destroyed** |
| **zero-mean per action kind** | −0.009 | +0.175 | +0.0332, t = 2.22 — **kept** |
| both | 0.000 | 0.000 | −0.0000, t = −0.53 — signal destroyed |

Kind-centering dodges the RM's global dislike of `done` while still letting a *bad* `done` be worse
than a typical one, and no kind is penalized merely for being that kind. It is one line in the
reward hook and changes no training hyperparameter. Cost: one more ~60-iteration run (~2 days on
8×H100) plus a same-window six-arm evaluation (~1.5 h).

*Our concern:* it repairs the mechanism we found, but the underlying signal is still AUROC ≈ 0.53,
and we would be putting a 13% RMS perturbation on the advantage using it. We may simply find a
cleaner null.

### Option 2 — change what the model contributes: group-relative level credit

Use the score **level** rather than the within-state contrast, standardized within the GRPO group
(all rollouts of a group share the task, so the difficulty component cancels and what remains is
policy-attributable). This is the z = +10.06 signal from §8.3. It costs **one fifth of exp B**
because it needs only the executed action scored — no K-sampling, no extra generation.

Two hard constraints from our data: **no `Φ(s′) − Φ(s)` difference form** (wrong sign), and **never
the raw level across tasks** (difficulty detector).

*Our concern:* we do not understand *why* the level carries this signal. Our best guess is that it
is a state-quality proxy — successful rollouts reach better pages, and the RM scores actions on
better pages higher — which would make it a crude value function learned incidentally. If that is
right it is a legitimate credit signal (steps that reach good states get credit), but it rewards
*where you are* rather than *what you did*, and we cannot tell those apart in this data.

### Option 3 — outcome-trained step model

Train our own `V(s) = P(episode succeeds | state)` on our rollouts: every pre-action state labeled
with its episode's 0/1 outcome, ~13k episodes ≈ 100k states, split by task before expanding into
steps. Use `A_t = A + β·[V(s_{t+1}) − V(s_t)]`. Or a PRIME-style implicit PRM, where an auxiliary
policy trained only on outcomes yields per-step rewards as
`β·log(π_φ(a|s) / π_ref(a|s))` for free.

This is the principled endpoint — the Scalar RM never saw an outcome, and this is exactly the gap.
We have deferred it for cost and because we wanted the cheap experiment to settle first.

*Our concerns:* (a) the §8.3 result shows the *difference* form of a state-value-like quantity has
the wrong sign in our data, which is unsettling for a TD-style shaping term; (b) one binary label
per episode, no repeated states (live web), so V learns entirely by generalization; (c) V is a
function of the current policy and goes stale; (d) it is a real training project, not a hook.
Proposed gate before spending GPU time: held-out AUC on unseen tasks, group-standardized, and the
length-controlled z, benchmarked against the Scalar-RM level's z = 10.06. If it does not beat that
baseline offline, it does not get trained on.

### Option 4 — stop working on credit assignment

The honest case for this: our reproduction gains essentially everything by iteration 38 and is flat
thereafter; a second curriculum stage adds nothing; two credit experiments produced nothing. It is
possible per-step credit is simply not the binding constraint for this agent at this scale, and
that exploration, task difficulty distribution, the 20–25% abort rate, or the judge's own noise
matter more. We would rather hear this argued than discover it after a third run.

---

## 10. What we are asking you

Please engage with whichever of these you have a view on. Disagreement with our framing is more
useful to us than confirmation.

1. **Is per-step credit the right problem here at all?** Given ~10-step episodes, a binary terminal
   reward with a 5-rollout group baseline, and a policy whose gains saturate by iteration 38 — is
   credit assignment plausibly the bottleneck, or are we optimizing a second-order term? What would
   you look at to decide? (Option 4.)

2. **The termination asymmetry seems like an instance of a general problem.** Any per-step term
   that treats one action class differently — by masking, or just because the model's scores have a
   class-dependent mean — creates an incentive unrelated to the per-step judgement. Is per-action-kind
   centering the right general fix? What else does it break? Is there a formulation that is
   asymmetry-free by construction rather than by correction?

3. **How should β be set from measured signal quality?** We chose 0.4 as "the largest dose that
   cannot flip the sign of A". That is a safety argument, not a statistical one. With a signal whose
   measured association with outcome is AUROC ≈ 0.53, is there a principled way to pick the weight —
   something like scaling by an estimated signal-to-noise ratio, or a shrinkage/empirical-Bayes
   treatment of the per-step term?

4. **The Φ(s′) − Φ(s) sign puzzle.** The RM's score *level* separates successful from failed
   siblings strongly (z = +10 after length control), but its *step-to-step change* is **negatively**
   associated with success (−1.58, and the mean per-step change sign z = −5.10). We do not have an
   explanation we trust. Candidates we have considered: the score is dominated by page quality and
   good pages are reached early, so later steps drift down; or successful episodes end sooner and
   so have fewer late (low-scoring) steps; or the terminal `done` contaminates the last difference.
   Does this pattern suggest a specific artifact to test? It matters because it is what rules out
   potential-based shaping, which would otherwise be the theoretically clean option.

5. **Is group-relative level credit sound?** (Option 2.) Our worry is that it credits *state
   quality* rather than *action quality*, which in a GRPO group might be close to "you happened to
   land on a better page". Is that a legitimate credit signal or a subtle form of the difficulty
   leak we are trying to avoid? How would you separate the two with data we already have (we have
   per-step scores, per-group outcomes, and the trajectories)?

6. **Measurement.** Our chain is: offline AUROC → paired within-group association → checkpoint
   evaluation. The first is a weak proxy, the second is confounded by the intervention itself when
   run with β > 0, and the third costs ~2 days and can only resolve ~5 pp. Is there a
   higher-powered or cheaper design we are missing? We considered but have not implemented: finding
   rollout pairs in a group that share an action prefix and diverge, then regressing the outcome
   difference on the score gap at the divergence point — a within-group natural experiment. Is that
   sound? What is its failure mode?

7. **Off-policy accounting for expert iteration.** Executing the RM's argmax makes the behaviour
   distribution best-of-5 under the RM, not π, while the trainer uses π's log-probs. Our collaborator
   argues such data must go in a separate imitation loss rather than into GRPO as ordinary samples;
   the implementation we have puts it in as ordinary samples, confined to 1 of 5 rollouts and
   relying on PPO clipping. Who is right, and does the answer change if only 20% of the data is
   affected?

8. **If you had one 8×H100 node for a week**, what would you run? We can do roughly two
   60-iteration training runs plus their evaluations in that budget.

---

## 11. Practical constraints

- **Compute**: an 8×H100 node (shared, reservation-based) and an 8×L40S node. A 60-iteration
  training run is ~2 days; a 300-task evaluation arm is ~15 min per arm on 8 GPUs, plus judge API
  cost.
- **Cost**: GPT-4.1 judge calls for both training rewards and evaluation. Non-trivial but not the
  binding constraint.
- **Hyperparameters**: the reproduction's training hyperparameters (48×5 groups, lr 1e-6, batch
  256, 2 PPO epochs, 15-step horizon, T = 0.8) are **frozen by agreement** so that every arm is
  comparable to the reproduction. New knobs introduced by an experiment (K, β, τ, which rollouts
  are labeled) are free.
- **Time**: this is one person's research project with agent assistance, not a team. Prefer
  proposals that are one hook plus one run over proposals that require modifying the trainer.
- **Evaluation discipline** (learned expensively): same window, all 300 tasks per arm, task-level
  verdicts kept, paired McNemar, abort counted as failure so no denominator choice can create an
  artifact.

---

## 12. Glossary

| term | meaning |
| --- | --- |
| **OM2W** | Online-Mind2Web, the 300-task live-web benchmark we evaluate on |
| **v2 scorer** | successes ÷ tasks that neither aborted nor ended on a confirmed anti-bot block |
| **valid-only** | successes ÷ non-aborted tasks; runs 1–3 pp below v2 |
| **group** | the 5 rollouts of one task in one GRPO iteration |
| **turn sample** | one step of one trajectory, as an independent training example |
| **A** | the group-normalized outcome advantage of a trajectory, broadcast to all its turns |
| **A_t** | the advantage actually used for turn t after our modification |
| **Scalar RM** | Piotr's Bradley–Terry action reward model; one (state, action) → one scalar |
| **Selection ARM** | Piotr's generative selector; state + 5 candidates → index of the best |
| **exp A** | credit from standardizing the RM score across the steps of a trajectory |
| **exp B** | credit from comparing the executed action against 4 fresh samples at the same state |
| **c_t** | exp B's per-step credit, in [−1, 1] |
| **τ** | the score-gap scale: median \|r_i − r_j\| between genuinely different actions at one state |
| **premature termination** | the agent calls `done` early and the judge fails the episode |

## 13. Links

- Results summary with the evaluation figure: [`CREDIT_ASSIGNMENT_SUMMARY.md`](CREDIT_ASSIGNMENT_SUMMARY.md)
- OpenWebRL paper: https://arxiv.org/abs/2606.02031
- Piotr Teterwak's action reward models: https://github.com/piotr-teterwak/action-reward-models
- Zixian Ma's parallel ARM experiments (inference selection; filtered SFT/DPO; online turn-level
  bonuses — the last is the hard-selection analogue of exp B, with a smaller dose on ~9% of turns,
  and it also showed no reliable gain): https://github.com/zixianma/OpenWebRL/blob/arm/openwebrl/docs/ARM_SUMMARY.md
- PRIME (process rewards from outcome labels only): https://arxiv.org/abs/2502.01456
- Rewarding Progress (predictive process scores vs. progress advantages): https://arxiv.org/abs/2410.08146
