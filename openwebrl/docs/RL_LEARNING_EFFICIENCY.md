# Faster reward improvement: research review

Reviewed 2026-10-11. This is a literature review and implementation record; the reported paper gains have not been reproduced in OpenWebRL. The current RL comparison selected for implementation is the [five-arm action-credit study](#action-credit-implementation-20261011).

<a id="action-credit-implementation-20261011"></a>
## Five-arm action-credit study: first iteration10 pilot

**The selected study starts five fresh runs from the original OpenWebRL-4B-SFT actor and compares their iteration10 evaluations. Native GPU mechanics validation has been submitted; acceptance results are pending, and the five training pilots have not been submitted.** The online collector, judge integration and recovery of pending training batches are still under construction. Existing outcome-only runs retain their separate controllers and budgets.

The [historical CPU audit](rl_results/action_credit_implementation_audit_20261011.json) records **137 tests**, upstream-oracle results and the exact source hashes tested before GPU preparation. Its no-GPU/no-browser/no-judge statement applies to that audit. Subsequent native-adapter/harness CPU checks require a separate fresh audit; they do not amend the historical result or establish GPU acceptance.

| Arm | Planned collection per iteration | Planned terminal draws | Credit and comparison |
| --- | --- | ---: | --- |
| Independent-chain outcome GRPO | 48 task groups ×8 independent chains | 384 | Ordinary outcome advantage; contemporaneous reference |
| Tree sampling + outcome GRPO | 48 groups; M=2, N=2, L=2, K=4 | 480 planned; 384 selected slots | Ordinary outcome advantage; measures the tree-collection package |
| Tree-GRPO | Same tree schedule and selection as tree-outcome control | 480 planned; 384 selected slots | Task-pooled leaf advantage plus within-tree advantage; isolates added tree credit within this pair |
| Continuation-Q action credit | 384 independent chains plus up to8 states ×5 fixed response candidates ×8 continuations | Up to704 | Outcome objective plus equally weighted candidate-Q terms; combines local credit with additional collection |
| GiGPO, strict adaptation | Same 48×8 independent chains as reference | 384 | Episode advantage plus state/history/horizon-matched step credit |

M is the number of initial independent chains per task, N the number of expansions per tree in each round, L the number of rounds, and K the terminal-draw slots selected per tree. Thus the tree schedule plans `2 × (1 + 2 × 2) = 10` leaves per task and selects8 without using rewards. The browser adaptation fixes a uniform ranking of all five planned slots per tree before dispatch and selects four, including any later invalid or missing slots. It never substitutes another leaf or task to fill a failed selection. A missing initial root leaves its dependent slots missing; no state or edge is fabricated. Both tree arms use this same rule. Unlike upstream's terminal-only selection, it preserves failed admissions in the selection population; report selected, valid, invalid and missing counts because valid-only conditioning can still change coverage.

In each continuation-Q panel,6 draws per candidate fit its target and2 are held out: up to240 fit and80 holdout continuations per iteration. Failed source admission or incomplete physical candidate collection withholds that panel's local term while preserving all planned slots and any collected outcomes; there is no replacement after inspecting rewards. The ordinary outcome objective uses only the 384 independent base chains; additional panel continuations estimate local Q and never enter the base advantage statistics or loss.

**Continuation-Q does not share the Tree-GRPO collector.** Its fixed action panels add both data and cost, so this five-arm design cannot attribute a Q-versus-chain difference solely to local credit. A separate matched-panel outcome-only control would be needed for that attribution. Tree-GRPO versus tree-outcome is the directly matched collection comparison.

The common task order is frozen before collection:48 prespecified groups per iteration from the original2,102-task RL pool, seed42, with fresh optimizer, scheduler and task cursor. Tasks are not replaced because their sampled rewards tie. Invalid/missing outcomes remain explicit and are excluded from valid-only targets; valid zero-advantage examples retain their loss denominators. This differs from historical runs that oversampled until48 nonzero-variance groups were accepted, so those runs provide timing context rather than the contemporaneous control. The original RL pool is not interchangeable with Piotr's ARM training split.

All five arms use actor temperature0.8, top-p1, top-k−1, up to15 browser turns,1,024 response tokens and32,768 context tokens; GPT-4.1/action_history terminal binary-success judging (native combined/format rewards remain separate diagnostics); response-mean loss; mean/std outcome normalization with zero singleton advantages; PPO clipping widths0.20/0.28, no dual clip, and zero KL/entropy coefficients. Local-Q and GiGPO step coefficients are1; GiGPO uses discount1, trajectory-weighted episode statistics and strict state/history/image/horizon anchors. These are controlled OpenWebRL adaptations, not full reproductions of the papers' recipes. Report GiGPO's repeated-anchor and nonzero-step-credit coverage because strict matching may yield mostly singleton groups.

Evaluation is one full300-task Online-Mind2Web cohort per iteration10 checkpoint, plus one shared contemporary iteration0 evaluation: local browser, actor temperature0, top-p1/top-k1,4,096 response tokens,30 turns, and GPT-4.1/action_history. Preserve every rollout and verdict disposition; report success counts, overall and valid-only rates, invalid counts and paired-task uncertainty. Continuation holdouts are separate from these held-out evaluation tasks. The1-seed pilot provides preliminary comparisons.

Each PPO pass deterministically shuffles all eligible base response occurrences into windows of 256. The final short window is retained with zero-loss padding and weights of 1/256, without renormalizing the tail. Shared-prefix occurrences retain separate clipped terms. Q evaluates all eligible candidate panels in each window, scaling the entire local mean by the real base-occurrence count divided by 256. Its extra candidate forwards are charged. Full pending batches, window schedules and native RNG/checkpoint identities are saved before optimizer work; recovery restores the previous complete update and replays the saved batch.

Equal 10 collection/update rounds are **not** equal 10 Adam steps or equal spending. Hold the optimizer recipe and two PPO passes fixed, then report actual optimizer updates, trained tokens, GPU-hours, browser/replay interactions and judge expenditure, including failed attempts. Endpoint gains at0/10 do not establish an exact time-to-target speedup.

<details>
<summary>Resource ceilings and current validation status</summary>

| Stage | GPU request | GPU-hour ceiling | API-dollar ceiling | Status |
| --- | --- | ---: | ---: | --- |
| Native mechanics validation | 8 H200,64 CPUs,960GiB;6 hours total | 48 | 0 | Submitted; actual GPU acceptance pending |
| Separate browser/judge verification, including shared iteration0 evaluation | Within the proposed validation envelope | Included above | 25 | API cap proposed, not approved; no paid judge verification launched |
| Each of five iteration0→10 pilots, including evaluation10 | 8 H200,64 CPUs,960GiB;24 hours per arm | 192 per arm | 200 per arm | Selected GPU ceilings; pilots not submitted; API caps proposed, not approved |

The proposed API total is **$1,025**; the GPU ceilings total1,008 GPU-hours across validation and five pilots. Each cap includes all attempts and retries, with no transfer from ongoing outcome-only approvals or between arms. These ceilings are limits, not a guarantee that iteration10 fits. Global storage reservations and a feasible forward quota forecast are also required; no new cleanup is authorized by this study.

The isolated GPU gate targets TP2/DP4 and TP8/DP1, actual-model distributed gradient parity, all five objective updates, full256-response accumulation, a separately labeled synthetic32K context with real image tensors, and model/Adam/scheduler save–perturb–reload. Its archived browser fixture contains three real sampled rows, with the longest eligible context11,746 tokens. The fixture and diagnostic ancestry/rewards are mechanics inputs, not proof of fresh on-policy collection or physical browser replay. W&B validation/evaluation belongs in `openwebrl-evals`; actual training belongs in `openwebrl`.

</details>

**CPU numerical checks:** hand-computed cases, ties/singletons, unequal tree/draw sizes, invalid/missing outcomes, permutation invariance, source/holdout rejection, token-mask gradients, a small optimizer update, and simulated uneven rank/microbatch reduction. The upstream oracle executes pinned official functions. CPU rank simulation does not establish distributed GPU correctness.

**State checks:** collection policy, objective settings, group/tree membership and protocol hashes are immutable; planned candidate/draw roles and all invalid, missing and holdout slots must remain in the sealed inventory. Retries preserve reservations and actual overages. A crash after optimizer intent leaves `training_pending`, requiring reconciliation with a bound durable checkpoint; a consumed-input flag alone cannot prove that training occurred. These interfaces are not an end-to-end validated online controller.

<details>
<summary>Estimator conventions, source fidelity and pitfalls prevented</summary>

The numerical module pins [Tree-GRPO commit 19bf3fa](https://github.com/AMAP-ML/Tree-GRPO/tree/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59) and [GiGPO commit 20bd331](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8). These are the agent methods, not the separate diffusion TreeGRPO.

- **Tree-GRPO:** the released `tree` dispatch sums two independently normalized advantages: selected training leaves pooled within a task group, and selected training leaves within each tree. It does not normalize tree means, normalize the sum again, or invoke the separate `tree_2norm` ablation. Upstream uses sample standard deviation plus epsilon; its singleton behavior differs from our explicitly selectable zero-singleton control. The released full-leaf loss repeats shared prefixes. A shared forward pass can be reused, but opposite-sign leaf advantages must remain separate through nonlinear PPO clipping. [Paper](https://arxiv.org/html/2509.21240v3)
- **Tree collection:** the original terminal-only offline planner retains its upstream oracle. The browser pilot uses the explicit planned-slot selection rule above, with uniform expansion parents sampled with replacement from available roots/nonterminal nodes and complete round barriers. The local ordering and keyed random seeds do not reproduce upstream's exact seed-to-node sequence. All generated artifacts and missing slots remain retained; training selection is separate from fit/holdout assignment and composes all48 task groups into one update. The online mapping and physical replay still require runtime acceptance.
- **Loss fidelity:** response-mean, trajectory-mean and token-mean reductions are distinct, named choices. The Tree-GRPO launcher uses token-mean dual-clipped PPO plus a separate KL term. The reference supports standard and dual clipping; oracle comparisons cover normal finite ratios. It deliberately fails on exponent overflow instead of silently reproducing the upstream log-ratio clamp. A controlled OpenWebRL adaptation is not a reproduction of the complete upstream recipe.
- **Local Q:** each candidate's target is its mean valid terminal return; subtract the mean across candidates, giving candidates equal weight despite unequal valid-draw counts. Missing/invalid draws are excluded, never converted to failures. If a prespecified candidate has no valid draw, withhold that state's local advantage rather than silently compare fewer alternatives. Zero-advantage candidates and tied states retain their denominators. Every candidate contributes once to the local objective; extra continuation draws improve its estimate without multiplying its gradient.
- **What the intervention means:** the sampled thought/action response remains in continuation history. Local Q therefore supervises the full response. Tool-call-only credit would be a separately labeled surrogate. Fixed-SFT archived panels are not on-policy training data for a later actor, even if their log probabilities are recomputed.
- **GiGPO:** the adapter exposes the published observation-anchor approximation and a stricter state/history/image/horizon match, scoped within task groups. Report repeated-anchor and nonzero-advantage coverage; a conservative match can yield many singleton groups. The released code's step-weighted episode statistics are separate from equal-trajectory statistics. Normalization and state matching must be pinned rather than silently changing the outcome control. [Paper](https://arxiv.org/html/2505.10978v3)
- **Holdouts:** continuation holdout draws from a training task are distinct from held-out evaluation tasks. Both are excluded from training targets, group normalizers and loss. A panel cannot omit unfavorable executed continuations. Candidate identities are sampled events: duplicate response text is not permission to merge independently sampled candidates.

</details>

<details>
<summary>Code, reproduction and remaining runtime gates</summary>

The implementation is isolated in `openwebrl/action_credit*.py`: numerical estimators, response graph/loss, replay and sealed-collection binding, objective adapter, sampling/selection planner and campaign ledger. Tests are `tests/test_action_credit*.py`. A separately frozen native validation source connects the adapter to the real trainer for diagnostics. Production collector/judge integration and pending-batch recovery are still being implemented and have not passed end-to-end online acceptance.

Run `python -B scripts/validate_action_credit.py --require-upstream --upstream-dir /tmp/action-credit-upstream --report /tmp/action-credit-audit.json` with a Torch/NumPy environment. Prepare the three upstream files using the immutable repository/commit/path and SHA256 entries in `tests/test_action_credit_upstream.py`; the runner performs no downloads. Omitting upstream fixtures explicitly skips that oracle and cannot satisfy `--require-upstream`.

Before launch, implement and validate the online collection/trainer adapter: freeze and verify the current actor version; bind each parent response, resulting observation and child multimodal actor context; capture exact sampled tokens and independently verified old-policy scores; preserve fixed candidate sets and fit/holdout draw roles. Pin sampling-request-to-draw identities before dispatch and independently verify terminal artifact linkage. Sealed terminal metadata prevents a later caller from permuting a retained mapping, but does not authenticate the collector that wrote it. A receipt's consistent hashes establish identity, not physical browser-state equivalence. Strict visible replay does not clone hidden remote server state.

The submitted mechanics gate addresses actual distributed gradient scaling, accumulation and model/optimizer/scheduler restoration. Production task-cursor restoration, physical replay and crash recovery around collection/training commits still require separate verification. The ledger's externally supplied checkpoint receipt is not itself a trainer transaction. Enable each pilot only after its runtime gates, concrete resource/API authority and storage forecast pass. The separately approved outcome-only continuation budgets do not fund this study.

</details>

## Research motivation and earlier experiment options

**The highest-upside research direction is to teach corrections at consequential decisions, supported by reliable comparisons between alternative actions.** Verified-success replay is the simpler near-term RL intervention. Adaptive task selection has supporting GUI evidence, but its benefit over our existing outcome-variance filter remains unmeasured. For wall-clock improvement, repeated context processing during optimization deserves attention alongside browser throughput.

The target should be **held-out task success per total training cost**. Reward on an adaptively sampled training batch can rise because the task distribution became easier. Better final accuracy at a fixed number of steps or rollouts is also weaker evidence than reaching a fixed accuracy sooner at matched total cost.

## Which ideas have the strongest evidence?

Paper results below are within-paper comparisons, not a common benchmark or a prediction of our gains. Percentages are the paper's task-success/accuracy metrics; percentage-point deltas are absolute differences.

| Direction and source | Controlled result | What it establishes; important limit | Assessment for OpenWebRL |
| --- | --- | --- | --- |
| **Localized hindsight correction.** [HinT-SD, May 2026](https://arxiv.org/html/2605.17873v1), Table 1 | Qwen3-4B BFCL Avg@4: GRPO **31.56%**, HinT-SD **41.88%**; AppWorld **7.49%**, **18.46%** | Better learning from matched rollout budgets. Main method uses a self/EMA teacher. Tool/API environments; total compute is not matched. Its **2.26×** step-time improvement is against dense OpenClaw-RL feedback, not GRPO. | Highest potential for extracting useful corrections from failed browser episodes. Requires evidence that the feedback identifies an actionable mistake. |
| **Several candidate actions per visited state, scored by a critic.** [Android Coach, April 2026](https://arxiv.org/html/2604.07277v1), Table 2 and Figure 4 | UI-TARS-7B AndroidLab: GRPO **34.8%**, Coach **39.4%**; AndroidWorld **38.2%**, **41.1%**. Reports **1.4×** efficiency at matched success | Direct GUI evidence under matched online training time. Additional candidate actions are scored without execution. Substantial critic/PRM preparation is not established as included in that timing. | Strong motivation for our action-comparison study. A reliable critic could amortize expensive browser interactions, but adds scoring and optimizer work. |
| **Adaptive positive replay.** [MobileRL, September 2025](https://arxiv.org/html/2509.18119v1), Figure 3 | AndroidWorld: full method **71.1%**; without adaptive replay **63.6%**; without failure-curriculum filtering **64.8%** | On-policy reward curves exclude replayed successes. Component evidence at a fixed step budget, not a measured time-to-target speedup. Replay includes selection/pruning, so this does not isolate naive replay alone. | Most practical first RL ablation: reuse verified successes when current groups fail, with explicit off-policy handling. |
| **Task selection near the current learning frontier.** [ScaleCUA, July 2026](https://arxiv.org/html/2607.11185v1), Table 4 | OSWorld: full method **68.7%**; without frontier sampling **63.7%** | Held-out benchmark evidence beyond a changed training-reward curve. EMA task success guides selection, with uniform exploration. Initial profiling costs rollouts; no isolated sampling speedup in total cost. | Predict informative groups before collection. Retain exploration and refresh stale difficulty estimates. |

Our existing filter removes constant-reward groups after collection. Predictive selection can avoid some of that expenditure, but does not automatically increase learning per accepted update. Replay and corrective supervision address a different limitation: the absence of a useful positive contrast or localized correction in failed experience.

## Recommended experiment order

**Scientific priority: outcome-grounded action correction.** Use the [branching study](ARM_FORMULATIONS.md#branching-experiment-map) to identify decisions where alternatives have distinguishable valid continuation outcomes. Generate concise comparisons grounded in observed consequences, then train the student from the original pre-action context. Future observations and outcomes may inform supervision, but must not enter the student's deployment input.

The smallest informative comparison extends [action-comparison SFT](ARM_SFT.md#arm-comparison-sft-20261007):

| Arm | Added supervision | Question answered |
| --- | --- | --- |
| A | Existing action-filtered SFT | Reference |
| B | Verified corrected action at selected states | Does better action supervision help? |
| C | The same states and corrected actions as B, plus a grounded comparison explanation | Does comparison reasoning add value beyond the corrected action? |

Keep task/state splits fixed, preserve an independent continuation holdout, and abstain on uncertain action rankings. Compare B/C with matched examples and report both token exposure and total compute; longer explanations create extra training work. A later critic-guided RL experiment should first demonstrate accurate ranking on fresh continuation outcomes.

**Practical RL priority: test verified-success replay, then task selection as a separate factor.** Preserve the terminal success definition and track original behavior-policy probabilities for reused data; merely limiting replay age is not a correction for off-policy bias. Start new scientific variants from the agreed iteration-zero initialization with fresh optimizer/scheduler/cursor. Keep the ongoing outcome-only continuations unchanged.

**Systems priority: profile repeated visual/text context processing.** First validate packing or batching changes on identical saved batches, preserving loss masks, update membership, global batch and PPO epochs. Treat changes to the model's visible history as a separate intervention requiring quality validation.

<details>
<summary>Timing evidence and why rollout savings are not total speedups</summary>

[SDPO, January 2026](https://arxiv.org/html/2601.20802v1), Table 3, supplies actual wall-clock learning evidence: Olmo3-7B reaches a five-hour GRPO chemistry result in 30 minutes. The closer tool-use comparison is smaller: Qwen3-8B reaches **68.0%** versus GRPO **64.9%** after one hour. These are best achieved average accuracies within the time budget, with hyperparameters selected using five-hour accuracy. The tool task is single-call ToolAlpaca, not long-horizon browsing; the 10× chemistry result is not a general agent-training multiplier.

ScaleCUA's **2.83×** timing claim belongs to visual-context segmentation, not frontier sampling: reported training-step time falls **750→265 seconds**, including actor update **485→154 seconds** and reference scoring **241→88 seconds**. This changes how trajectory contexts are processed; it is not evidence that our full collection/evaluation pipeline becomes 2.83× faster. [Source, §4.3](https://arxiv.org/html/2607.11185v1#S4.SS3)

Our completed extension-cycle aggregates are below; [machine-readable timing evidence](rl_results/learning_efficiency_profile_20261010.json). These windows exclude milestone evaluations, startup/restarts, storage-review waits and failed partial collections. Native timer boundaries and iteration windows differ, so this is a bottleneck diagnostic, not a causal comparison between recipes.

| Run | Completed iteration window | Cycles | Mean collection, min | Mean training, min | Mean cycle, min | Training share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Batch56 outcome-only | 21–34 | 14 | 20.06 | 25.41 | 46.09 | 55.15% |
| Uniform G8 outcome-only | 11–21 | 11 | 24.79 | 45.38 | 70.85 | 64.05% |

Holding all other work fixed, halving only collection time implies approximately **1.28×** and **1.21×** more completed cycles per hour, respectively: `cycle / (cycle − 0.5 × collection)`. This illustrative calculation is not an observed speedup or a prediction of reward growth.

</details>

<details>
<summary>Preparation costs, related papers, and transfer risks</summary>

- **Android Coach:** uses 20,000 action labels, GPT-4o annotations/filtering, separate critic preparation and PRM training. Critic training without suitable pretraining is unstable. In its action-count ablation, four candidates cost 1.62× the one-action training time; eight cost 2.18× with little further improvement. Its demonstration-based labels can penalize valid alternative actions. Our independent continuation outcomes would be an important validation layer. [Method and ablations](https://arxiv.org/html/2604.07277v1)
- **PivotRL, March 2026:** training at informative decision points reaches comparable SWE-Bench end-to-end-RL accuracy with roughly four times fewer rollout turns and 5.5 times less cumulative rollout time. This excludes some preparation/training costs and depends on programmatic local action verification. It supports investigating selected decision states, but that verifier is not directly available on live websites. [Paper](https://arxiv.org/html/2603.21383v1)
- **Tree-GRPO, ICLR 2026:** at approximately four full-trajectory equivalents, Qwen2.5-3B multi-hop QA rises from 31.8% with chain GRPO to 36.8% with trees. Experiments use search tools; browser replay/reset costs and failures could erase shared-prefix savings. [Paper](https://arxiv.org/html/2509.21240v2)
- **DPS, March 2026:** predictive sampling achieves 63.13% average math accuracy with 287k rollouts/39 hours versus dynamic sampling's 62.42% with 1,147k/73 hours. Uniform sampling takes 30 hours for 59.31%. Thus its efficiency advantage depends on the comparator; it is not simply faster than uniform sampling. [Table 1](https://arxiv.org/html/2603.10887v1)

</details>

<details>
<summary>Evaluation and accounting needed to establish faster learning</summary>

Use the same initialization, fixed held-out task cohort and evaluation protocol. Plot task success against cumulative allocated GPU-hours and browser/judge/teacher cost, and report time to a predeclared success threshold. Charge critic preparation, rejected rollouts, replay failures, teacher generation/scoring and all training attempts. Keep overall and valid-only success separate, with counts and invalid rates next to percentages.

Report fresh on-policy training reward before filtering and separately from replayed data. Compare matched tasks when changing the sampling distribution. Use paired task-level uncertainty for held-out differences; repeated evaluation draws do not replace independent training seeds. Distinguish accuracy per rollout, accuracy per optimizer update and accuracy per wall-clock hour.

This review proposes tests, not an allocation request. Existing compute/API caps and the paused adaptive run remain unchanged.

</details>
