# ARM branching: action quality, execution evidence and selector training

**Continuation outcomes contain useful action-selection signal; the value of immediate post-action evidence is still unresolved.** The studies below address different parts of that question. Their principal controlled results are:

| Question | Matched states | Paired difference, percentage points | 95% interval | Takeaway |
| --- | ---: | ---: | --- | --- |
| Does Luna select better after seeing immediate execution evidence? | 119 | +2.24, after − before | [−0.28, +4.95] | No clear advantage yet; historical invalid-as-zero score |
| Can old continuation outcomes select better than uniform on fresh outcomes? | 58 | +8.51, select on three − uniform | [+3.45, +14.37] | Evidence of outcome signal; gains over actor-first and Luna remain uncertain |
| Does selecting with 24 continuations beat selecting with eight? | 14 | +3.52, select with 24 − select with eight | [−0.12, +7.35] | Inconclusive on the same fresh, valid-only holdout |
| Do teacher-trained selectors pick better actions than uniform? | 138 | Piotr ARM +2.48; outcome-selected on five draws +5.32 | [−0.14, +5.21]; [+3.09, +7.62] | Only outcome selection is clearly above uniform ([details](#selection-head-20261009)) |
| Do execution evidence or a stronger teacher improve picks? | 194 | Sol after − before −0.31; Luna after − before +0.15; after: Sol − Luna +0.41 | [−2.40, +1.68]; [−1.86, +2.10]; [−1.17, +1.89] | No measurable gain from either; all teachers ≈ +3 pp over uniform ([2×2](#teacher-labels-211-20261011)) |
| Does multi-step lookahead help? | 194 | Luna lookahead (answers hidden) − immediate +1.98 | [+0.08, +3.98] | Modest gain; reaches the outcome reference; showing final answers hurts ([details](#lookahead-labels-20261011)) |
| Does the lookahead gain replicate on fresh Piotr-matched states? (prespecified primary test) | 121 | Luna lookahead (answers hidden) − immediate +4.75 | [+1.58, +8.27] | Yes; lookahead also beats uniform (+4.19) and is within 1 pp of the outcome reference ([stage a](#lookahead-stage-a-20261012)) |

The first interval resamples states; the others resample task groups. All are exploratory, pointwise intervals. **Students trained on branching outcomes or on matched pre- and post-action labels have so far stayed at uniform** ([student results](#teacher-labels-211-20261011)). Selectors trained on teacher choices (Piotr ARM and the [actor-based selectors](#selection-head-20261009)) do not clearly beat a uniform pick on held-out draws of 138 eight-draw states, while selecting by outcomes on five draws does: +5.32 pp [+3.09, +7.62].

<a id="branching-experiment-map"></a>

## Two data families, each with a continuation extension

An extension adds **continuations after the same five fixed candidate actions**, not additional action proposals. The historical family and the new pilot are separate collections; their extensions are not independent replications.

| Family | Initial states | Initial draws per action | Added draws per action | States with all added draws | Primary outcome rule |
| --- | ---: | ---: | ---: | ---: | --- |
| [Historical 124 + extra two](#arm-continuation-branches-results-20261007) | 124 | 3 | 2 | 58 | Committed invalid outcomes count as zero |
| [New pilot 32 + extra 24](#branch-selector-scaleup-20261008) | 32 | 8 | 24 | 18 | Invalid outcomes excluded; valid-only success |

The historical target was 149 states; 124 were collected and frozen. The new pilot collected its full 32-state target from the released WebVoyager-source **training pool**, not the WebVoyager evaluation benchmark. It introduced multiple prespecified states per task and train/seen-task/unseen-task roles. The extra24 analysis qualifies only 14 of its 18 complete grids because every candidate needs valid selection and held-out observations.

**Group these records by lineage; do not pool their success rates.** Cohort, task mix, collection time, holdout draws and invalid-outcome policy differ. In particular, the pilot's 48.50% and extra24's 53.32% are not a controlled improvement. The matched eight-versus-24 comparison is 49.80% versus 53.32% on the same 14 states. Both extensions ended partially; no further collection is active.

[Historical teacher comparison](#arm-continuation-branches-results-20261007) · [Fresh extra-two test](#branch-extra2-heldout58-20261008) · [Extra24 common-holdout test](#branch-extra24-20261008) · [Teacher-trained heads](#selection-head-20261009) · [Remaining experiments](#branch-post-to-pre-study-20261008)

<a id="branching-design"></a>

## What is measured

A **state** is the task, screenshot and action history before the candidate action. A **candidate** is one of five fixed actor responses; duplicate actions remain in the panel. A **continuation/draw** executes that candidate in a separately replayed browser and then follows the frozen SFT actor to a terminal outcome. Only the root-action choice differs between selection rules; these are conditional continuation rates, not full-benchmark pass@1 or episode pass@k.

| Selection rule | Information used to choose the root action | Interpretation |
| --- | --- | --- |
| Actor first / uniform | First candidate / random index among all five | Baselines scored on the same saved outcomes |
| Before / after critic | State and candidates / additionally immediate execution evidence | Direct test of the value of post-action evidence; neither sees future returns |
| Outcome-selected reference | Separate continuation outcomes for each candidate | Privileged offline reference, evaluated on held-out draws; not a trained critic |
| Same-data hindsight maximum | The same outcomes used for selection and scoring | Optimistically biased diagnostic, not a true upper bound |

“Select on *m*, score on *h*” counts draws **per candidate action**. Exact uniform averaging handles tied winners without looking at the holdout. In the valid-only family these are draw slots: invalid observations reduce the usable denominator. Score each candidate's valid held-out mean, then weight states equally. Native valid unjudged action-limit zeros stay failures; missing/replay-rejected records stay missing. Saved judge labels can credit partial or misinterpreted success, so artifact verification does not establish semantic correctness.

<details>
<summary>Original three-draw diagram and shared actor/judge protocol</summary>

![Original historical design: replay the saved state, execute each of five candidates, and collect three separate SFT continuations per candidate](arm_results/rl_integration/branching-experiment-design.svg)

The diagram depicts the historical five-by-three design. The new pilot uses eight draws per action; its extension targets 32 in total. Before, after and uniform choices reuse the same outcomes within each comparison. Uniform requires no separate rollout set. [PNG](arm_results/rl_integration/branching-experiment-design.png).

All families use official OpenWebRL SFT with temperature 1.0, top-p 0.95, top-k off and 4,096 response tokens, and canonical o4-mini/AgentTrek terminal-success labels. The 30-action budget includes the prefix. Strict observable-state replay checks are preserved; they do not guarantee identical hidden state or eliminate website drift. The new pilot requires forty ready contexts; its candidate intervention retains the complete sampled reasoning/action response in the continuation history.

</details>

<a id="arm-continuation-branches-results-20261007"></a>

## Historical family: 124 states, three draws, then two fresh draws

### Does immediate execution evidence help the critic?

Both index-only Luna-high conditions choose among the same candidates without seeing continuation returns. After-execution additionally sees each candidate's immediate execution evidence. The paired comparison uses **119 states and 1,785 outcomes**; four invalid teacher panels and one unavailable-after panel are excluded only here. Their states remain in the 124-state outcome analyses. Committed invalid outcomes count as zero.

| Critic input | Matched states | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Before execution | 119 | 35.11% | [27.73%, 42.67%] |
| After immediate execution | 119 | 37.35% | [29.88%, 45.10%] |

The paired after-minus-before interval in the overview includes zero. After-execution judgments saw the immediate result of execution draws 0–2, which are also the first steps of the continuations scored here. Scoring only on draws 3–7 avoids that overlap. On 57 states with such draws, after minus before is +1.77 pp [−0.51, +4.40]; after minus uniform is +4.61 [+0.39, +9.16]. Choosing by outcomes of draws 0–2 gives +7.22 [+2.88, +12.21] over uniform ([aggregate](arm_results/rl_integration/selection-outcomes-20261010.json)). Repeated teacher choices average over the same saved outcomes; they add no rollout data. Repeat consistency and label changes are diagnostics, not evidence of better action success. [Original aggregate and paired audit](arm_results/rl_integration/continuation-fixed124-20261008.json).

<details>
<summary>What one Luna judgment sees, and why there are 4 before and 10 after judgments per state</summary>

Each call is one independent GPT-6 Luna request at high reasoning effort. The request contains the task, the last five history entries, the original screenshot and the five candidate responses, and Luna returns one candidate index (`openwebrl/arm_branch_teacher.py`). An after-execution call also receives, for every candidate, the screenshot and tool feedback **immediately after that candidate's action in one execution draw**. No call sees later steps of a continuation, final screenshots or rewards.

| Condition | Evidence | Calls per state |
| --- | --- | ---: |
| Before | State and candidates only | 3 repeats + 1 with candidates shuffled = 4 |
| After, execution draw 0 | Immediate results from draw 0 | 3 repeats + 1 shuffled = 4 |
| After, execution draw 1 | Immediate results from draw 1 | 3 repeats |
| After, execution draw 2 | Immediate results from draw 2 | 3 repeats |

- **Repeats** send identical input, to measure Luna's own variability.
- **Shuffled calls** test whether Luna favors candidates by position.
- **The three draws** show the same actions' immediate screens from separate browser replays, which can differ.
- **Coverage.** Every state planned 14 calls. All 14 are valid on 119 states. Two states have 4 before and 6 after judgments; three have 4 before and none after.

Reported "before" and "after" scores average each condition's valid calls per state. Because after-evidence comes from draws 0–2, which are also the first steps of the continuations scored on those draws, the [leakage-free check](#arm-continuation-branches-results-20261007) scores only draws 3–7. Applying this script unchanged to an eight-draw panel would request after-judgments for all eight draws. A labelling manifest must instead fix the evidence draws in advance and keep held-out draws out of the teacher's view.

</details>

<a id="branch-extra2-heldout58-20261008"></a>

### Does selection from old outcomes improve fresh continuations?

**The extra-two pass is audited partial:** 58 of 124 states passed strict replay, adding 580 outcomes. All 1,860 original records are unchanged, giving **2,440 saved outcome records**. The remaining 66 states failed reconstruction; their 660 missing continuations are not counted as zero. The approved 3,100-record target and original 149-state target remain incomplete.

Each row below uses the **same 58 states from 58 task groups, with 580 fresh records and two draws per candidate**. The outcome-based choice uses only the three original draws. Tied best actions receive their exact uniform expected score. Luna’s score averages the fresh outcomes of its three saved before-execution choices; no new teacher calls were made. Committed invalid outcomes count as zero.

| Root-action choice | States | Continuation success | 95% task-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first candidate | 58 | 26.72% | [17.24%, 37.07%] |
| Uniform over five candidates | 58 | 26.03% | [17.76%, 34.83%] |
| Saved Luna before execution | 58 | 31.03% | [20.11%, 42.24%] |
| Select on original three; score fresh two | 58 | 34.54% | [23.97%, 45.23%] |

The outcome-based choice gains **+8.51 percentage points over uniform [3.45, 14.37]**. Its gains over actor-first, **+7.82 points [−0.26, 16.84]**, and Luna, **+3.51 points [−2.84, 10.80]**, remain unresolved. These exploratory, pointwise intervals support a useful continuation-outcome signal; they do not establish a trained selector’s performance or superiority to teacher-label training. [Aggregate estimates and audit hashes](arm_results/rl_integration/continuation-extra2-heldout58-20261008.json).

**Replay selection materially changes the cohort.** Using only original draws, actor-first scored 17.82% on these 58 states versus 40.40% on the 66 that later failed reconstruction. Consequently, comparing fresh 34.54% directly with the earlier 124-state LOO result of 38.12% would confound cohort composition, draw count and collection time. The accepted states span candidate decisions 1–3/4–9/10–15 with 36/11/11 states respectively. Frozen observable-state checks do not guarantee identical hidden state or eliminate website drift.

<details>
<summary>Five-draw sensitivity, ties, uncertainty and judge limitations</summary>

These secondary rows pool old and fresh draws on the same 58 states, yielding 1,450 outcome records. Five-fold LOO selects with four draws and scores the fifth, averaging all overlapping folds. It mixes collection times; the temporally separated old-three/fresh-two comparison above remains primary.

| Root-action choice | States | Pooled continuation success | 95% task-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first candidate | 58 | 21.38% | [13.10%, 30.35%] |
| Uniform candidate | 58 | 24.97% | [16.76%, 33.66%] |
| Select on four; score held-out fifth | 58 | 33.95% | [23.94%, 44.16%] |
| Same-data five-draw hindsight oracle | 58 | 41.38% | [31.03%, 51.72%] |

The hindsight row chooses and scores on the same outcomes and is optimistic. Original-three selection ties at the top in 39/58 states; 29/58 give every candidate equal success counts, including 25 all-zero panels. Five-draw hindsight still ties in 35/58 states. Five-fold training maxima tie in 187/290 folds. Ties and duplicate candidates are retained.

Every state has equal weight. Ten thousand bootstrap draws, seed 20261007, resample whole task groups with all their states and normalize by sampled state count. The 58 groups each contain one state here. Intervals are pointwise, exploratory and unadjusted for multiple comparisons. Exact tie expectations avoid choosing winners from held-out outcomes. Valid-only sensitivity, with its own weighted denominators, is included in the aggregate.

The official SFT actor and decoding remain temperature 1.0, top-p 0.95, top-k off and 4,096 response tokens, with the original 30-action budget including replayed prefixes. Canonical o4-mini/AgentTrek labels remain unchanged. Of the 580 fresh records, 523 are valid and 57 invalid; 303 received a judge call and 220 valid records retain native unjudged zero outcomes. Qualitative checks found positives that conflate website statistics or accept a different language edition. These scores measure the frozen judge’s operational reward, not independently established factual or instruction-complete success. Source and identity checks support seed assignment; an independently echoed request seed was unavailable.

</details>

<details>
<summary>Partial collection, reconstruction failures and final accounting</summary>

| Coverage | States | Saved outcome records |
| --- | ---: | ---: |
| Full five draws per action | 58 | 1,450 |
| Original three only | 66 | 990 |
| Preserved combined data | 124 | 2,440 |
| Approved combined target | 124 | 3,100 |

All 124 states were processed once under the frozen reconstruction rule. The first failures on 66 rejected states were 58 visible-snapshot mismatches, five raster mismatches, one page mismatch, one prefix-capture timeout and one initial-navigation timeout. Their 147 unreleased attempt receipts and 513 untouched queue slots remain evidence of missing continuations. They are not added to the outcome dataset. One historical post-commit release-read invalid remains among the original records as previously audited; no candidate execution is invented.

The sole extra-two allocation consumed 14,238 of 28,800 approved seconds on four H200s / 32 CPUs / 480 GiB; 14,562 seconds remain unused and are not transferred. All 727 attempts are charged. Extra-two used 727 browser starts, 6,788 SFT calls and 303 judge calls / $2.406679, with no teacher calls. Combined with prior charges, the shared ledger contains 6,923/9,000 browser starts, 40,512/110,000 SFT calls, 1,438/6,600 judge calls / $11.5469541 of $25, and 1,698/2,200 Luna calls / $6.5299025 of $15.

Scheduler and W&B finished. Independent audits reconciled artifacts, all-attempt accounting, the original-record hashes and controller-owned actor/browser teardown. Post-allocation SSH was unavailable, so teardown does not claim an independent physical node scan. The pass is closed as audited partial, with full-target completion false. The frozen rule forbids replacement or outcome-based replay after reconstruction rejection; no automatic rerun or relaxed replay was used. The separately approved 32-state pilot remains a separate experiment and budget.

</details>

<a id="branch-original-three-details"></a>

<details>
<summary>Original three-draw reference, depth, repeat controls and judge limitations</summary>

All 2,090 fixed candidate tasks were considered under the frozen replay checks and depth quotas, yielding 124 of the 149 target states: 1,860 outcomes, with 1,710 valid and 150 invalid. The original 149-state target remains incomplete. These estimates use only the original draws, before the extra-two pass.

### Original 124-state outcome comparison

| Root-action choice | States | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first sample | 124 | 29.84% | [22.85%, 37.10%] |
| Uniform candidate | 124 | 32.42% | [26.13%, 38.98%] |
| Select on two continuations; score held-out third | 124 | 38.12% | [31.06%, 45.31%] |
| Same-data hindsight action oracle | 124 | 50.00% | [42.47%, 57.53%] |

Leave-one-out selection improves over the actor’s first sample by **+8.28 percentage points [+3.84, +12.99]** and over uniform selection by **+5.70 points [+2.60, +9.03]**. For each state, select using two continuations per action, score the held-out third, and average all three overlapping folds. Ties use their exact uniform expectation without consulting the held-out outcome. Candidate 0 is the first actor sample; uniform selection averages all five entries, including duplicates.

The hindsight oracle chooses and scores using the **same three outcomes**, making its 50.00% optimistic. It does not select a successful trajectory from all 15. LOO uses additional continuation outcomes and does not measure a trained selector’s performance.

### Where are the states, and how does selection perform by depth?

The groups below are disjoint. These are descriptive strata: tasks, replayability and remaining action budgets differ, so these rows do not identify a causal depth effect.

| Candidate decisions | States | Share | Actor first | Uniform | LOO selection | Hindsight oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 74 | 59.7% | 39.19% | 39.91% | 43.60% | 56.76% |
| 4–9 | 27 | 21.8% | 16.05% | 22.96% | 32.55% | 43.21% |
| 10–15 | 23 | 18.5% | 15.94% | 19.42% | 27.05% | 36.23% |

The early 74-state cohort is unchanged; **50 of the targeted 75 later states** were collected. Adding later states changes cohort composition, not the actor or teacher. Per-stratum intervals are in the aggregate.

### Exact decision counts and teacher-matched depth comparisons

| Candidate decision | Complete states | Teacher-matched states |
| --- | ---: | ---: |
| 1 | 33 | 31 |
| 2 | 23 | 22 |
| 3 | 18 | 17 |
| 4 | 7 | 7 |
| 5 | 6 | 6 |
| 6 | 6 | 5 |
| 7 | 4 | 4 |
| 8 | 2 | 2 |
| 9 | 2 | 2 |
| 10 | 4 | 4 |
| 11 | 6 | 6 |
| 12 | 3 | 3 |
| 13 | 1 | 1 |
| 14 | 5 | 5 |
| 15 | 4 | 4 |

| Candidate decisions | Matched states | Actor first | Before teacher | After teacher | LOO selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 70 | 40.48% | 44.29% | 44.76% | 44.98% |
| 4–9 | 26 | 16.67% | 24.36% | 25.64% | 31.24% |
| 10–15 | 23 | 15.94% | 19.32% | 28.02% | 27.05% |

Matched groups total 119 states and use different denominators from the all-state rows above.



### Repeat controls, oracle-action agreement and ties

| Choice disagreement | Matched states | Rate |
| --- | ---: | ---: |
| Before versus before repeat | 119 | 32.77% |
| After versus after repeat | 119 | 24.28% |
| Before versus after | 119 | 46.62% |

Cross-condition disagreement exceeds mean within-condition disagreement by +18.10 points [+12.53, +23.93]. More consistent choices do not by themselves establish better success. Presentation-order controls are separate from primary estimates and mix order sensitivity with teacher stochasticity; they do not identify a causal order effect.

Oracle-action agreement credits any candidate tied for the highest observed success across three continuations. This is a noisy same-data label; all-equal candidates make agreement automatic.

| Oracle subset | Matched states | Before agreement | After agreement | Actor-first agreement | Uniform agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any tied best action | 119 | 68.07% | 69.37% | 59.66% | 63.70% |
| Unique best action | 36 | 22.22% | 21.91% | 11.11% | 20.00% |

Across 124 states, **86 have tied hindsight maxima (69.35%)** and **70 distinguish candidate success counts (56.45%)**. LOO training maxima tie in **286 of 372 folds (76.88%)**. Always choosing the lowest-index tied action gives 36.83%, versus 38.12% for uniform ties; this is a sensitivity calculation, not an independently validated selection rule. Agreement uncertainty and independent uniform tie-breaking are retained in the aggregate.



### Method, uncertainty, judge limitations and next continuations

Actor: official OpenWebRL SFT, temperature 1.0, top-p 0.95, top-k off, 4,096 response tokens. The 30-action budget includes the replayed prefix. Isolated browsers must pass strict observable-state replay checks; hidden state need not match. The accepted cohort is not a random sample of tasks or decision points.

Teacher: unchanged index-only Luna-high, three before judgments and three after judgments per execution repetition, plus separate order controls. Judge: unchanged canonical o4-mini/AgentTrek terminal-success protocol. Saved positive examples retain partial-progress, temporal-grounding, absence-claim and filter-application concerns. Additional examples accept goal reinterpretation and conflate page/article or registered-user/active-editor metrics. Native labels remain unchanged; these rates do not establish strict independently verified task completion.

Every state contributes equally. Ten thousand bootstrap draws resample whole states, retaining candidates, repetitions and overlapping folds. Main tables use the LOO aggregation seed; repeat controls use their original seed. Intervals are exploratory and not adjusted for multiple comparisons. Saved outcome/teacher fingerprints and independent exact LOO calculations agree. Task-level artifacts remain private.

The cohort amendment froze these 124 states and all 1,860 original outcomes. The extra-two pass subsequently preserved these records and added 580 outcomes on 58 states; its full 1,240-new-outcome target remains incomplete. No candidates or teachers were regenerated. The original 149-state target remains unmet. The candidate pool was exhausted without relaxing quotas, replay checks or scientific settings. [Collection protocol and history](ARM_INTEGRATION_PLAN.md#arm-continuation-finish100-20261007).

</details>

<a id="branch-selector-scaleup-20261008"></a>

## New family: 32-state pilot and extra24 continuation extension

**The initial eight-draw pilot is verified complete; the extension is audited partial.** Both use the same 32 states, 29 task groups and five fixed candidates per state. The pilot saved 1,280 records: 1,008 valid and 272 invalid. Its valid set retains 224 native unjudged action-limit zeros. Existing records and candidates were preserved when adding draws 8–31.

| Coverage dimension | Category | Accepted states |
| --- | --- | ---: |
| Prespecified role | Train | 23 |
| Prespecified role | Seen-task validation | 4 |
| Prespecified role | Unseen-task validation | 5 |
| Candidate decision | 1–3 | 27 |
| Candidate decision | 4–9 | 2 |
| Candidate decision | 10–15 | 3 |

Replayable early decisions dominate. Both reported pilot analyses pool the prespecified roles as collection diagnostics; neither is a trained-selector validation score.

<a id="branch-extra24-20261008"></a>

### Do more selection continuations improve the action choice?

**Extra-24 collection ended partially; more selection draws have not yet established a reliable gain.** On the same 14 eligible states from 11 task groups, selecting with 24 draws scores **53.32%** on fresh draws 24–31, versus **49.80%** when selecting with eight. The paired difference is **+3.52 percentage points, 95% task-bootstrap interval [−0.12, +7.35]**. The interval includes zero. This is a descriptive outcome-selection diagnostic combining the pilot's prespecified roles, not a learned selector validation score or evidence that a post-action critic helps.

| Policy on the same fresh held-out panel | States | Task groups | Valid-only held-out success |
| --- | ---: | ---: | ---: |
| Actor's first candidate | 14 | 11 | 45.66% |
| Uniform over all five candidates | 14 | 11 | 47.95% |
| Select using draws 0–7 | 14 | 11 | 49.80% |
| Select using draws 0–23 | 14 | 11 | 53.32% |

Twenty-four-draw selection minus actor-first is +7.65 pp [−5.65, +16.07]; minus uniform it is +5.37 pp [−2.12, +11.30]. Both intervals include zero. The matched pre/post critic comparison on this new cohort, and training from its branch outcomes, remain unperformed. [Aggregate, split-specific results and evidence hashes](arm_results/rl_integration/branch-extra24-partial-20261009.json).

The run saved **2,200 of 3,840 requested extra records**, preserving all **1,280 original records**: 3,480 of 5,120 combined. All 96 scheduled blocks resolved: 55 released and 41 failed strict replay. Eighteen states have complete 160-record grids; four of those lack a usable valid-only five-candidate panel, leaving 14 eligible states. The other 14 admitted states lack a complete extra-draw grid. Missing and rejected measurements remain missing; the full study stays **incomplete**. No proposals were regenerated, no retry replaced a rejected draw, and no budget cap was exhausted.

<details>
<summary>Coverage, split isolation and uncertainty</summary>

For each candidate, fit its success fraction using native-valid observations from draws 0–7 or 0–23; evaluate both choices only on draws 24–31. Require every original candidate to have a positive valid fitting and held-out denominator, retain duplicate candidate indices, and use exact uniform weights over tied maxima. Average valid held-out rates within candidates, then states equally; these percentages are not pooled episode fractions. Fitting ties occur in five of 14 states with eight draws and six of 14 with 24 draws. Native invalids are excluded; valid failures, including unjudged action-limit zeros, remain failures.

| Panel on the common 14 states | Observed draws | Valid draws | Excluded invalid draws |
| --- | ---: | ---: | ---: |
| Fit, draws 0–7 | 560 | 483 | 77 |
| Fit, draws 0–23 | 1,680 | 1,452 | 228 |
| Holdout, draws 24–31 | 560 | 480 | 80 |

| Prespecified role | States | Task groups | Actor first | Uniform | Select with eight | Select with 24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 10 | 10 | 47.68% | 48.19% | 45.96% | 49.64% |
| Seen-task validation | 2 | 2 | 62.50% | 67.50% | 87.50% | 81.25% |
| Unseen-task validation | 2 | 1 | 18.75% | 27.20% | 31.25% | 43.75% |

Ten thousand paired whole-task bootstrap resamples, seed 20261008, keep each task's states together and normalize by sampled state count. Intervals are pointwise percentile intervals with linear interpolation, condition on the observed fitting panels and tie weights, and are unadjusted for multiple comparisons. Two seen-task clusters are insufficient for stable inference; the unseen split has only one task cluster, so no interval is reported. These tiny validation subsets cannot establish generalization. Replayability, unequal valid denominators, later website/judge drift and native judge errors further limit interpretation. A same-data maximum across 32 draws is optimistic and is not a true upper bound.

</details>

<details>
<summary>Partial endpoint audit and all-attempt accounting</summary>

All 3,480 committed small outcome records are bound by hashes to preserved parent or extra records; the 2,200 extra records also retain their audited ready/release identities. Native trajectory, judge and image checks reuse prior bounded reviews; this partial analysis does not claim a new exhaustive payload audit. The 392 finished reconstruction attempts without released outcomes remain charged and excluded from outcome denominators. Owned process/browser teardown passed. Slurm ended with exit 0; W&B intentionally records a failed run because the full collection target was unmet, and its remote final counts/costs reconcile. A stale final W&B browser sample is one; the authoritative final ledger and teardown show zero active browsers.

| Resource, including all attempts | Consumed | Approved cap |
| --- | ---: | ---: |
| Allocation seconds, four H200s / 32 CPUs / 480 GiB | 19,371 | 57,600 |
| Browser attempts | 2,592 | 5,000 |
| Local SFT call attempts | 16,607 | 240,000 |
| Canonical judge HTTP attempts | 1,287 | 8,000 |
| Judge charge, USD | 9.7330376 | 75 |
| Teacher calls | 0 | 0 |

The 38,229 unused allocation seconds remain separate from the pilot and historical extra-two budgets. All scheduled blocks have been processed under the approved no-retry rule; remaining budget does not justify relaxing replay checks or replacing outcomes. No additional collection is active.

</details>

<details>
<summary>Earlier eight-draw pilot: 6/2 selection, split coverage and verified accounting</summary>

Select using draws 0–5 and score valid observations from draws 6–7. Use the same unchanged five-candidate panel for every method: all candidates must have at least one valid fitting draw and one valid held-out draw. This leaves 25 states from 22 task groups and excludes seven states: five train and two unseen-task validation states. The common panel contains 18 train, four seen-task validation and three unseen-task validation states, with 721 valid fitting draws out of 750 and 241 valid held-out draws out of 250. Average valid held-out success within each candidate, then average states equally. Selection ties occur in 13 of these 25 states and receive exact uniform averaging without consulting held-out outcomes. These state-averaged rates are not pooled episode success fractions.

| Method on the same held-out panel | States | Task groups | Held-out success |
| --- | ---: | ---: | ---: |
| Actor's first candidate | 25 | 22 | 40.00% |
| Uniform over five entries | 25 | 22 | 41.20% |
| Select by six-draw valid success rate | 25 | 22 | 48.50% |

Selection minus actor-first is +8.50 pp [−2.72, +20.14]. Selection minus uniform is +7.30 pp, 95% task-bootstrap interval [+0.24, +15.17]. Intervals are exploratory, use 10,000 whole-task resamples and preserve paired states/candidates. This pools the pilot's prespecified roles for collection diagnostics; it is not a learned selector validation score. Native invalid records remain saved and charged but excluded from these rates. Valid-only performance is conditional on measurement availability, which can differ by action. The historical invalid-as-zero results above estimate a different quantity.

All 1,280 record identities, fixed candidate responses, strict replay releases, immediate post-action images, saved continuation and native judge bindings passed independent artifact checks. Prior completed payload audits were reused only after fingerprint verification. Native judge labels are preserved; integrity checks do not independently establish semantic correctness. The pilot used 27,263 of 43,200 approved allocation seconds, 3,360 of 5,000 browser attempts, 14,439 of 100,000 local SFT calls and 784 of 2,000 judge HTTP attempts, with a conservative judge-budget charge of $6.2050626 against $25. Discovery, failed reconstruction and every other attempt remain accounted for; no teacher calls or additional budget were used.

</details>

<a id="selection-head-20261009"></a>

## Selectors trained on teacher labels: do they pick better actions?

**No.** Piotr ARM and every actor-based selector were trained on GPT-5.5 choices, not on branching outcomes. Scored on held-out draws of every state that has eight draws for all five actions, none is clearly above a uniform pick. Selecting by observed outcomes on five draws is. Each fixed selector is scored on draws 0–7 of its chosen action. The outcome-selected reference picks on five draws and is scored on the other three, averaged over all 56 splits. Invalid outcomes count as zero, and intervals resample 120 task groups.

| Selector, 138 states | Held-out success | − Uniform, pp [95%] |
| --- | ---: | --- |
| Outcome-selected on five draws (privileged reference) | 36.99% | +5.32 [+3.09, +7.62] |
| Piotr SelectionARM | 34.15% | +2.48 [−0.14, +5.21] |
| Frozen-feature actor head | 33.70% | +2.03 [−0.26, +4.36] |
| Actor's first sample | 33.33% | +1.67 [−1.05, +4.30] |
| Kev-style pass on the actor, step 600 | 32.79% | +1.12 [−1.38, +3.75] |
| Uniform candidate | 31.67% | — |

The 138 states are 51 historical (topped up to eight draws), 32 pilot and 55 fresh states. Outcome selection gains about 5 pp in each collection. Teacher agreement does not predict these outcomes: the Kev-style pass matches held-out GPT-5.5 choices best of any actor-based selector (67.1%), yet its chosen actions succeed no more often than uniform. Outcome-labelled data now covers 188 eight-draw states in 151 task groups; that is 14× fewer states than the 2,692 teacher-labelled training states.

![Five selector designs compared by weights, inputs, candidate interaction, order effects, readout, added cost and status](arm_results/methods/selection_designs_compared.svg)

Method, metric definitions, teacher-agreement results, robustness checks and data sufficiency are in [ARM_SELECTION_HEAD.md](ARM_SELECTION_HEAD.md#selection-head-outcomes-20261010). The original 124-state, three-draw comparison is in its [robustness details](ARM_SELECTION_HEAD.md#selection-head-teacher-agreement).

<a id="teacher-labels-211-20261011"></a>

## Execution evidence and a stronger teacher on 211 states

**Neither immediate execution evidence nor a stronger teacher measurably improves action choice; every teacher sits about 3 pp above uniform, still below a pick from three observed continuations.** Each teacher's judgments are scored directly as picks on held-out continuation draws 3–7, which no teacher evidence touched. After-execution judgments saw only the immediate results of draws 0–2 (Luna) or draw 0 (Sol). The 211 states are the 160 new pilot, fresh and successor states, plus the 51 historical states with draws 3–7 for every action. Invalid draws count as zero, and intervals resample whole task groups.

**The 2×2: teacher strength × execution evidence.** These pairs share an identical layout: 3 repeats + 1 shuffled order per state, with after-evidence from draw 0. They are paired on the 194 states with complete draw-0 evidence.

| Paired comparison, 194 states | Difference, pp [95%] |
| --- | --- |
| Luna: after − before | +0.15 [−1.86, +2.10] |
| Sol: after − before | −0.31 [−2.40, +1.68] |
| After execution: Sol − Luna | +0.41 [−1.17, +1.89] |
| Outcome reference (best on draws 0–2) − Sol after | +1.83 [−0.98, +4.73] |

| Picker | States | Success, draws 3–7 | − Uniform, pp [95%] |
| --- | ---: | ---: | --- |
| Outcome reference: best on draws 0–2 | 211 | 39.16% | +5.68 [+3.67, +7.85] |
| Sol after (draw-0 evidence) | 194 | 39.20% | +3.45 [+1.17, +5.84] |
| Luna after (draw-0 evidence) | 194 | 38.79% | +3.04 [+0.82, +5.34] |
| Luna after (evidence draws 0–2, 10 calls) | 201 | 38.37% | +3.44 [+1.31, +5.72] |
| Sol before | 211 | 36.78% | +3.30 [+1.07, +5.63] |
| Piotr SelectionARM | 138 | 35.51% | +3.54 [+0.53, +6.59] |
| Luna before | 211 | 35.92% | +2.45 [+0.34, +4.64] |
| Actor's first sample | 211 | 35.55% | — |
| Uniform candidate | 211 | 33.48% | — |

On all 211 states, Sol before − Luna before is +0.85 [−0.79, +2.51] pp, and Luna's full 10-call after − before is +0.73 [−1.08, +2.51] pp. The historical states alone gave +2.37 for Luna after − before (49 states), consistent with the earlier +2.24; the new states give +0.20.

<details>
<summary>Why immediate post-action evidence does not help here: diagnostics on 194 states</summary>

These diagnostics use saved data only (`scripts/analyze_post_action_value.py`). The 194 states have draw-0 evidence and held-out draws 3–7 (146 new, 48 historical). After-picks are each teacher's draw-0 judgments.

| Check | Luna | Sol |
| --- | ---: | ---: |
| States whose majority action changes after execution | 69 (36%) | 49 (25%) |
| Changed picks: better / worse / tied on draws 3–7 | 21 / 15 / 33 | 16 / 10 / 23 |
| After − before on changed states, pp [95%] | +0.58 [−3.88, +4.93] | +0.10 [−7.10, +6.98] |
| Share of picks on visibly failed candidates (error or unchanged screen), before → after | 18.0% → 15.1% | 15.6% → 15.3% |

- **The evidence often cannot discriminate.**
  - In 39 states all five candidates are the same action.
  - In 18 states every distinct action produces the same screen and URL.
  - In 60 states some do.
  - Only 77 states (40%) show a distinct result for every distinct action.
- **Visible failures are rare and not costly.**
  - Only 4 of 970 candidate executions report a tool error.
  - 134 leave the screen unchanged, and those candidates succeed only 1.0 pp less than their state's mean.
  - Teachers already pick them less often than uniform (20%) before execution.
- **The kind of immediate result barely predicts the outcome.** Within-state relative success by immediate-result type, draw 0:

| Immediate result | Candidates | Relative success, pp |
| --- | ---: | ---: |
| navigated to a new URL | 482 | +0.7 |
| same URL, screen changed | 329 | −0.4 |
| terminated (done) | 81 | −3.3 |
| screen unchanged | 74 | +0.6 |
| errored | 4 | −2.0 |

- **The evidence replays consistently.** 98% of later draws reach the same URL as draw 0, and 76% the same screenshot.
- **No stratum shows a gain.** After − before stays within ±1.3 pp when states are split by whether the evidence distinguishes the actions, or by how much the actions' outcomes differ on draws 0–2. The draw-0–2 split is independent of the scored draws.
- **Changed picks (qualitative).**
  - The largest losses pick a page that looks closer to the goal: a direct article URL instead of a search, or an official site instead of a news hub. From there the fixed SFT continuation then fails more often.
  - Several other changes switch between near-identical actions, whose outcome differences are draw noise.
  - The teachers judge apparent progress. The outcome depends on whether this particular continuation policy can finish from the new page, and one post-action screen does not reveal that.

</details>

<a id="lookahead-setups"></a>

### Lookahead labelling: what Luna sees in each condition

Every condition is the same GPT-6 Luna request (`gpt-6-luna`, high reasoning effort), built by `openwebrl.arm_branch_teacher.request` with condition-specific additions (`scripts/run_branch_lookahead_labels.py`).
- **Calls per state:** four: three in the canonical candidate order and one in a seeded shuffled order. Luna returns `{"selection": k}`.
- **Scoring:** each pick is scored on held-out draws 3–7 of the chosen action. Draws 1–7 are never shown, nor is any judge verdict or reward.

| Input | Before execution | Immediate result | Lookahead K (K = 2, 3, 5) | K = 1 + 3 history screenshots |
| --- | :---: | :---: | :---: | :---: |
| Task and current screenshot | ✓ | ✓ | ✓ | ✓ |
| History: last 5 past actor outputs as text, **reasoning included** | ✓ | ✓ | ✓ | ✓ |
| Screenshots before the last 3 past actions | — | — | — | ✓ |
| Five candidates' full text, **reasoning included** | ✓ | ✓ | ✓ | ✓ |
| Immediate result of each candidate (draw 0): post-action screenshot, page observation, tool feedback | — | ✓ | ✓ | ✓ |
| Next K actor steps after each candidate (draw 0): action text, **reasoning stripped**, ≤1,500 characters, plus the resulting screenshot | — | — | ✓ (K = 2, 3 or 5) | ✓ (K = 1) |
| Final answer if the episode ended within K steps | — | — | hidden: shown as "[the episode ended here; any final answer is not shown]" | hidden |
| Images per call (maximum) | 1 | 6 | 16 / 21 / 31 | 14 |

- **What a lookahead step contains.** Each is an (action, resulting screenshot) pair. The screenshot after an episode's final step is not shown. An earlier arm that showed final answers did worse (−1.49 pp) and is not used.
- **Reasoning.** It is included for the candidates and the history in every condition, and stripped from lookahead steps in every condition. Lookahead with the actor's reasoning kept is an untested variant.

| State set | States | Task groups | Conditions run | Status |
| --- | ---: | ---: | --- | --- |
| [First lookahead study](#lookahead-labels-20261011): teacher-campaign states with aligned draw-0 rollouts (62 fresh64, 57 fresh64succ, 48 historical, 27 pilot32) | 194 | — | before, immediate (from the 211-state study), K = 3 hidden, K = 3 shown | done |
| [Stage (a)](#lookahead-stage-a-20261012): fresh states on Piotr's training-split tasks (`piotr-matched-parallel10-v1`) | 121 | 53 | before, immediate, K = 3 hidden | done |
| Depth and history ablation: the 315 states above | 315 | — | K = 2, K = 5, K = 1 + 3 history screenshots | running (jobs 359595, 359596, 359602; cap USD 40 / 5,000 calls) |
| Proposed scale-up: `fast1k-20261012` | 713 | — | all six conditions | awaiting the ablation results and approval |

- **Primary comparison for the ablation:** each new condition minus K = 3 hidden, on the same states, with task-group bootstrap intervals.
- **Power.** 315 states can detect a 3–4 pp difference between conditions, not a 1 pp difference.

<a id="lookahead-labels-20261011"></a>

### Does multi-step lookahead help where immediate evidence did not?

**Yes, modestly: showing the fixed actor's next three steps improves Luna's picks over immediate evidence, and only when its final answers are hidden.** For each candidate, Luna additionally sees the SFT actor's next three actions in draw 0, each with the screenshot it produced. In arm A, an episode that ended within those steps is marked only as ended, without the final answer. In arm B, the final answer is shown; in 192 of 970 candidate continuations the actor submits one within three steps. No arm sees a judge verdict or reward, and scoring uses draws 3–7 only. All 194 states have complete draw-0 evidence, and every comparison is paired on them.

| Luna, 194 states | Success, draws 3–7 | − Uniform, pp [95%] |
| --- | ---: | --- |
| Lookahead, final answers hidden | 40.77% | +5.02 [+2.73, +7.40] |
| Lookahead, final answers shown | 39.28% | +3.53 [+1.34, +5.83] |
| Immediate result only (draw 0) | 38.79% | +3.04 [+0.82, +5.34] |
| Before execution (same states) | 38.64% | — |

| Paired comparison, 194 states | Difference, pp [95%] |
| --- | --- |
| Lookahead (answers hidden) − immediate result | +1.98 [+0.08, +3.98] |
| Lookahead (answers hidden) − before execution | +2.14 [−0.53, +4.92] |
| Answers shown − answers hidden | −1.49 [−2.64, −0.42] |

- **Lookahead reaches the outcome reference.** On these 194 states, picking the best action on three observed outcomes scores 41.03%. The answers-hidden lookahead scores 40.77%, from one draw of intermediate steps.
- **Seeing the answer hurts.** One plausible reading is that Luna rewards confident or plausible answers that the success judge does not accept. This was not tested.
- **These are exploratory, pointwise intervals** among several comparisons. The lookahead-versus-immediate gain is the only one with an interval above zero, and it was not prespecified as the primary test.
- **Accounting.** Job `355930` made 1,552 GPT-6 Luna calls at high effort, all valid, for USD 10.48 against an approved USD 35 / 1,700-call cap (`scripts/run_branch_lookahead_labels.py`).

<a id="lookahead-stage-a-20261012"></a>

#### Replication on fresh Piotr-matched states (stage a)

**The lookahead gain replicates on independent states: lookahead minus immediate result is +4.75 pp [+1.58, +8.27], the prespecified primary test.**
- **States:** 121 fresh states from the Piotr-matched collection, in 53 task groups. All have draw-0 evidence aligned for all five candidates, plus held-out draws 3–7. None were used in the 194-state study.
- **Design:** three matched Luna conditions on each state, 4 calls each (3 repeats + 1 shuffled order):
  - before execution;
  - immediate result from draw 0;
  - three-step lookahead from draw 0, final answers hidden.
- **Scoring:** draws 3–7 only. Intervals resample task groups (10,000 draws).

| Luna, 121 states | Success, draws 3–7 | − Uniform, pp [95%] |
| --- | ---: | --- |
| Lookahead, final answers hidden | 38.31% | +4.19 [+1.26, +7.22] |
| Before execution | 35.00% | +0.88 [−1.20, +2.87] |
| Immediate result only (draw 0) | 33.55% | −0.56 [−3.15, +1.84] |
| Uniform | 34.12% | — |
| Outcome reference: best on draws 0–2 | 39.36% | +5.24 [+1.71, +8.79] |

| Paired comparison, 121 states | Difference, pp [95%] |
| --- | --- |
| **Lookahead − immediate result (primary)** | **+4.75 [+1.58, +8.27]** |
| Lookahead − before execution | +3.31 [+0.32, +6.43] |
| Immediate result − before execution | −1.45 [−4.05, +1.17] |
| Outcome reference − lookahead | +1.05 [−1.86, +3.73] |

- **Immediate evidence still does not help.** As on the 194 states, the post-action screen alone leaves Luna at uniform.
- **Lookahead closes most of the gap to outcome selection** from a single draw of intermediate steps. It is within 1.05 pp of choosing on three observed outcomes.
- **Accounting.** Job `356327` made 1,452 GPT-6 Luna calls at high effort, all settled, for USD 6.46 against the approved USD 30 / 5,000-call cap. The script is `scripts/run_branch_lookahead_labels.py` with `LOOKAHEAD_CONDITIONS=before,after,lookahead`. The analysis is `selection_outcome_experiments.py stage_a`; evidence: [stage-a JSON](arm_results/rl_integration/lookahead-stage-a-20261012.json).

**Can these labels supervise a pre-action student?** Not at this scale. The [actor-feature head](ARM_SELECTION_HEAD.md#selection-head-label-types-20261011) trained from scratch on each label type stays at uniform:

| Training labels, original 124 states | Labelled states | 5-fold task-grouped CV, − uniform, pp [95%] | Trained on 124, tested on 87 new states, − uniform, pp [95%] |
| --- | ---: | --- | --- |
| Luna before | 124 | −1.71 [−4.82, +1.31] | −0.46 [−3.44, +2.67] |
| Luna after (evidence draws 0–2) | 121 | −0.52 [−3.60, +2.49] | +0.26 [−2.72, +3.15] |
| Outcomes on draws 0–2 (same executions) | 59 | −1.05 [−4.02, +2.00] | −1.03 [−4.27, +1.94] |
| GPT-5.5 labels, same number of other states | 124 | −1.13 [−4.07, +1.77] | −0.46 [−3.30, +2.33] |
| Sol before | 124 | +0.17 [−2.96, +3.27] | +0.26 [−2.76, +3.31] |

Outcome labels exist only where draws 0–2 separate the actions (59 states).

<details>
<summary>Labelling runs: layout, evidence protection and accounting</summary>

| Run | Model, effort | States | Calls | Charged, USD | Approved caps |
| --- | --- | ---: | ---: | ---: | --- |
| Luna, job `355533` | GPT-6 Luna, high | 160 | 2,115 | 8.22 | 2,400 calls, USD 15 |
| Sol, job `355536` | GPT-6.1 Sol, medium | 284 | 1,136 | 24.76 | 1,250 calls, USD 50 |
| Sol after, job `355686` | GPT-6.1 Sol, medium; draw-0 evidence | 194 of 211 (17 lack complete draw-0 evidence) | 776 | 70.87 | 850 calls, USD 100 |

- **Luna layout.** It uses the historical [4 before + 10 after layout](#arm-continuation-branches-results-20261007), with `arm_branch_teacher.py` unchanged. The teacher-only campaign's per-state release lists only draws 0–2, so draws 3–7 are never opened.
- **Luna coverage.** 144 states have all 14 judgments. 16 states lack evidence for some candidate in one or more of draws 0–2, because the action failed to execute; those after-calls are recorded invalid without a request. Every API call returned a valid choice.
- **Sol run.** It sends the same request as Luna before, with only the model and effort changed: 3 repeats and 1 shuffled order per state, with no execution evidence. It covers the 160 new states and all 124 historical states. Its own ledger reserved each call's worst case before dispatch and settled actual usage; all 1,136 calls were valid.
- **Compute.** Both runs were CPU-only Slurm jobs, without GPUs.

</details>

<a id="branch-post-to-pre-study-20261008"></a>

## What remains to establish post-action value and train a pre-action critic

The old Luna comparison has not established a post-action advantage. The new family supplies saved immediate next screenshots and continuation targets, but its zero-teacher-call budget included no new critic judgments. Training the actor-feature heads on existing labels does not answer whether next-screenshot labels improve a pre-action student.

| Question | Matched comparison | Evidence required |
| --- | --- | --- |
| Does immediate execution evidence help? | Same frozen critic and repeated index-only selection protocol, with versus without each candidate’s immediate next screenshot | Paired selected-action success on held-out continuations; task-bootstrap interval and valid-panel coverage |
| Can it train a better pre-action critic? | Identical pre-action students, training states and optimization budgets; targets from pre-action versus post-action teacher choices | Improvement on distinct seen-task states and unseen tasks, using only pre-action inputs at inference |
| How informative are the continuation outcomes themselves? | Separately labeled continuation-return-supervised selector/reference | Held-out outcomes that never enter its selection or training targets |

The needed sequence is a matched pre/post labeling comparison, then matched student fits, with an independent continuation holdout and task splits preserved. More continuations can improve a return estimate; they do not substitute for testing the post-action critic itself. New critic calls, branch-supervised training and larger collection require their own exact resource approvals.

<details>
<summary>Planned controls, task splits and larger collection proposal — not an active run</summary>

The post-action input must be the screenshot immediately after the fixed candidate action, bound to its execution receipt. A terminal screenshot or later suffix cannot substitute. Prespecify the evidence draw independently of outcomes; do not choose the successful or clearest-looking transition. Teachers receive no continuation returns or future screenshots. Repeated teacher choices can supply matched target distributions for distillation; both students use the same loss and see only the original state and candidates. Keep teacher labels and continuation-derived targets distinct. Missing evidence excludes a paired comparison, with coverage reported, while other valid targets remain usable.

| Dataset role | Total target states | Retained existing states | New state target | New continuation target |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,000 | 124 | 876 | 35,040 |
| Seen-task validation | 250 | 0 | 250 | 10,000 |
| Unseen-task validation | 250 | 0 | 250 | 10,000 |

New states use five candidates × eight draws. Existing 124 states remain training eligible with their actual observations: 2,440 audited records, comprising five draws per action on 58 states and three on 66 states. Thus the full aspirational design requires 55,040 new outcomes and 57,480 combined outcomes; it does not require topping up historical states to eight draws. This larger collection and branch-supervised fitting remain unapproved; the separate teacher-trained head experiment above is already complete.

Task and state assignments are fixed before new outcomes. All candidates, draws and retries from one state stay together. Seen-task validation requires independently seeded discovery trajectories, distinct pre-action task/screenshot/history inputs, and an accepted training state from the same task. A different candidate panel alone does not create a new state. Previously collected task groups cannot enter unseen-task validation or test, but genuinely new states from those tasks may enter seen-task validation. Additional task groups remain locked for final testing. “Unseen” means unseen by this selector fit, not proven absent from backbone pretraining.

Fit a fresh adapter on released SelectionARM to each candidate’s success fraction using its **valid draw count**, retaining ties and all-equal panels. The proposed loss is independent sigmoid/binomial BCE over five candidate-index logits; serving still returns one index. Inputs contain only pre-action information and the candidate panel. **User amendment, October 8: skip invalid continuation measurements in targets and future success-rate denominators.** Keep valid unsuccessful outcomes, including native unjudged action-limit zeros. A candidate with no valid draws has no target and is masked from the loss. Preserve raw invalid records, reasons and costs without rerunning them. Missing records are unresolved, not invented failures. This is outcome adaptation of a teacher-pretrained model.

Evaluate selected-action continuation success separately on both validation sets against actor-first, uniform and the frozen selector. A rollout-based reference selects on the valid observations within six new draws and scores valid observations from the other two. Paired five-candidate reference comparisons require at least one valid selection draw and one valid held-out draw per candidate, on the same states for every method; report excluded states and per-candidate coverage. Do not renormalize the candidate set using held-out validity. Valid-only rates condition on measurement availability and can have unequal draw counts; the historical operational results above retain their original invalid-as-zero definition. Bootstrap whole task groups. Keep final test tasks unopened during checkpoint selection and label selected-checkpoint validation scores accordingly. Judge limitations and replayability selection remain; this does not establish end-to-end benchmark improvement or superiority to matched teacher-label training.

### Frozen opportunity pool and original pilot allocation

The full draft groups 2,090 task identities and defines three decision slots per task across decisions 1–3, 4–9 and 10–15. It has 2,978 training, 1,489 seen-task validation, 1,203 unseen-task validation and 600 locked-test opportunities. The pilot freezes 1,200 opportunities across 400 task groups: 600 train, 300 seen-task validation, 300 unseen-task validation, and 400 in each depth band. These are opportunities, not collected states. Eight CPU tests and 12 provenance/isolation/arithmetic checks passed. Assignment never consults new successes, ties or teacher preferences.

Exact task/normalized-goal comparison found no overlap with protected OM2W, WebVoyager, DeepShop or WebGym test cohorts. This does not establish semantic decontamination. Earlier selector dev/future/retention and joint-validation splits overlap 366 task groups, including 25 old states. Those historical splits are study-specific and remain untouched: retaining these examples is allowed for this new fit, but reused historical cohorts cannot support independent evaluation claims for it.

The recent later-state collection produced 38 accepted states from 1,173 discoveries in 39,868 seconds on four H200s, consuming 2,695 browser starts with three draws per action. Eight draws require all 40 isolated replay contexts to pass the unchanged readiness gate before candidate dispatch. The pilot reached and audited all 32 accepted states after 647 considered opportunities and 294 resolved replay groups. The existing pool cannot guarantee the full state target; source expansion or more prespecified slots will be budgeted from measured pilot yield.

**Original eight-draw pilot allocation:** four H200s, up to 40 CPUs, 480 GiB and 43,200 seconds total including startup, tests and retries. The scheduler requires at most eight CPUs per GPU, so the deployed request uses 32 CPUs. Caps: 5,000 browser-start attempts / 40 concurrent browsers, 100,000 local SFT call attempts, 2,000 canonical judge HTTP attempts / $25, no teacher calls. Stop at 32 newly accepted states / 1,280 new continuations, 1,200 opportunities, or the first binding cap. Diagnose zero acceptance among the first 24 resolved replay groups. Preserve every attempt and reserve time for cleanup. Scheduler admission, GPU startup and scientific completion are separate milestones.

The later branch-supervised fitting estimate is one H200, eight CPUs, 120 GiB and two hours total, pending exact approval and validated data/loss code. [Dated proposal and pilot aggregate](arm_results/rl_integration/branch-selector-scaleup-proposal-20261008.json). That file retains historical planning fields: its extra24 running status is superseded by the [terminal partial aggregate](arm_results/rl_integration/branch-extra24-partial-20261009.json), and its 58,140-outcome target assumes all 3,100 historical outcomes. Using the actual 2,440 preserved historical outcomes gives the 57,480 combined-outcome proposal above. Neither number is a collected dataset size.

</details>

<details>
<summary>Earlier snapshots and historical plots</summary>

These overlapping cohorts are history, not independent replications. Rows use their own teacher-matched cohorts.

| Completed states | Matched states | Actor first | Before teacher | After teacher | LOO selection | Hindsight oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 86 | 82 | 38.21% | 42.01% | 42.68% | 43.39% | 55.69% |
| 88 | 84 | 37.70% | 41.40% | 42.06% | 42.69% | 55.16% |
| 97 | 93 | 36.56% | 41.58% | 42.05% | 42.47% | 55.20% |
| 120 | 115 | 31.01% | 35.56% | 37.78% | 38.21% | 50.72% |

The [97-state](arm_results/rl_integration/continuation-fixed97-20261008.json) and [120-state](arm_results/rl_integration/continuation-fixed120-20261008.json) aggregates remain unchanged. Historical values and hashes are also indexed in the [124-state aggregate](arm_results/rl_integration/continuation-fixed124-20261008.json). The plots below describe the older 86-completed/82-matched snapshot; a zero-width interval in a tiny all-failure group is not population certainty.

![Historical continuation success by decision group](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/paired-success.png)
![Historical teacher choice disagreement](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/repeat-controls.png)

[Historical depth and coverage aggregate](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/aggregate.json) · [Early-state diversity](arm_results/rl_integration/continuation-state-diversity-20261007/aggregate.json).

</details>

## Related diagnostics with different endpoints

<details>
<summary>Saved-action label changes and full-episode random selection</summary>

### Does execution change a saved-action progress judgment?

This separate diagnostic presents the same saved Piotr action to Luna-high before and after execution. The teacher returns progress probabilities, evidence and a rationale, rather than selecting among five actions.

| Comparison | Examples | Changed labels | Change rate |
| --- | ---: | ---: | ---: |
| Before → after, full set | 200 | 32 | 16% |
| Before → after, repeat subset | 100 | 18 | 18% |
| Before → before repeat | 100 | 13 | 13% |
| After → after repeat | 100 | 7 | 7% |

On the repeat subset, change exceeds before-repeat noise by 5 points [−3, +14]. Changed labels do not establish better accuracy, and only the executed action has evidence. [Diagnostic details](ARM_RESULTS.md#arm-teacher-evidence-primary200-20261006).

### Does random selection explain full-episode gains?

The root-state uniform control above randomizes one action and then continues with SFT. A separate full-episode control samples five candidates and picks uniformly at **every step**, under the same actor, decoding, browser, step cap and judge as its paired controls.

| Full-episode policy | Successes | Tasks | Task success |
| --- | ---: | ---: | ---: |
| SFT alone | 99 | 300 | 33.00% |
| Random of five | 100 | 300 | 33.33% |
| Piotr SelectionARM | 120 | 300 | 40.00% |

Random minus SFT is +0.33 points [−4.67, +5.67]; Piotr minus random is +6.67 points [+1.00, +12.33]. This supports useful learned selection in that run; random/SFT equivalence is not established. [Full-episode comparison](ARM_INFERENCE_SCALING.md#random5-sameday-results-20261007).

</details>
