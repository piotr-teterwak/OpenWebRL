# WebPRM baselines

<a id="repair-status-20261010"></a>
## Repair status — October 10, 2026

**The v4 repair run is closed and incomplete: 16 of 72 fixed cells were attempted, nine had valid terminal judgments, and none succeeded.** A proven browser-geometry mismatch hid visible Arbiter controls and stopped further task admission. The four active episodes finished, but the paused controller was not closed promptly; this supervision failure consumed substantial idle allocation time. No evaluation allocation remains live. A v5 geometry correction passed a real-browser boundary-control fixture and 85 focused tests. An optional real-site check was inconclusive because browser observation extraction was unstable; no corrected live benchmark has run, and no new allocation or API budget has been approved or submitted. [Aggregate repair evidence](webprm_results/debug_20261010.json).

| Repair cohort | Planned cells | Attempted | Valid | Invalid | Missing | Successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3, closed after context failure | 72 | 6 | 5 | 1 | 66 | 0 |
| v4, closed after geometry diagnosis | 72 | 16 | 9 | 7 | 56 | 0 |

These separate partial cohorts do not establish critic effectiveness. Validity requires a completed episode and a complete, parseable terminal judgment; it does not establish correct browser geometry or error-free execution. Invalid attempts remain in the fixed 12-task denominator for each condition, while missing cells remain unobserved. All saved outcomes and prior cohorts are preserved.

<details>
<summary>Repairs and validation</summary>

| Problem | Evidence | Repair and validation |
| --- | --- | --- |
| One malformed Shepherd sample terminated the whole batch/episode | All 50 saved multi-sample batches had usable candidates; the old rule discarded 316 valid samples | Parse every fixed draw with the native parser, retain valid candidates, and record every rejection; an empty pool consumes a turn and returns error feedback without executing an action or resampling |
| Arbiter omitted the released state-difference and terminal-detection stages | Its sampler does not advertise terminal actions; all 25 valid Arbiter episodes reached the turn limit | Restore native auxiliary prompts/parsers in all three conditions, with the same pinned actor, T1/top-p0.9/2,048-token caps; preserve terminal refinement state |
| Arbiter failed actions were absent from subsequent prompt history | 112 of 153 execution errors repeated a previously failed action | Include exact execution errors in history and alongside the generated state difference |
| Synthetic browser activation raced navigation | At least 16 click-related context-destroyed errors came from the adapter callback | Restrict activation bridging to tab changes; 24 real-browser navigation, typing, save and tab actions passed |
| Startup and request preflight were brittle | Long shared temporary paths failed; a standalone worker reserved budget before discovering missing credentials | Force short local temporary/cache paths and validate/load credentials before browser startup or reservation |
| Initial navigation waited for every page subresource | A controlled interactive page timed out under the old 10-second load check | Wait for DOM readiness with a 30-second cap bounded by the episode deadline; retain HTTP/navigation failures explicitly |
| The released Shepherd serving example caps context at 10,000 tokens | The first repaired live guided episode failed on a 12,574-token prompt after ten executed turns | A distinct v4 retry uses the checkpoint's native 40,960-token limit, with no input truncation or RoPE scaling; GPU startup replay passed on the exact 12,574-token request with native score parsing |
| Native Arbiter filtering used an 800×600 screen inside a 1,280×720 viewport | After BrowserGym screenshot extraction, the native screen-based filter omitted visible right/bottom controls | Restore screen and viewport dimensions before every observation; three repeated screenshot/action cycles and right/bottom controls in a new tab passed a real-browser fixture, without changing the native observation processor |

This is a **new, versioned evaluation protocol**, not a correction to old outcome labels. Individual Shepherd parsing agrees with the native single-action contract on all 1,136 saved samples; the fix changes candidate handling and recovery, not extraction heuristics. Original source code, rollouts, verdicts and shared API reservations remain preserved.

**All 85 focused tests passed after the v5 geometry and shutdown changes**; the 14-file source freeze and batch syntax were also verified. All six conditions had passed a real-browser integration fixture using scripted completions. That earlier fixture did not cover the screen/viewport mismatch; a separate v5 geometry fixture now passes. An optional real-site check was inconclusive because browser observation extraction was unstable; it provides no evidence that the live failure is repaired. The six-condition fixture exercised actual Chromium observations, native action execution, candidate ranking/tournaments, malformed-action recovery, error feedback, state differences, terminal detection and saved terminal artifacts. Both fixtures make zero provider or model-server calls; their scripted outcomes establish pipeline mechanics, not benchmark success.

During the initial v4 stage, both actual critics passed GPU preflight and completed a 30-turn live guided episode with a valid failure judgment. Shepherd retained 597 of 600 fixed actor draws and made 126 native scoring requests; its largest critic prompt contained 22,805 tokens. The exact 12,574-token context regression also passed without truncation at the native 40,960-token limit. These checks establish that collection and scoring executed on the supplied observations. They do not resolve the subsequently identified observation defect or establish critic effectiveness or deterministic serving.

The original judge was also audited: all 31 valid episodes supplied the correct task, pinned prompt, matching trajectory and byte-identical terminal screenshot; all complete model responses explicitly returned failure. The judge wiring did not cause universal zero success. Missing actor completion logic can harm behavior, but does not itself force this judge to return failure.

The two originally unavailable sites still time out under a 30-second DOM-readiness check. Their failures cannot be claimed repaired or relabeled as actor failures, and their tasks have not been silently replaced. The frozen browser version and live-site availability remain limitations.

</details>

<details>
<summary>Failed gate, geometry diagnosis, supervision failure and accounting</summary>

The original v4 rule required at least one valid episode per condition across the initial two tasks. It **failed**: Arbiter unguided's first-task policy response ended with `finish_reason=length` at the unchanged 2,048-token cap, and its second task failed at initial navigation with HTTP 403. All six conditions encountered that second-task website failure. The initial stage therefore had 12 attempts, five valid episodes and no successes; both guided conditions had a valid first-task episode.

An explicit operational waiver admitted missing cells after reviewing those failures; the original gate did not pass. The continuation preserved all initial results and the frozen model, worker, prompts, sampling and per-task arm shuffle, with four workers. Its driver hash, failure proof and revised deadline were recorded. Four additional episodes completed before the run closed, giving the final 16 attempts; the remaining 56 cells were never attempted. No failed cell was rerun or relabeled.

The subsequent geometry diagnostic proved that the released native screen-based observation filter used 800×600 bounds after BrowserGym screenshot extraction while the rendered viewport remained 1,280×720. This omitted visible controls on the right and bottom. Replaying the exact released metrics override independently reproduced the same screen reset in this Chromium build. The correction is therefore a disclosed browser-compatibility repair of an inherited assumption, not a change to the native observation processor. Before each observation, it retains the page's CDP session, clears the stale device-metrics override and explicitly restores screen/viewport dimensions. A real-browser fixture verified three repeated screenshot/observation/click cycles, an original-tab bottom control and new-tab right/bottom controls, with zero provider calls. The optional real-site check was inconclusive because browser observation extraction was unstable; no corrected live benchmark has run. V4 remains a preserved diagnostic cohort; v5 would start a fresh cohort with the same 12 tasks, six conditions, actor sampling and turn limit, with the geometry change explicitly versioned.

The admission controller was paused while its four active workers drained. Active supervision then failed to terminate the paused controller and release the allocation. The final allocation remained idle for **3,746.32 seconds** after the last durable result. The launcher eventually killed its remaining child processes, and the final allocation ended `FAILED`. All workers are now stopped. The saved aggregate was reconstructed from the 16 durable cells, and the earlier aggregate was archived. The implemented and tested cooperative stop-admission guard watches `stop-admission.json`, stops admitting new cells, drains active workers into durable results and returns without pausing the process. The launcher can then exit normally. The 85-test suite also validates retry accounting for the proposed three-hour budget while retaining every attempt.

All allocation time, including the idle interval, is charged to the approved **one H200, eight CPUs, 120 GiB and 7,200-second repair budget**. The context-failure attempt, initial v4 stage and v4 continuation consumed 1,073, 1,371 and 4,576 seconds respectively: **7,020 seconds used, 180 remaining**. That remainder does not support another full evaluation. The prepared v5 proposal requests **one H200, eight CPUs and 120 GiB for 10,800 additional allocation seconds (three hours), including all retries**, under the same USD 30 shared API cap with no additional API budget. It starts a fresh 72-cell cohort using the same 12 tasks and six conditions; the old 180-second remainder is not added. The proposed initial validity gate covers the first three fixed tasks, expanded from two after the earlier second-task website failures; all fixed tasks and failure outcomes remain included. This proposal is **not approved or submitted** and does not guarantee a successful outcome or complete coverage.

The shared API ledger contains **1,905 request reservations: 1,904 settled and one unresolved**. Total settled charges are **USD 1.91658900**, with **USD 0.00609330** retained conservatively, totaling **USD 1.92268230** against the unchanged **USD 30** cap. V4 alone contributed 523 settled calls and USD 0.60449560. Original v2, its two supplemental attempts, v3 and v4 outcomes remain separate; their request journals reconcile the shared ledger. The independent closure audit verified 108 preserved files, all 14 frozen v4 files and every archived ledger prefix, with no orphan reservations.

</details>

<a id="pilot-status-20261010"></a>
## Pilot results — October 10, 2026

**This feasibility pilot did not produce a usable critic comparison.** Across 12 tasks and six conditions, all 72 first attempts were recorded, but only 31 had valid terminal judgments; none of those 31 succeeded. The pilot primarily exposes actor-output and browser reliability limits. Shepherd guided produced no valid episode, so this pilot cannot establish its critic effectiveness or rank the two reward models. [Aggregate results and accounting](webprm_results/pilot_20261010.json).

“Attempted” includes browser startup and initial navigation failures. “Valid” requires a completed episode, saved rollout/screenshots and a complete, parseable terminal-judge response; it does not require error-free actions or a separate page-availability screen. Overall success is recorded successes divided by the 12 fixed tasks per condition, including invalid attempts in the denominator. Valid-only success uses the valid count; it is undefined when that count is zero.

| Condition | Attempted | Valid | Invalid | Successes | Overall success (%) | Valid-only success (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shepherd unguided | 12 | 5 | 7 | 0 | 0.0 | 0.0 |
| Shepherd random candidates | 12 | 1 | 11 | 0 | 0.0 | 0.0 |
| Shepherd 8B guided | 12 | 0 | 12 | 0 | 0.0 | — |
| Arbiter unguided | 12 | 9 | 3 | 0 | 0.0 | 0.0 |
| Arbiter random candidates | 12 | 8 | 4 | 0 | 0.0 | 0.0 |
| Arbiter 7B guided | 12 | 8 | 4 | 0 | 0.0 | 0.0 |

These are complete **attempt counts**, not 72 valid evaluations. All guided-minus-control success differences are zero percentage points; relative gains are undefined because control success is zero. The all-zero paired percentile bootstrap degenerates to [0, 0] and does **not** establish equivalence. Arbiter guided has eight common-valid tasks with unguided and seven with random selection, all unsuccessful; Shepherd guided has no common-valid comparison.

<details>
<summary>Why valid coverage is low</summary>

| First-attempt failure category | Episodes |
| --- | ---: |
| Shepherd native parser found zero or multiple actions | 24 |
| Initial website-load timeout, two sites across all six conditions | 12 |
| Collection deadline | 2 |
| Arbiter local critic timeout | 1 |
| Browser closed during an episode | 1 |
| Browser startup failure in the final CPU control attempt | 1 |
| Total invalid | 41 |

Shepherd guided encountered 33 malformed choices among 220 samples. Its frozen rule aborts a whole 20-sample response if any choice fails native parsing: ten of eleven responses aborted, discarding 167 otherwise one-action choices before selection. The guided condition's twelve failures comprise ten parser failures and two initial website-load failures. This diagnoses a brittle actor-output policy; it does not establish a critic-scoring defect. Changing that policy would define a new intervention and must preserve this cohort.

Fifteen of the 31 valid episodes contain action-execution errors. All 121 Arbiter coordinate-bound errors reference IDs absent from the contemporaneous saved accessibility tree, which supports stale or ungrounded actor targets rather than proving an observation-to-action mapping defect. These diagnostic frequencies come from different trajectories and are not matched estimates of critic effects.

</details>

<details>
<summary>Execution, recovery and budget accounting</summary>

The approved cap is one H200, eight CPUs and 120 GiB for 7,200 total allocation seconds, including retries, plus USD 30 shared actor/judge spending. Four terminal Slurm attempts consumed 20, 6,076, 4 and 849 seconds: **6,949 seconds total**. They include CUDA-toolkit discovery and scheduler-duration validation failures; neither made an API call. Scientific source, model, sampling and cohort identities were preserved.

The final random-control cell required no local critic. Its CPU attempt failed during Chromium startup before an API call. A zero-API diagnostic reproduced failure with the project temporary directory and passed with a short local temporary directory; this comparison does not isolate path length from filesystem behavior. Subsequent infrastructure recovery is stored separately, with the first-attempt table unchanged. All CPU attempt and diagnostic time is conservatively accounted against the unused original allowance. No additional GPU allocation was submitted.

Two separately retained recovery attempts added no valid outcome: the first stopped before transmission because its standalone launcher omitted credential-file loading; after that repair, the second stopped with a browser-closed error after four turns. Its parent receipt was interrupted, so actual parent duration is unavailable; the full configured 217-second allowance is conservatively accounted, and a host process check confirmed no remaining worker/browser. Verified GPU allocation time plus this conservative CPU accounting totals **7,194.93 seconds**. The final shared ledger contains **USD 0.98945495 settled** and **USD 0.00609330 retained as a conservative reservation**, totaling USD 0.99554825 against the USD 30 cap. The reservation belongs to the known pre-transmission credential-loading failure; it is preserved rather than silently released.

The primary cohort settled **USD 0.98745575 across 1,034 API calls**, with no unresolved reservations at its audit cutoff. Supplemental accounting uses the same USD 30 ledger and is reported separately in the aggregate artifact. Every primary result, saved verdict and earlier API entry is preserved. The original complete-cohort audit uses its archived budget snapshot; a separate audit reconciles the later shared ledger with supplemental journals.

</details>

<details>
<summary>Frozen protocol, serving validation and reproducibility</summary>

The actor is `gpt-4o-mini-2024-07-18`, temperature 1. Shepherd uses top-p 1, up to 512 output tokens and one sample for unguided versus 20 for random/guided selection. Arbiter uses the native five-BLOCK response, top-p 0.9 and up to 2,048 output tokens; its unguided control chooses the first candidate. Both critics use temperature 0. The terminal judge is `o4-mini-2025-04-16` with the Online-Mind2Web/AgentTrek success prompt, a 4,096-token response cap and a 30-turn episode limit. Truncated actor, critic and judge responses are rejected. The token caps and serving engine are disclosed adaptations of the released interfaces.

BrowserGym 0.14.3 and Playwright 1.44.0 use the shared Chromium-1208 executable, with a fresh browser per episode and native author observation/action interfaces. Live websites, differing episode durations and later continuations limit contemporaneous comparability; this is a 12-task feasibility pilot, not a full-300 benchmark or exact reproduction of a published paper result.

Both reward servers passed simultaneous H200 startup and local preflight. Explicit KV caches are 32,768 tokens for Shepherd (4.5 GiB) and 65,536 for Arbiter (3.5 GiB), with a 0.70 memory-profiling ceiling; observed combined GPU use was approximately 40 GiB. Repeated temperature-zero Shepherd preflight responses produced nonidentical text and scores (0.00004055 and 0.00005150), so deterministic serving is not established.

All **40 focused tests passed**, including complete/partial cohort handling, recovery-budget checks and immutable API snapshot validation. The independent audit checks cohort/identity hashes, saved result equality, readable screenshots, full terminal-judge evidence, request journals, API accounting and preservation across attempts. Task payloads, screenshots, raw trajectories, credentials and internal operational paths remain private; the linked artifact contains aggregate results only.

</details>

<a id="supported-harness-20261009"></a>
## Supported-harness reproduction — October 9, 2026

Both released baselines support GPT-4o-mini as the actor, but their candidate generation, observations and critic interfaces differ. Compare each critic's improvement over controls in its own supported harness; a difference between those improvements is a comparison of complete systems, not an isolated estimate of reward-model quality.

The diagnostic used the same 12 prespecified OM2W tasks for six conditions. All conditions use the frozen `gpt-4o-mini-2024-07-18` actor and the same terminal-success judge. Random-selection controls distinguish critic selection from the benefit of generating more candidates. Twelve tasks provide a mechanical pilot with wide uncertainty, not a definitive ranking.

| Family | Unguided control | Candidate-generation control | Reward-guided condition |
| --- | --- | --- | --- |
| Web-Shepherd | One sampled action | Random choice among up to five frequency-ranked actions from 20 samples | WebShepherd 8B checklist-probability selection from that pool |
| WebArbiter | First candidate from the released five-BLOCK policy response | Random choice among the five candidates | WebArbiter 7B pairwise knockout tournament |

Arbiter's unguided control preserves its released multi-candidate prompt; it is not a separate single-action prompt. Shepherd's online search loop is reconstructed from the paper because its repository does not release that loop. Its actor prompt comes from the authors' public BrowserGym demo. Published WebArena-Lite results and static Mind2Web preference accuracy must remain separate from live Online-Mind2Web success.

<details>
<summary>Source evidence and reproduction limits</summary>

| Component | Source | Immutable revision |
| --- | --- | --- |
| Shepherd critic prompts, label-probability parser and score | [Official repository](https://github.com/kyle8581/Web-Shepherd) | `b87898f614e43d7a5bdb738eaf6b01d416be24c2` |
| Shepherd actor prompt and preprocessing | [Author demo source](https://huggingface.co/spaces/LangAGI-Lab/Web-Shepherd-Demo/blob/main/app.py) | File digest recorded in the pilot manifest |
| Arbiter policy prompt/parser, observations, actions and tournament | [Official repository](https://github.com/YaoZ720/WebArbiterCode) | `764ec2670750c50388bac56efb1a3989fd5c1b54` |
| Shepherd 8B weights | [Author model](https://huggingface.co/LangAGI-Lab/WebShepherd_8B) | `37b57dbaa1e4c0c19b062172a7077b0b632a66d0` |
| Arbiter 7B weights | [Author model](https://huggingface.co/ZYao720/WebArbiter-7B) | `6518479bcf55043b92d5ffdc34e23ce7ed59bf65` |

The Shepherd serving script names `WebShepherd/WebShepherd_8B`. Public repository metadata shows the same 15 file objects, including all weight-shard LFS digests, as the staged LangAGI-Lab checkpoint. This is checked rather than assumed from model names.

Arbiter's released viewer critic prompt differs from the benchmark prompt used with its released model. The pilot uses the official benchmark's WebArena prompt template, recovered with byte-exact roundtrips over all 4,600 released rendered rows. This combination of viewer actor and benchmark critic is an explicit adaptation, not evidence that the paper used precisely this online loop. The static rows supply a template only; their labels and trajectories are not live-task inputs.

The completed manifest records hosted model snapshots, dependency versions, browser lifecycle, token limits, serving engine, malformed-output policy and thought/action splitting. Actor generation, full critic requests, response log probabilities, selection brackets and execution errors are saved privately. Highest available source fidelity does not establish exact reproduction of an unreleased paper harness.

</details>

<details>
<summary>Budget, analysis and evidence retention</summary>

The existing pilot authorization is one H200 for two hours, eight CPUs, 120 GiB RAM and a shared USD 30 API cap, including all retries. It does not authorize full-300 collection or a new allocation. Model staging and public local fixtures use no paid API calls. The allocation controller must own and await every server and evaluation worker, preserve consumed time across retries, and stop within both caps.

Report successes, valid attempts and planned tasks in separate count columns. Primary success uses all planned tasks in each arm as the denominator; report valid-only success separately. Compare within-family paired differences in percentage points with a paired task bootstrap and common-valid sensitivity. Relative percentage gain is undefined when the control has zero success. Incomplete cohorts cannot silently become completed comparisons, and invalid or malformed results cannot be replaced by whichever retry succeeds.

Retain task-file/cohort hashes, immutable model revisions, source/dependency manifest, shared API ledger, allocation attempt ledger, every task rollout and terminal verdict. Publish only aggregate results and methodology; task payloads and browser evidence remain local.

</details>
