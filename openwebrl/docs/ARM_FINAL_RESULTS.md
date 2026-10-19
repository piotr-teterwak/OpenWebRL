# Final inference results

## Headline performance cost and efficiency

<!-- three-benchmark-headlines-v1 -->
**Four methods × three benchmarks · BrowserCode · October 19, 2026.** N3/N5 use high proposals and a high selector; both single-call methods use high effort. Limits are 12,288 output tokens per call and 30 decisions.

Luna single on OM2W is **not run**; DeepShop Sol is **incomplete**. These cells are marked explicitly rather than assigned a performance rate.

### Matched task success

Common-valid cohorts are **256 OM2W tasks across Sol/N3/N5, 489 WebVoyager tasks across all four methods, and 145 DeepShop tasks across the three Luna methods**. These rates condition on validity; the full fixed-task rates follow.

| Method | Online-Mind2Web · 256 | WebVoyager · 489 | DeepShop · 145 |
|---|---:|---:|---:|
| GPT-6 Sol single high | 64.5% | 72.2% | — |
| GPT-6 Luna single high | — | 69.1% | 49.0% |
| GPT-6 Luna N3 high | 66.8% | 69.9% | 54.5% |
| GPT-6 Luna N5 high | 69.5% | 72.0% | 52.4% |

### Overall task success

Rates use every fixed task, including invalid outcomes as unsuccessful. Unrun or incomplete methods have no headline rate; counts are listed separately below.

| Method | Online-Mind2Web · 300 | WebVoyager · 595 | DeepShop · 150 |
|---|---:|---:|---:|
| GPT-6 Sol single high | 57.0% | 64.7% | Incomplete |
| GPT-6 Luna single high | Not run | 65.5% | 48.7% |
| GPT-6 Luna N3 high | 60.0% | 66.1% | 54.0% |
| GPT-6 Luna N5 high | 62.7% | 65.4% | 51.3% |

OM2W uses the original 300-task secondary cohort; the primary comparison on 200 new tasks is retained below. DeepShop Sol has 31 budget-affected cells and two additional invalid captures; it is excluded from the DeepShop shared-valid comparison.

### Model API cost across all attempts

Includes every proposal/selector and invalid attempt, with conservative unknown-call charges. Judge and compute costs are excluded; dollar amounts are frozen-rate receipt estimates. Separate OM2W retry cohorts and the separate Sol multi-proposal pilot are excluded.

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | $57.21 | $115.60 | $47.88 · partial |
| GPT-6 Luna single high | Not run | $4.61 | $2.10 |
| GPT-6 Luna N3 high | $15.75 | $25.81 | $13.20 |
| GPT-6 Luna N5 high | $19.68 | $27.61 | $16.22 |

### Matched efficiency

Each benchmark uses exactly the same common-valid tasks for its included methods: **OM2W 256, WebVoyager 489, DeepShop 145**. OM2W includes Sol/N3/N5; WebVoyager includes all four methods; DeepShop includes the three Luna methods. Latency excludes judge-only work. Browser steps count selected BrowserCode batches.

#### Mean model API cost per matched task

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | $0.2046 | $0.1882 | — |
| GPT-6 Luna single high | — | $0.0073 | $0.0140 |
| GPT-6 Luna N3 high | $0.0546 | $0.0407 | $0.0892 |
| GPT-6 Luna N5 high | $0.0699 | $0.0443 | $0.1106 |

#### Mean task latency

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | 91.5 s | 89.5 s | — |
| GPT-6 Luna single high | — | 94.1 s | 154.2 s |
| GPT-6 Luna N3 high | 177.7 s | 156.9 s | 259.1 s |
| GPT-6 Luna N5 high | 208.0 s | 167.0 s | 283.7 s |

#### Mean browser steps

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | 14.9 | 12.7 | — |
| GPT-6 Luna single high | — | 12.9 | 20.6 |
| GPT-6 Luna N3 high | 14.8 | 11.3 | 18.0 |
| GPT-6 Luna N5 high | 14.7 | 10.5 | 17.6 |

[Audited aggregate metrics](arm_results/sol6_luna_three_benchmarks_20261019/aggregate.json) · [Definitions and audit scope](ARM_INFERENCE_SCALING.md#sol6-luna-three-benchmarks-20261019)

<details>
<summary>Coverage and success counts</summary>

| Benchmark | Method | Fixed tasks | Successes | Valid | Invalid | Status |
|---|---|---:|---:|---:|---:|---|
| Online-Mind2Web | GPT-6 Sol single high | 300 | 171 | 266 | 34 | Audited |
| Online-Mind2Web | GPT-6 Luna single high | 300 | — | — | — | Not run |
| Online-Mind2Web | GPT-6 Luna N3 high | 300 | 180 | 269 | 31 | Audited |
| Online-Mind2Web | GPT-6 Luna N5 high | 300 | 188 | 267 | 33 | Audited |
| WebVoyager | GPT-6 Sol single high | 595 | 385 | 553 | 42 | Audited |
| WebVoyager | GPT-6 Luna single high | 595 | 390 | 579 | 16 | Audited |
| WebVoyager | GPT-6 Luna N3 high | 595 | 393 | 568 | 27 | Audited |
| WebVoyager | GPT-6 Luna N5 high | 595 | 389 | 555 | 40 | Audited |
| DeepShop | GPT-6 Sol single high | 150 | 55 | 117 | 33 | Incomplete; 31 budget-affected cells |
| DeepShop | GPT-6 Luna single high | 150 | 73 | 150 | 0 | Audited |
| DeepShop | GPT-6 Luna N3 high | 150 | 81 | 148 | 2 | Audited |
| DeepShop | GPT-6 Luna N5 high | 150 | 77 | 147 | 3 | Audited |

</details>

## OpenWebRL-4B actor with action selectors

<!-- selection-head-4b-v1 -->
**Five-candidate selection for the OpenWebRL-4B-SFT actor · local browser · October 18–19, 2026.** Every task of each benchmark, one episode per task, the same tasks for every method. Actor decoding T 1.0 / top-p 0.9, 1,024 tokens per step. Judges: o4-mini for Online-Mind2Web; GPT-4o for WebVoyager and DeepShop. A different actor and harness from the GPT-6 rows above, so the two sections are not directly comparable.

**A 22.6 M-parameter selection head on the actor's hidden states, trained on Piotr's data, matches Piotr SelectionARM's separate 4.44 B selector at 64% of its compute; both beat a single sample by about 7 pp at 30 steps. Raising the step limit from 30 to 100 adds about 5–6 pp for both SFT and Piotr SelectionARM, and the selector's lead over SFT holds (+5.9 pp).**

### Overall task success, 30 steps

Rates use every fixed task, with invalid outcomes counted as unsuccessful. − SFT is the paired difference on all 1,045 tasks, pp [95% CI].

| Method | Online-Mind2Web · 300 | WebVoyager · 595 | DeepShop · 150 | All · 1,045 | − SFT |
|---|---:|---:|---:|---:|---:|
| SFT 4B, 1 sample | 36.3% | 44.4% | 34.0% | 40.6% | — |
| Piotr SelectionARM (4.44 B selector) | 41.0% | **52.4%** | 41.3% | 47.6% | +7.0 [+3.9, +10.0] |
| Selection head, Piotr's data | **43.7%** | 51.3% | 44.0% | **48.0%** | +7.5 [+4.4, +10.5] |
| Selection head, our K = 1 labels (1,908 SFT + RL states) | 37.3% | 49.1% | **45.3%** | 45.2% | +4.6 [+1.5, +7.6] |
| Selection head, our K = 1 labels (1,631 SFT-actor states) | 35.0% | 49.6% | 42.7% | 44.4% | +3.8 [+0.8, +6.9] |

### Overall task success, 100 steps

Same tasks and decoding; the last 20 turns are kept in full and older turns are reduced to their action and tool-result line. Gain is the paired difference from the same method at 30 steps.

| Method | Online-Mind2Web · 300 | WebVoyager · 595 | DeepShop · 150 | All · 1,045 | Gain vs 30 steps | − SFT |
|---|---:|---:|---:|---:|---:|---:|
| SFT 4B, 1 sample | 40.0% | 50.8% | 42.0% | 46.4% | +5.8 [+2.9, +8.8] | — |
| Piotr SelectionARM | **44.3%** | **56.5%** | **52.0%** | **52.3%** | +4.8 [+1.8, +7.9] | +5.9 [+3.0, +9.0] |

### Efficiency

Averaged over all 1,045 tasks. Parameters are those loaded at inference. PFLOP per step counts every actor and selector token (2 × parameters × uncached prompt + generated tokens). Seconds per decision is the wall clock from the first candidate request to the pick, with 8 episodes per H200.

| Method | Steps | Parameters | Added for selection | PFLOP per step | Seconds per decision | Browser steps per episode |
|---|---:|---:|---:|---:|---:|---:|
| SFT 4B, 1 sample | 30 | 4.44 B | — | 0.019 | 2.81 | 13.6 |
| Piotr SelectionARM | 30 | 8.88 B | 4.44 B | 0.096 | 8.68 | 14.3 |
| Selection head (any training data) | 30 | 4.46 B | 22.6 M | 0.060–0.062 | 8.2–8.3 | 13.6–15.6 |
| SFT 4B, 1 sample | 100 | 4.44 B | — | 0.067 | 5.16 | 24.0 |
| Piotr SelectionARM | 100 | 8.88 B | 4.44 B | 0.163 | 11.87 | 26.3 |

Compute per step rises at 100 steps because late steps carry longer prompts.

[Selection-head results and method](ARM_SELECTION_HEAD.md#selection-head-crossbench-full-20261018) · [30-step evidence](arm_results/rl_integration/crossbench-full-t10-5cond-20261019.json) · [100-step evidence](arm_results/rl_integration/crossbench-full-h100-20261019.json)

<details>
<summary>Selector coverage and success counts</summary>

| Steps | Benchmark | Method | Fixed tasks | Successes | Valid | Invalid | Context overflows |
|---|---|---|---:|---:|---:|---:|---:|
| 30 | Online-Mind2Web | SFT 4B, 1 sample | 300 | 109 | 275 | 25 | 1 |
| 30 | Online-Mind2Web | Piotr SelectionARM | 300 | 123 | 263 | 37 | 11 |
| 30 | Online-Mind2Web | Head, Piotr's data | 300 | 131 | 265 | 35 | 6 |
| 30 | Online-Mind2Web | Head, 1,908 SFT + RL states | 300 | 112 | 261 | 39 | 11 |
| 30 | Online-Mind2Web | Head, 1,631 SFT-actor states | 300 | 105 | 259 | 41 | 14 |
| 30 | WebVoyager | SFT 4B, 1 sample | 595 | 264 | 587 | 8 | 3 |
| 30 | WebVoyager | Piotr SelectionARM | 595 | 312 | 580 | 15 | 4 |
| 30 | WebVoyager | Head, Piotr's data | 595 | 305 | 571 | 24 | 11 |
| 30 | WebVoyager | Head, 1,908 SFT + RL states | 595 | 292 | 579 | 16 | 9 |
| 30 | WebVoyager | Head, 1,631 SFT-actor states | 595 | 295 | 582 | 13 | 9 |
| 30 | DeepShop | SFT 4B, 1 sample | 150 | 51 | 149 | 1 | 1 |
| 30 | DeepShop | Piotr SelectionARM | 150 | 62 | 148 | 2 | 1 |
| 30 | DeepShop | Head, Piotr's data | 150 | 66 | 147 | 3 | 2 |
| 30 | DeepShop | Head, 1,908 SFT + RL states | 150 | 68 | 146 | 4 | 2 |
| 30 | DeepShop | Head, 1,631 SFT-actor states | 150 | 64 | 146 | 4 | 1 |
| 100 | Online-Mind2Web | SFT 4B, 1 sample | 300 | 120 | 277 | 23 | 0 |
| 100 | Online-Mind2Web | Piotr SelectionARM | 300 | 133 | 255 | 45 | 14 |
| 100 | WebVoyager | SFT 4B, 1 sample | 595 | 302 | 584 | 11 | 6 |
| 100 | WebVoyager | Piotr SelectionARM | 595 | 336 | 569 | 26 | 15 |
| 100 | DeepShop | SFT 4B, 1 sample | 150 | 63 | 143 | 7 | 6 |
| 100 | DeepShop | Piotr SelectionARM | 150 | 78 | 146 | 4 | 3 |

- **Training data.**
  - Piotr's data: GPT-5.5 picks on 2,692 states (35,916 five-candidate panels). The head on Piotr's data is the listwise head.
  - Our K = 1 labels: Luna lookahead votes, one panel per state.
- **Invalid episodes:** mostly site resets and browser step errors common to all methods; context overflows are listed separately because the selector's picks lead to them.

</details>

<details>
<summary>Original controlled Online-Mind2Web comparison and full-300 cohort</summary>

### GPT-6 Sol single high versus GPT-6 Luna N3 and N5

**Online-Mind2Web 300 · BrowserCode · Final audited results · October 18, 2026.** All proposals and selectors use high reasoning effort; the limit is 30 decisions per episode.

**Luna reduces model API cost per task by 73.2% (N3) and 66.6% (N5), while taking 1.99× and 2.33× as long on the primary shared-valid comparison.** Its observed success rates are higher, but neither primary paired gain is statistically established.

#### Primary comparison on new tasks

The new 200-task cohort is primary. Efficiency below uses exactly the same 165 tasks valid in all three arms. Model cost includes every proposal and selector call; model cost and task latency exclude the judge.

| Model and policy | Shared valid tasks | Successes | Success rate | Mean model cost / task | Mean task latency | Model cost / successful task | Cost reduction vs Sol | Latency vs Sol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 165 | 104 | 63.0% | $0.2084 | 94.1 s | $0.3306 | 0.0% | 1.00× |
| GPT-6 Luna N3 high + high selector | 165 | 108 | 65.5% | $0.0558 | 186.9 s | $0.0852 | 73.2% | 1.99× |
| GPT-6 Luna N5 high + high selector | 165 | 114 | 69.1% | $0.0696 | 219.7 s | $0.1008 | 66.6% | 2.33× |

Cost per successful task is total model cost across these 165 tasks divided by successes, including cost spent on valid failures. It excludes infrastructure-invalid attempts. N3’s paired success difference is +2.4 percentage points (95% CI −4.8 to +9.7); N5’s is +6.1 points (−0.6 to +12.7).

#### Full 300 task results

The combined cohort is secondary because the first 100 Luna tasks had already been observed. All 900 original dispositions are retained; retry outcomes are separate. Overall success divides by all 300 tasks; valid-only success divides by each arm’s valid count.

| Model and policy | Tasks | Valid | Invalid | Successes | Overall success | Valid-only success |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 300 | 266 | 34 | 171 | 57.0% | 64.3% |
| GPT-6 Luna N3 high + high selector | 300 | 269 | 31 | 180 | 60.0% | 66.9% |
| GPT-6 Luna N5 high + high selector | 300 | 267 | 33 | 188 | 62.7% | 70.4% |

For a matched comparison across the full cohort, the following rows use only the same 256 tasks valid in all three arms.

| Model and policy | Shared valid tasks | Successes | Success rate | Mean model cost / task | Mean task latency | Model cost / successful task |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 256 | 165 | 64.5% | $0.2046 | 91.5 s | $0.3175 |
| GPT-6 Luna N3 high + high selector | 256 | 171 | 66.8% | $0.0546 | 177.7 s | $0.0817 |
| GPT-6 Luna N5 high + high selector | 256 | 178 | 69.5% | $0.0699 | 208.0 s | $0.1005 |

[Detailed comparison and audit](ARM_INFERENCE_SCALING.md#sol6-luna-browsercode300-20261018) · [Audited aggregate metrics](arm_results/sol6_luna_browsercode300_20261018/aggregate.json)

</details>
