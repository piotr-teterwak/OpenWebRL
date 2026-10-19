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
