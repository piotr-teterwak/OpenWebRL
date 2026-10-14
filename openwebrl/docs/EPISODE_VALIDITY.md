# Episode validity: one rule for RL, branch collection and MolmoWeb

<a id="episode-validity-20261012"></a>

**Rule (adopted 2026-10-12):** an episode is **invalid** when infrastructure ended it; anything the agent caused is a **valid** outcome, scored by that system's own judge. Wall-clock timeouts and model-server aborts count as infrastructure.

**Use going forward:**

| Purpose | Code |
|---|---|
| Classify one episode | [`openwebrl/outcome_validity.py`](../outcome_validity.py): `classify_terminate_reason`, `classify_rl_trajectory`, `classify_branch`, `classify_molmoweb` |
| RL sample removal (new runs) | `openwebrl/generate_browser.py:_should_remove_browser_sample` calls `classify_terminate_reason` |
| Relabel saved episodes (read-only) | [`scripts/relabel_outcome_validity.py`](../../scripts/relabel_outcome_validity.py) `--out <summary.json>` (use a CPU-only job; the RL archives are ~138 GB) |
| Tests | [`tests/test_outcome_validity.py`](../../tests/test_outcome_validity.py) |

## Main result

Relabeling every saved episode (2026-10-12, CPU job 357123) leaves branch collection and MolmoWeb almost unchanged. It shows that the 11 archived RL runs (Sep 10–24) **trained on 940 infrastructure failures as reward-0 negatives**.

| System | Episodes | Valid (native rule) | Valid (shared rule) | Changed | Changed (%) |
|---|---|---|---|---|---|
| Branch collection (old shards + fast1k) | 9,718 | 6,863 | 6,870 | 9 | 0.09 |
| RL training (11 runs with rollout archives) | 67,610 | 61,561 | 60,621 | 940 | 1.39 |
| MolmoWeb (N1 and N5 selector) | 4,132 | 4,008 | 4,004 | 4 | 0.10 |

- **RL:** 849 episodes failed at browser **reset**, with text `env_server error [500] /reset`. The old removal markers looked for `env_server /reset error`, so these were kept. Another 91 model-server errors (`generation_error`) were also kept as reward 0. Since 2026-10-03 (`0290863`), `arm` has removed every ABORTED sample, which already excludes both; the shared rule keeps that behaviour and names the cause. Runs launched from preserved sources older than that commit still use the old markers. Past checkpoints are unchanged.
- **Branch:** 8 actor-prompt context-limit endings become valid (agent-caused, as in RL). 1 episode-deadline ending becomes invalid. Usable-state counts are unaffected.
- **MolmoWeb:** 4 completed episodes without a native judge verdict become invalid.

Source: `/gpfs/scrubbed/zixianma/openwebrl-runtime/outcome-validity-relabel-20261012/relabel-full.json`.

<a id="benchmark-validity-audit-20261013"></a>
## Benchmark success rates: exact-cohort rerun — October 13

**The shared rule leaves all 5,400 independently mapped, auditable benchmark episodes unchanged.** Original judge verdicts and overall success totals are preserved. The earlier 940-failure correction concerns **training rollouts**, and must not be applied as a numerical adjustment to checkpoint-evaluation tables.

These historical Online-Mind2Web rows were reconstructed from exact saved episode files, not inferred from a matching percentage:

| Experiment / cohort | Tasks | Successes | Native valid | Shared-rule valid | Overall % | Shared-rule valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting SFT, original dedicated baseline | 300 | 90 | 267 | 267 | 30.00 | 33.71 |
| ScalarRM, same dedicated collection | 300 | 114 | 251 | 251 | 38.00 | 45.42 |
| SelectionARM, same dedicated collection | 300 | 128 | 256 | 256 | 42.67 | 50.00 |
| C2 update 500, fixed 100 + fresh holdout 200 | 300 | 95 | 258 | 258 | 31.67 | 36.82 |
| 1A endpoint 263, fixed 100 + fresh holdout 200 | 300 | 100 | 256 | 256 | 33.33 | 39.06 |
| Joint C2 + Piotr SFT, update 174, fresh 300 | 300 | 102 | 270 | 270 | 34.00 | 37.78 |
| Joint C2 + Piotr DPO, update 174, fresh 300 | 300 | 104 | 254 | 254 | 34.67 | 40.94 |

Valid-only success is **shared-rule-valid successes / shared-rule-valid episodes**. The seven full cohorts above contain 2,100 distinct episode artifacts. The C2/1A fixed-100 and holdout-200 subgroup checks reuse those artifacts and are not additional episodes. Every source result set matches its frozen manifest; the subgroup IDs match `arm_c2_full300_cohorts.json`, and each complete cohort matches the pinned 300-task file. Original native counts must reproduce the documented row before relabeling is accepted. Judge failures are read from saved flags; step/context/format endings remain valid failures under the shared rule. No missing outcome is imputed or rejudged.

[Aggregate counts, exact experiment/source bindings and proof hashes](arm_results/outcome_validity_20261013/benchmark-audit.json).

<details>
<summary>Eleven additional RL evaluation cohorts: 3,300 saved episodes, unchanged</summary>

| Exact evaluation cohort | Judge / protocol | Tasks | Successes | Native valid | Shared-rule valid | Shared-rule valid-only % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original bonus90, `349508` | Local GPT-4.1, T0 | 300 | 121 | 228 | 228 | 53.07 |
| Mixed bonus90, `340425` | Local GPT-4.1, T0 | 300 | 122 | 233 | 233 | 52.36 |
| Mixed reweight90, `340423` | Local GPT-4.1, T0 | 300 | 113 | 227 | 227 | 49.78 |
| Outcome-only G5/56 groups, iteration60 | Local GPT-4.1, T0 | 300 | 117 | 245 | 245 | 47.76 |
| Uniform G8, iteration60 | Local GPT-4.1, T0 | 300 | 127 | 234 | 234 | 54.27 |
| Mixed bonus90, repeat1, `346992` | Stealth o4-mini, T0.6 | 300 | 182 | 299 | 299 | 60.87 |
| Mixed bonus90, repeat2, `346994` | Stealth o4-mini, T0.6 | 300 | 181 | 294 | 294 | 61.56 |
| Mixed bonus90, repeat3, `346996` | Stealth o4-mini, T0.6 | 300 | 168 | 297 | 297 | 56.57 |
| Mixed reweight90, repeat1, `346993` | Stealth o4-mini, T0.6 | 300 | 170 | 298 | 298 | 57.05 |
| Mixed reweight90, repeat2, `346995` | Stealth o4-mini, T0.6 | 300 | 166 | 295 | 295 | 56.27 |
| Mixed reweight90, repeat3, `346997` | Stealth o4-mini, T0.6 | 300 | 170 | 298 | 298 | 57.05 |

Task IDs match each exact evaluation manifest. Terminal metadata was read with a restricted, tensor-free ZIP/pickle reader; unapproved pickle globals are rejected. Normalized exports retain original sidecar hashes, the ZIP `data.pkl` member hash and explicit terminal/judge flags. These exports were passed through the same relabel CLI. The differing judges and collection dates remain distinct; a common validity rule does not make their success rates controlled comparisons.
</details>

<details>
<summary>Evidence gaps: historical rates retained without a new shared-rule claim</summary>

Eleven mapped cohorts retain verdict sidecars but lack their referenced rollout payloads: local additive90 and Gate B90, plus all nine baseline/additive/Gate B stealth90 repeats. Noncompleted sidecars omit the terminal reason, so their shared-rule denominators cannot be independently reconstructed from those files. Merged original/recovery selections are retained; no retry is substituted for a missing episode. The historical local baseline90 table also lacks an exact saved-cohort mapping in the inspected aggregate. Those numbers remain historical native-validity results. Other checkpoint panels and inference cohorts are outside this targeted audit; they are not implicitly revalidated.
</details>

<details>
<summary>Reproduce the audit without scanning training archives</summary>

```bash
python3 scripts/relabel_outcome_validity.py \
  --evaluation-manifest /path/to/private-cohort-manifest.json \
  --out /path/to/recomputed-report.json
```

This mode requires exact expected task IDs, per-episode artifact hashes and documented native counts. It refuses mismatched or duplicate cohorts and never scans the training archives. Private manifests, per-episode evidence and normalizer receipts stay under runtime `outcome-validity-docs-20261013/sft/` and `evaluations/outcome-validity-docs-20261013/preparation-v1/headline-eval-audit/`. The default relabel mode still audits RL training, branch collection and MolmoWeb. This rerun used saved artifacts only, with no new GPU allocation, browser session or model/judge request.
</details>

## Endings

| Ending | Valid | Meaning | RL signal | Branch signal | MolmoWeb signal |
|---|---|---|---|---|---|
| `done` | yes | agent finished; judge decides success | `task_completed` | rollout `task_completed`, terminal `done` | `completed` |
| `step_limit` | yes | step budget used | `max_steps_exhausted` | — | inside `completed` (30 steps) |
| `action_limit` / `decision_limit` | yes | controlled-browser budgets | — | `browser_terminal_reason` / `policy_boundary_reason` | — |
| `context_limit` | yes | response or context too long | `generation_length_limit`, `rollout_max_context_length_exceeded` | prompt "must be below" guard | — |
| `format_failure` | yes | unparseable actions | `format_error_failed` | same | — |
| `requires_user_input` | yes | task needs a user | — | boundary/terminal reason | — |
| `browser_env` | no | browser step, navigation, observation or reset failure | `env_step_error`, `env_server …` | `env_step_error` without a limit | `completed_invalid` |
| `model_server` | no | actor or selector abort, timeout or outage | `generation_abort`, `generation_error: …` | same | `ACTION_PREDICTION_ERROR` (503s, disconnects) |
| `wall_timeout` | no | whole-episode clock | `rollout_task_timeout` | `episode_deadline` | `completed_timeout` |
| `capture` | no | final observation missing on a completed episode | — | `missing_fresh_terminal_evidence` | — |
| `judge` | no | judge timeout, error or missing verdict | `judge_timeout` / `judge_invalid` | `judge_budget_halt`, no score | native verdict missing |
| `contract` / `unknown` | no | budget or contract guard; unrecognized | `no_generated_turns` | input/budget guard | `budget_blocked` |

The branch controlled browser enforces its 30-operation limit inside `env.step`. An action-limit ending therefore appears as `env_step_error`, and `classify_branch` checks the limit first.

<details>
<summary>Native rules before unification, and what still runs them</summary>

| System | Native rule location | Main difference from the shared rule |
|---|---|---|
| RL | `_should_remove_browser_sample` in each preserved source (e.g. `reference-stage1-to100-20260922-24h-v3`); `reward_browser.py` judge timeouts; `slime/ray/rollout.py` drops whole trajectories | before `0290863`: kept model-server aborts and `/reset` failures as reward 0; after: removed every ABORTED sample (equivalent in practice) |
| Branch collection | `pilot32-v1/source/openwebrl/controlled_sft_worker.py:480-551` (`judge_terminal`) | context-limit guard was invalid; episode deadline was valid |
| MolmoWeb | `molmoweb_rollout_worker.py:41, 343-351`, `molmoweb_timeout.py` | already matched, apart from a missing judge verdict |

The running fast1k branch collection and the MolmoWeb evaluation keep their frozen code until their runs finish; report their counts through the relabeler. Their next versions should call `openwebrl.outcome_validity` directly. Each system keeps its own success judge (RL: `reward_browser.py`; branch: canonical OM2W o4-mini; MolmoWeb: native judge), so valid-only success rates remain non-comparable across systems.
</details>
