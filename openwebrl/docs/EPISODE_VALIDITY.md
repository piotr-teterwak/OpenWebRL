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
