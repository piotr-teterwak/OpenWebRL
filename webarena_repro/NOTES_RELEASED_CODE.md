# Released WebArbiter code (YaoZ720/WebArbiterCode @ 764ec26): what it actually does

Read-only audit, 2026-11-01. Paths are relative to `viewer/`. Evidence for PROTOCOL.md §2 and §5.

## It cannot run Table 5 as shipped

- No headless runner. `scripts/run_*_som.sh` call a `run.py` that is not in the repo; `utils/run_evaluation.py`
  is an interactive GPT-4o judge over external trajectories; `cli_instructions/*.md` are frontend dev briefs.
- The only automatic path is `POST /api/v2/runs/step-loop[/batch]` (`api/routes/trajectories.py:67-114`).
  Per step it runs terminal check → sampling → `rerank(strategy="hybrid")` → executes
  `rerank_result.candidates[0]` (`service/executors/step_loop_executor.py:270-443`). **No reward model, no
  tournament.** `reward_model_spec` is built (`auto_run.py:167-170`) and only logged.
- `CoordinatorAgent.execute_step` (sample → rerank "confidence" → `run_tournament`,
  `agents/coordinator_agent.py:261-399`) is the intended pipeline but is never instantiated.
- The tournament runs only in manual UI mode: `/sampling` → `/rerank` → `/tournament/run` → `/runs/{id}/execute`
  per step. The UI hard-codes `model:"gpt-4o-mini"` for the policy (`services/api.ts:262-266`).
- `configs/reward_guided_agent.json` (num_samples 10, num_calls 3, max_tournament_candidates 16, T 0.6–0.9) and
  the `env.example` variables are read by no code.

## Effective behavior

| Item | Value | Where |
|---|---|---|
| Policy decoding | T 1.0, top-p 0.9, max 2,048 (gen_config=None fallbacks) | `core/tools/sampling.py:255`, `llms/utils.py:133-142` |
| Candidates | 1 call, 5 BLOCKS with thought/action/confidence; bare JSON only (fenced output → 0 candidates) | `prompts/sample_p.py`, `core/tools/policy.py:105` |
| Filter / dedup | element IDs must exist in the observation; dedup by action string, keep max confidence, count frequency | `core/tools/sampling.py:65-133, 276-313` |
| Rerank | auto: hybrid with an LLM quality score by the policy model; manual: user-chosen, top_k 6 | `core/tools/reranking.py` |
| Bracket | FIFO single elimination, consecutive pairs; odd one carries over; N=5 → 4 comparisons | `core/tools/tournament.py:123-331` |
| Comparison parse | regex `<Answer>\s*(Response\s*[12])\s*</Answer>`; anything else → A wins; an exception drops both | `core/tools/reward.py:192-200`, `tournament.py:226-276` |
| Reward-model decoding | T 1.0, top-p 0.9, max 2,048 (README / WebPRMBench eval use greedy) | `tournament.py:409` |
| Reward-model routing | model name exactly `WebArbiter` → `OPENAI_API_URL`; anything else → api.openai.com | `llms/providers/openai_utils.py:362-374` |
| Observation | full CDP AX tree, flattened `[backendDOMNodeId] role name`, no viewport limit, no truncation | `core/browser/playwright_wrapper.py:206-223` |
| Viewport | 1024×2490 | `core/browser/browser_executor.py:431-436` |
| Terminal detector | gpt-4o-mini, T 1.0; in auto-run effectively only at step 0 | `core/tools/terminal.py:184-232`, `step_loop_executor.py:158` |
| State diff | gpt-4o-mini, computed in a daemon thread (races the next step) | `service/step_management.py:644-653` |
| Max steps | auto default 5; limit → `send_msg_to_user [N/A]` | `auto_run.py:51-52`, `step_loop_executor.py:519-589` |
| Error recovery | ReAct BrowserAgent (gpt-4o-mini) on execution failure, on by default | `core/browser/browser_agent_executor.py` |
| Judge | `gpt-4-1106-preview`, T 0 (retired) for fuzzy_match / ua_match | `evaluation_harness/helper_functions.py:501-574` |
| Reset | none called by any code; `require_reset` is false for all 165 tasks | grep |

## Other defects

- Manual tournament sends the reward model `trajectory_text: ''` and an observation with `[object Object]`
  for text-less nodes (`client/src/components/TrajectoryEditViewer.tsx:534-549`).
- Each step uses a fresh browser that reloads the stored URL, losing page state; element actions fall back to
  role+name and `type` does not press Enter (`browser_executor.py:595-660`, `playwright_wrapper.py:633-692`).
- A terminal action on the last step is overwritten by the N/A fallback (`step_loop_executor.py:484`).
- `confidence=None` breaks dedup with a TypeError (`sampling.py:286,304`).
- Silent timeouts: sampling 60 s, rerank 30 s, terminal 20 s, state diff 20 s, tournament 120 s (restarts once).
- URL maps disagree on Map (`server/utils/url_mappings.py` → openstreetmap.org; `evaluation_harness/url_service.py`
  → localhost:8085).

## Task file facts (`config_files/wa/test_webarena_lite.raw.json`)

165 tasks: shopping 45, shopping_admin 35, gitlab 30, map 26, reddit 19, multi-site 10. Evaluator types: string_match
80, url_match 39, program_html 67; LLM fuzzy_match 35 (map 18), ua_match 1. Unachievable (N/A) tasks: 27, 56, 59,
74, 86, 114, 161, 162.
