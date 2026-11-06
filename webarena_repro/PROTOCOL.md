# WebArbiter Table 5 reproduction: protocol

Status: draft, 2026-11-01. Fixed before any paid run; changes after the first run are logged at the bottom
with a date and reason.

## 1. Target

WebArbiter (arXiv 2601.21872, ICLR 2026), Table 5: reward-guided Best-of-N search on WebArena-Lite.
Policy GPT-4o (GPT-4o-mini rows optional, later). Published GPT-4o rows (per-site success, %):

| Row | Shopping | CMS | Reddit | GitLab | Map | Avg (5) | Source |
|---|---|---|---|---|---|---|---|
| No search* | 23.91 | 31.43 | 28.57 | 56.25 | 19.35 | 31.90 | WebShepherd paper, 165-task set, BrowserGym |
| GPT-4o-mini as PRM | 26.67 | 37.14 | 42.11 | 40.00 | 19.23 | 33.03 | WebArbiter authors |
| WebShepherd-8B* | 30.43 | 42.86 | 47.62 | 46.88 | 35.48 | 40.65 | WebShepherd paper |
| WebArbiter-7B | 44.44 | 42.86 | 52.63 | 56.67 | 38.46 | 47.01 | WebArbiter authors |

\* Imported rows. Their denominators (46/35/21/32/31) are the 165-task set including multi-site tasks; the
authors' own rows use the 155 single-site tasks (45/35/19/30/26). Only the two unstarred rows were run in
the authors' harness, so **the in-harness comparison is WebArbiter-7B vs GPT-4o-mini-as-PRM**, and that
gap (+14.0 on the 5-site mean) is the primary reproduction target, alongside the absolute WebArbiter row.

We add one row: our SelectionARM (`PTeterwak/OpenWebRL-4B-SelectionARM` @ 81b452d).

## 2. Why this is a reconstruction

The released code (`YaoZ720/WebArbiterCode` @ 764ec26) cannot produce Table 5 as shipped:
the batch endpoint never calls the reward model or the tournament; the tournament runs only when a person
drives the UI step by step; the UI hard-codes GPT-4o-mini as the policy; the batch loop checks for
termination only at step 0; the UI sends WebArbiter an empty trajectory; the config file
`configs/reward_guided_agent.json` is read by no code. Per-file evidence: `NOTES_RELEASED_CODE.md`.

So the method is reconstructed from the paper text plus the released components, reusing every released
prompt, parser and algorithm unchanged where it works. Each choice the paper leaves open is listed in §5;
the three most consequential get a sensitivity run (§7).

## 3. Tasks and environment

- Tasks: `viewer/config_files/wa/test_webarena_lite.raw.json` @ 764ec26, the 155 single-site tasks:
  Shopping 45, CMS (shopping_admin) 35, Reddit 19, GitLab 30, Map 26. The 10 multi-site tasks are excluded,
  matching the authors' denominators. Map runs in phase 2 (its 189 GB backend is not yet hosted).
- Images: `webarenaimages/*` on Docker Hub. **All four are identical to the original WebArena tars** on the
  official Google Drive links (2026-11-01): the image config digest read from each original `docker save` tar
  (HTTP range reads of its headers, `verify/tar_index.py`) equals the registry copy's, and the config commits
  to every layer diff_id. GitLab `3e6c8590…`, CMS `d0709482…`, Shopping `73e22b80…`, Reddit `bd8ae768…`
  (Reddit also matches Zixian's earlier check). Full layer check (`verify_layers.py`, job 410634, 2026-11-02):
  every compressed blob matches its manifest digest and every decompressed layer matches its config diff_id —
  CMS 31/31 (9.6 GB), GitLab 12/12 (77.8 GB), Reddit 29/29 (53.4 GB, Zixian's copy); uncompressed totals equal
  the original tar sizes. Shopping 28/28 (67.6 GB, job 410732). Reports: `verify/layers-*.json`.
- Hosted-site checks (`probe_site.py`, standard WebArena accounts): CMS admin login/orders/search, GitLab
  `byteblaze` login/172 public projects/issues page, Reddit `MarvelsGrantMan136` login/forum list — all pass.
- Hosting: Apptainer sandbox per site inside an unprivileged user + network namespace with loopback only
  (Zixian's recipe). Model calls leave the namespace through a UNIX-socket relay.
- **Fresh site state for every episode.** 67 of the 155 tasks modify site state (program_html checks) and
  the released code never resets, so every (task, method, repeat) cell starts from the pristine image.
- Pre-flight per site, before any paid call: services up, login works, a mutation is undone by reset, and
  the evaluator scores the reference answer 1 and a wrong answer 0 on every string_match task (catches
  evaluator, URL-mapping and judge breakage; does not prove the site data is complete).

## 4. Episode loop (one cell)

Follows the authors' intended pipeline, `CoordinatorAgent.execute_step` (sample → rerank by confidence →
knockout tournament), which the released code defines but never calls:

1. Observe: released `ScriptBrowserEnv` accessibility tree (the released run service's environment,
   `service/run_management.py`; backendDOMNodeId element IDs as in the authors' demo trajectories), viewport
   1024×2490 (`run_management.py:172`), `sleep_after_execution` 1.5 s, run through Zixian's
   `native_episode.py` wrapper (`vendor/`). Screen metrics aligned to the viewport (D15).
2. Stop check: released terminal detector, every step (released auto-run only checks step 0: bug). If it
   proposes `send_msg_to_user [...]`, that action joins the candidate pool (released manual-mode behavior).
3. Sample: one policy call with the released 5-BLOCK prompt (`prompts/sample_p.py`), released validity
   filter and deduplication.
4. Order by confidence (released `confidence` strategy); no LLM rerank (§5, D4).
5. Select: method-specific (§6).
6. Execute the selected action in the same live browser session (released per-step browser restart dropped:
   it loses page state).
7. State diff: released prompt, computed synchronously after execution (released code races it).
8. Stop when a `send_msg_to_user` action is executed or after 30 steps (then answer `N/A`, released fallback).
9. Evaluate with the released `evaluation_harness`.

## 5. Decisions the paper leaves open

| # | Choice | Main setting | Why | Alternative tested? |
|---|---|---|---|---|
| D1 | Candidate generation | 1 call, released 5-BLOCK prompt | Released code path; authors' demo shows 5 distinct candidates with stated confidences | Yes: 20 single-action samples at T 1.0, top-5 by frequency (WebShepherd App. F.2; Zixian v9) |
| D2 | Policy decoding | T 1.0, top-p 0.9, max 2,048 tokens | Effective released values (`llms/utils.py` fallbacks) | No |
| D3 | Policy snapshot | `gpt-4o-2024-08-06` | Paper unspecified; matches Zixian v9/v11 for comparability | No |
| D4 | Rerank before tournament | Confidence order only | `CoordinatorAgent.execute_step`; paper describes no LLM rerank | No |
| D5 | Tournament entrants | All valid deduplicated candidates (≤5) + terminal candidate | Paper N=5; released manual mode adds the terminal candidate | No |
| D6 | WebArbiter decoding | Greedy, max 2,048 tokens | README / model card `do_sample=False`; released tournament code uses T 1.0 | Yes: T 1.0, top-p 0.9 |
| D7 | Unparseable comparison | Candidate A wins | Released parser behavior | No (count reported) |
| D8 | Terminal handling | Terminal action joins the pool | Released manual mode; overriding ended 40/54 episodes in Zixian's v7 | Yes: terminal action overrides the step |
| D9 | Helper models (terminal detector, state diff) | `gpt-4o-mini` | Released defaults | No |
| D10 | Max steps | 30 | WebArena standard; released auto-run default of 5 is a UI setting | No |
| D11 | Retired fuzzy judge (`gpt-4-1106-preview`; 16 of the 129 four-site tasks, 35 of 155 with Map) | `gpt-4.1-2025-04-14`, released prompt, T 0 | The whole GPT-4 Turbo family (incl. `gpt-4-turbo-2024-04-09`, `gpt-4-0613`) was retired on 2026-11-02, so no GPT-4-era judge is servable; gpt-4.1 is what the no-search pilot and Zixian's re-score used | Report with and without fuzzy tasks (`wa_report.py --no-fuzzy`, A10) |
| D12 | Reward-model inputs | Full trajectory text + state diff | Paper's prompt design; the UI's empty trajectory is a bug | No |
| D13 | Fenced JSON replies | A whole-reply ```json fence is unwrapped before the released parser | Released parser `json.loads(raw)` drops fenced output (0 candidates); count logged per call (`unwrapped_fence`) | No |
| D14 | No-search terminal handling | Released override: a detector stop action is executed | Same as Zixian v8 for no search; only selector rows put it in the pool (D8) | No |
| D15 | Screen metrics | CDP screen size set equal to the viewport each step | Headless default screen 800×600 hid lower elements in the viewport-only tree (Zixian v8 diagnosis) | Could test released (no override) |

### Environment and runtime adaptations (not method choices)

- No-root hosting (`wa_site.py`): native in-image fakeroot; GitLab `runsvdir` started directly with the
  environment kept, `gitlab-ctl reconfigure` in the same fakeroot session, PostgreSQL local-socket `trust`
  (peer auth reads the kernel's real uid), `gitlab_kas` and monitoring off; stale PostgreSQL PID/lock moved
  aside and data dirs chowned in-session (Reddit); nginx moved to the task port (Magento, Postmill).
- Site origins `http://localhost:<port>`, the released URL-mapping defaults, so released start URLs,
  `url_match` and `program_html` checks run unchanged.
- Postmill `@RateLimit` annotations removed and cache cleared, as the authors' own `configure_websites.sh`.
- Released auto-login (`browser_env/auto_login.py`) builds each task's cookies inside the namespace; the
  pinned Chromium (build 1208) path is injected because Playwright 1.44 expects build 1117.
- Python: Zixian's browser-env (3.12, Playwright 1.44) + their `venv` (openai 2.6.1, httpx 0.28.1) +
  `pydeps/` (nltk 3.9.1 with punkt_tab, aiolimiter 1.2.1). SDKs for providers we never call (Anthropic,
  DashScope, Google/Vertex, HF `evaluate`) get placeholder modules that raise if used.
- Model egress: `wa_relay.py` (namespace loopback → UNIX socket → cluster proxy), given only to OpenAI-bound
  clients; the browser has no proxy.
- Validity: a cell is invalid if any model call errored or none was made (never scored as a failure).
- Released `evaluators.py` imports only three helpers, so 14 tasks whose `program_html` checks `eval()` a helper
  (Reddit 133–140, 144: `reddit_get_post_url`; Shopping 104, 117, 118: `shopping_get_latest_order_url`; GitLab 112,
  130: `gitlab_get_project_memeber_role`) raise NameError. WebArena's original import of the released
  `helper_functions` module is restored.
- An unparseable action (e.g. the policy's `no_action_needed`) executes a none action and the episode continues, as
  WebArena's `run.py`; the wrapper used to abort and void the cell.

## 6. Methods (rows)

| Row | Selection |
|---|---|
| No search | Policy prompt edited to request exactly one BLOCK (Zixian's `single_blocks_envelope_v2`); executes it |
| GPT-4o-mini as PRM | Released pairwise reward prompt and knockout bracket with `gpt-4o-mini` as judge |
| WebShepherd-8B | `LangAGI-Lab/WebShepherd_8B` @ 37b57dba; released checklist + per-candidate score; argmax, ties to the lower index |
| WebArbiter-7B | `ZYao720/WebArbiter-7B` @ 6518479b; released knockout tournament |
| SelectionARM (ours) | Released selection prompt builder over the same 5 candidates + screenshot; Zixian's v11 action-description adapter |

All rows share steps 1–4 and 6–9 of §4, so they differ only in selection.

## 6a. Sanity target: match the published numbers first (user requirement, 2026-11-02)

Our added rows are only trusted once this harness reproduces the authors' own rows. Targets (GPT-4o policy,
per site Shopping / CMS / Reddit / GitLab, 4-site mean): **GPT-4o-mini as PRM 26.67 / 37.14 / 42.11 / 40.00
(36.5)** and **WebArbiter-7B 44.44 / 42.86 / 52.63 / 56.67 (49.2)**; Map later. "Match" = 4-site mean inside our
95% interval and the WebArbiter − GPT-4o-mini gap (+12.7) reproduced. Until then, every difference from the
authors' likely setup is ablated one at a time on the same tasks (paired), highest expected impact first:

| # | Difference | Main setting | Alternative(s) to switch to |
|---|---|---|---|
| A1 | Observation / browser path | `ScriptBrowserEnv` (released run service), live session | Server UI path the tournament actually ran in: `browser_executor` full CDP tree, 1024×2490, browser reloaded per step |
| A2 | Stop detector | Every step; stop joins the selector pool | Released auto-run behavior (step 0 only); override |
| A3 | Max steps | 30 | Released UI values 5 / 8 / 10 / 16 (the authors' demo episodes are ≤ 5 steps) |
| A4 | Candidate generation | 1 call, 5-BLOCK, T 1.0 / top-p 0.9 | Several merged "Generate" rounds (UI); 20 samples top-5 (WebShepherd App. F.2) |
| A5 | Rerank before tournament | Confidence order | Released hybrid LLM rerank, top_k 6 (UI default) |
| A6 | WebArbiter decoding | Greedy | T 1.0, top-p 0.9 (released tournament code) |
| A7 | Screen / viewport | 1024×2490, screen = viewport | No screen override; 1280×720 |
| A8 | Policy snapshot | gpt-4o-2024-08-06 | gpt-4o-2024-11-20; gpt-4o-2024-05-13 if still callable |
| A9 | Helper models | gpt-4o-mini | gpt-4o |
| A10 | Fuzzy judge | gpt-4.1 | Fuzzy tasks excluded |
| A12 | Policy action space | Released `sample_p.py`: no stop action; episodes end only via the gpt-4o-mini terminal detector (D14) or the 30-step N/A fallback | `WA_POLICY_STOP=1`: `send_msg_to_user [answer]` added (WebArena `stop` wording), as the authors' demo shows. `WA_POLICY_STOP=full` (user, 2026-11-03: "add everything including report infeasible"): the whole WebArbiter-Data vocabulary — also `report_infeasible [reason]` (parsed to a send_msg_to_user action with answer N/A, WebArena's infeasible convention), `select_option [id] [option]` (released SELECT_DROPDOWN_OPTION, visible-text match), `clear [id]` (released), `noop` (released NONE). Detector unchanged |
| A13 | `type` content | Released parser types everything after the id verbatim + Enter; GPT-4o wraps content in quotes 68% of the time (1,540 of 2,265 r1 type candidates; GPT-4o-mini 26%; authors' demo 0/12), so literal quote characters are typed into search and date fields | Wrapping quotes/brackets stripped from the policy reply (`WA_TYPE_UNQUOTE=1`), as the authors' candidates had bare content |
| A14 | Judge prompt | Released `reward_p.py`: "[General Evaluation Criteria]" intro + fixed five-criteria block + Prior Knowledge + State Difference sections | The prompt WebArbiter was trained on (WebArbiter-Data; = paper Appendix D, which the paper says holds "all prompts"): two-sentence system text, no criteria block, no extra sections (`WA_ARBITER_FORMAT=training`) |
| A15 | Stop regime | D14: the gpt-4o-mini detector proposes a stop every step and it competes in the pool (conf 1.0, "has been set…") | No detector (`WA_DETECTOR=off`) with the full action space: the episode ends only when the judge picks a policy-proposed `send_msg_to_user` / `report_infeasible`, or at 30 steps (Web-Shepherd/BrowserGym regime) |
| A16 | Released stop regime + complete action set (user request) | D14 | `WA_DETECTOR=released` + `WA_POLICY_STOP=full`: the released `_detect_terminal_if_needed` as shipped (runs at step 0 only, a detected stop executes without a tournament), stop never in the pool; policy may propose `send_msg_to_user` / `report_infeasible` / `select_option` / `clear` / `noop`; no quote fix; greedy judge as in step 1 |
| A17 | Candidate generation on top of A16 (user request: "generate actions like Web-Shepherd") | D1: one call, released 5-BLOCK prompt, T 1.0 / top-p 0.9 | `WA_SAMPLING=shepherd` + the A16 settings: the one-BLOCK policy prompt (no-search envelope v2, full action set) sampled 20 times in one chat call (`n=20`, released decoding), the 5 most frequent action strings become the candidates, most frequent first (confidence = frequency/20 so the released confidence rerank keeps that order; `frequency` field set). Web-Shepherd App. F.2 / Sec. 6.2: 20 samples at T 1.0, top-n by frequency with n = 5, equal scores broken by frequency. Still the released knockout tournament, not Web-Shepherd's independent scoring |
| A18 | A17 without the terminal detector (user: "stop A17, remove the detector at step 0") | A17 | `WA_DETECTOR=off` + `WA_POLICY_STOP=full` + `WA_SAMPLING=shepherd`: the released detector runs only at step 0, so turning it off removes it entirely (A15's regime); episodes end when the tournament picks a policy-proposed `send_msg_to_user` / `report_infeasible`, or at 30 steps. No quote fix, greedy judge |
| A20 | A18 with stops merged (user, after the CMS failure audit) | A18 | `WA_MERGE_STOP=1`: all `send_msg_to_user` / `report_infeasible` samples count as one stop candidate with the summed frequency, worded as the most frequent stop; everything else as A18 (20 samples, top-5 by frequency, detector off, full action set, no quote fix, greedy judge). Account vlm |
| A11 | WebArbiter input format | Released tournament prompt (WebArena bracket actions, chat system/user, Prior Knowledge + State Difference sections, ~5k tokens median) | WebPRMBench training format (BrowserGym actions, no extra sections); decided by the offline WebPRMBench check (`webprm_check.py` C0–C3) |

The no-search row is imported from the WebShepherd paper (BrowserGym harness, 165-task set), so it is a
secondary target; matching it may need WebShepherd's own agent.

## 7. Runs

- Phase 1 (pilot): no search on GitLab + CMS, 1 repeat (65 cells). Compare to the published 56.3 / 31.4 and
  Zixian v9 23.1 / 20.7. Tests the environment explanation.
- Phase 2 (main): all 5 rows × 4 sites (129 tasks) × 3 repeats, task-major order so every row runs a task
  within the same window. Then Map.
- Sensitivity: WebArbiter row with D1, D6 and D8 each switched to its alternative, 1 repeat each.
- Per cell we save: every prompt and response, candidates, bracket, actions, screenshots, final answer,
  evaluator transcript, token usage and cost, wall time.

## 8. Validity and statistics

- Invalid cell = infrastructure failure (site or browser crash, model API error, evaluator crash). Retried
  up to 2 times from fresh state; never counted as a task failure; counts reported per row and site.
- Report per-site success and the unweighted 5-site (and 4-site) mean, as the paper does.
- Paired comparisons on tasks valid for both rows: exact sign test per repeat, and a task-level bootstrap
  (10,000 draws) for the mean difference pooled over repeats. Primary contrasts: WebArbiter vs GPT-4o-mini
  PRM, each row vs no search, ours vs WebArbiter.

## 9. Change log

- 2026-11-02: §4 step 1 changed from the server's CDP tree (`browser_executor.py`) to `ScriptBrowserEnv`
  (the released run service path, as Zixian's harness). D13–D15 and the adaptations list added while
  building the runner; no paid cell had run under the earlier wording.
- 2026-11-02 (after the no-search pilot, jobs 411181–411196): D11 judge → gpt-4-turbo-2024-04-09; evaluator helper
  import restored; parse failures → none action. The pilot's fuzzy-judged cells can be re-scored from logged calls;
  its two GitLab helper cells (112, 130) and CMS 51/59 are rerun. User approved step 1 + ablations A1–A3 (≈$200).
- 2026-11-02 (during step 1): OpenAI retired `gpt-4-turbo-2024-04-09` partway through the day (every step-1 call to it
  404'd "deprecated"; none succeeded, so no repeat mixes judges). D11 → gpt-4.1-2025-04-14; `wa_episode.py` maps the
  retired names to it, so step-1 jobs submitted with `WA_JUDGE=gpt-4-turbo-2024-04-09` use gpt-4.1 from their next
  task. The cells that hit the 404 (GitLab 34 ×3 repeats, CMS 80 r1, and any later before the switch) are rerun.
- 2026-11-02 (during step 1): ~1% of gpt-4o policy replies open a ```json fence and never close it (17 of 1,858 so
  far; another ~10% use a closed fence, already unwrapped). The v7 unwrapping required the closing fence, so those
  replies reached the released `json.loads` parser, the step got no candidates, and the episode ended on the released
  failure terminal ("I was unable to generate any valid actions…"). `wa_protocol.JSON_FENCE` now makes the closing
  fence optional (all 1,858 logged replies parse). Cells with such a reply are marked invalid (`unclosed_fence`) by
  `wa_report.py` / `wa_gaps.py` and rerun; the rule depends only on the logged reply, not on the outcome.
- 2026-11-02: observed fuzzy-judge leniency: gpt-4.1 under the released `llm_fuzzy_match` prompt graded the failure
  terminal text as "correct" against reference "0" (CMS 80, arbiter r2). Kept (released prompt), but the
  `--no-fuzzy` view (A10) is reported alongside.
- 2026-11-02 (during step 1): GitLab task 81 (WebArena 349) crashed at evaluation in every repeat (released
  `StringEvaluator` asserts `must_include` is a list; the Lite config has the bare string "yjlou", the only such task).
  `wa_episode.py` wraps a bare string in a list (original WebArena: ["yjlou"]); its cells are rerun.
- 2026-11-02 (mid step 1, user approved): mini-PRM matches the paper on GitLab/CMS while WebArbiter does not beat it on
  35 paired tasks, so A1–A3 (which act on both rows) are deferred and the WebArbiter-specific factors go first:
  (1) offline WebPRMBench check of checkpoint + serving against paper Table 2 (89.19 pairwise / 74.60 BoN) and of
  chat delivery / WebArena action syntax / released sampling (job 413126, no API cost); (2) A6 released sampling
  T 1.0 / top-p 0.9, one repeat on GitLab + CMS (jobs 413127–413130, tag A6-r1); (3) A11 if (1) shows a format effect.
  Facts found: released `model_config.py` defaults are T 1.0 / top-p 0.9 / context_length 8192 (not used to truncate);
  the authors' WebPRMBench command decodes greedily; WebArbiter-7B was trained with an 8,192-token prompt cap, and >10% of
  our reward prompts exceed it.
- 2026-11-02 WebPRMBench check (job 413126, 19 min, no API): C0 (authors' command) 89.37 pairwise / 73.16 BoN vs paper
  89.19 / 74.60 (WebArena 88.06 / 70.65 vs 88.43 / 68.66): checkpoint 6518479b and our SGLang serving reproduce Table 2.
  WebArena subset: C1 chat delivery 88.56 / 69.65, C2 + bracket syntax 89.30 / 71.14, C3 + released T 1.0 / top-p 0.9
  86.94 / 66.67. Delivery and action syntax do not matter; released sampling costs ~1.4 pairwise / ~3.5 BoN. WebPRMBench's
  WebArena prompts are all < 8,192 tokens, so prompt length beyond training is not tested by it. Stop handling is
  similar across judges (stop chosen when offered: WebArbiter 68%, mini 58%). A11 is narrowed to the extra sections and
  prompt length.
- 2026-11-02 (after step 1): 8 GitLab cells never started (reconfigure failed twice: "Peer authentication failed for
  user gitlab-psql" — reconfigure's CREATE USER can run before Postgres reloads the rendered trust entry). wa_site.py now
  puts the same `local all all trust` line into each task copy's pg_hba.conf before services start. 19 cells (mostly
  CMS) hit the released 30 s page-load timeout; they are rerun like other infrastructure invalids.
- 2026-11-02 step 1 final (388/390 valid; 2 CMS mini page-load timeouts left out): GitLab mini 38.9 ± 3.8 (paper 40.0),
  CMS mini 25.2 ± 1.5 (37.1), GitLab WebArbiter 32.2 ± 6.9 (56.7), CMS WebArbiter 27.6 ± 8.7 (42.9); pooled mini 31.6
  (38.5), WebArbiter 29.7 (49.2). A6 released sampling: WebArbiter pooled 26.3% (1 repeat).
- 2026-11-02 sensitivity test (job 413753, 1,200 logged WebArbiter calls, 600 over 8,192 tokens; 18% of all step-1
  WebArbiter calls exceed the training cap): a greedy re-query reproduces the logged verdict 92.7% of the time (batching
  nondeterminism); removing Prior Knowledge + State Difference flips 12.0% vs the re-query, cutting the AXTree to 8k
  flips 12.1%, and flips are no more frequent over 8k (11.7%) than under (12.5%). Length beyond training is not the
  cause; the extra sections move ~5% of verdicts beyond the noise floor (and lower the Response-2 rate 69% -> 63%).
- 2026-11-02 viewport / tree-format check: our viewport-only trees are not smaller than WebPRMBench's (median 169 vs 83
  lines, footer present 57% vs 44%); the difference is format (WebArena `key: value`, link URLs, RootWebArea header,
  pruned list wrappers vs BrowserGym). C4 (job 413862: WebPRMBench WebArena trees rewritten to WebArena style, no URLs)
  vs C2 in the same job: 87.06 vs 88.93 pairwise, 66.67 vs 69.15 BoN; paired 33 pairs right only in C2 vs 18 only in C4
  (sign test p ~ 0.05). Tree format costs ~2 points of judge accuracy: real but small.
- 2026-11-03 (user approved items 1 and 3): reference-judge test `ref_judge.py` (job 414435: 400 logged step-1
  comparisons, 200 from each method's runs; gpt-4.1 reference + 100 swapped, gpt-4o-mini, WebArbiter released prompt,
  WebArbiter A11 prompt; ~$9). A11 end to end (jobs 414436–414439, tag A11-r1, WA_ARBITER_FORMAT=browsergym):
  `wa_protocol.to_browsergym` rewrites only WebArbiter's judge prompt to the WebPRMBench layout — Prior Knowledge and
  State Difference dropped, BrowserGym AXTree attributes (no link URLs, tab header or RootWebArea; list wrappers cannot
  be restored from text), JSON trajectory, BrowserGym actions (`type [id] 'x'` -> fill('id', 'x')). run_cells.sbatch
  gained POLICY (for the GPT-4o-mini-policy rows of Table 5) and WA_ARBITER_FORMAT.
- 2026-11-03 online check: no GitHub issues on YaoZ720/WebArbiterCode (8 stars, single commit 764ec26, 2026-04-10) and
  no HF discussions; our checkpoint is ZYao720/WebArbiter-7B @ 6518479b (latest). The model card's Quick Start is
  greedy, max_new_tokens 2048, a single user message in the WebPRMBench format. The promised dataset
  ZYao720/WebArbiter-Trajectories ("72 successful trajectories" = exactly the GPT-4o + WebArbiter successes of Table 5:
  20+15+10+17+10) is still "Coming Soon". The project page's Guided Search Explorer ships 5 example episodes
  (static/js/explorer-data.js; copy in webarena/authors-demo/): GPT-4o-mini policy + WebArbiter, docker-hosted sites
  (host.docker.internal:8023), 5 candidates and 4 knockout comparisons per step, bracket actions, 3–5 steps each.
  **Their policy's candidates include `send_msg_to_user [answer]`** (Map step 2 and Reddit step 4 won the tournament;
  CMS step 2 lost), while the released `sample_p.py` has no stop action and in our 6,479 GPT-4o policy replies (and the
  first 52 GPT-4o-mini ones) the policy never proposed one. So the released prompt is not the one behind Table 5, and
  in our runs only the detector can end an episode: WebArbiter episodes end on a detector stop 61% of the time vs 70%
  for mini-PRM, time out at 30 steps 38% vs 29%, and succeed equally often when stopped (39.5% vs 40.7%). The demo's
  GitLab step 4 has no candidates (a detector short-circuit), so their harness also had a detector beyond step 0.
- 2026-11-03 released-code facts: no special-casing of the WebArbiter model name anywhere; rerank defaults to
  gpt-4o-mini; the knockout pairs candidates in confidence order from a deque (bye candidate meets a round-1 winner),
  which is why both judges pick "Response 2" ~68% of the time; verdict parse default is candidate A; released
  PolicyModel literal is "gpt-4o" (alias → gpt-4o-2024-08-06 since 2024-10, so A8 is moot; 2024-05-13 is retired).
  Table 5 is identical in arXiv v1 (2026-01-29, no Map column) and v2.
- 2026-11-03 action-space comparison (user question): Web-Shepherd (BrowserGym `chat`+`tab`+`nav`+`bid`+`infeas`
  subsets, 15 actions) vs the released WebArbiter policy prompt (11 actions). Shared: click, hover, type/fill, press,
  scroll, new_tab, tab_focus, close_tab, go_back, go_forward, goto. Only in Web-Shepherd: send_msg_to_user,
  report_infeasible, select_option, noop. Semantics differ for type/fill (WebArena `type` always presses Enter and types
  the content verbatim; BrowserGym `fill` does not press Enter) and scroll (direction vs pixel delta). The quoted-content
  defect (A13) was found while checking this: executed key ids for `type [11597] '12/31/26'` include the quote key at
  both ends.
- 2026-11-03 reference-judge test (job 414435, $8.37, 400 logged step-1 comparisons, 200 from each judge's runs):
  agreement with gpt-4.1 (greedy, released prompt; position-consistent on 90% of 100 swapped re-asks): gpt-4o-mini
  60.2%, WebArbiter 56.9%, WebArbiter on the A11 BrowserGym-format prompt 56.0%. On the 109 comparisons where mini and
  WebArbiter disagree, gpt-4.1 sides with mini 61 times and WebArbiter 48 (two-sided binomial p = 0.25). gpt-4.1's
  "Response 2" rate is 52% vs ~70% for both tournament judges. Reading: on our actual tournament pairs WebArbiter is at
  best no better a judge than gpt-4o-mini, and the input format (A11) does not change that.
- 2026-11-03 A13 motivation (step-1 episodes by executed `type` actions, success %): GitLab arbiter no-type 31.8 /
  quoted 22.0 / unquoted 48.1; GitLab mini 72.0 / 16.3 / 45.5; CMS arbiter 34.6 / 11.8 / 50.0; CMS mini 21.4 / 14.3 /
  50.0. 184 of 388 episodes executed a quoted type (literal quote characters typed, then Enter). Same bug for both
  judges, so it does not bear on WebArbiter vs mini-PRM, but it depresses every absolute number. A13 alone launched
  (both judges, GitLab + CMS, 1 repeat, tag A13-r1) so that the A12 bundle's effect can be separated from it.
- 2026-11-03 A12 'full' smoke (job 414513, CMS 0/27/114, WebArbiter, 19 min): the policy proposed send_msg_to_user
  (9 of 25 policy replies), noop (4), clear (4); no report_infeasible or select_option in these three; a `clear` won and
  executed; two episodes ended on a policy-proposed answer (task 0 at step 1, a premature wrong answer the judge
  accepted; task 114 at step 8); task 27 ended on a policy answer at step 16. 0/3 successes (the two N/A-reference
  tasks got concrete answers). Bundle launched: tag A12full-r1, WA_POLICY_STOP=full + WA_TYPE_UNQUOTE=1, both judges,
  GitLab + CMS, 1 repeat (8 jobs).
- 2026-11-03 A11 final (jobs 414436–414439, 63/65 valid, $23.5): WebArbiter on its training-format prompt — GitLab 26.7,
  CMS 24.2, pooled 25.4 vs step 1 29.7 (paired on the same tasks: −2.8, CI [−10.7, +5.1] at 59 tasks). Null; the two
  CMS page-load timeouts (30, 47) are not rerun. Format is closed as an explanation (with WebPRMBench C1–C4 and the
  reference-judge test).
- 2026-11-03 failure inspection (user request). 31 of 65 GitLab+CMS tasks (12 GitLab, 19 CMS) were never solved in any
  configuration across ~1,000 episodes (both judges, no search, A6/A11/A12/A13, both policies); 18 GitLab and 16 CMS
  tasks are ever solved, so the paper's WebArbiter rates (17/30, 15/35 per run) sit at or above our solvable ceiling.
  The `'EvaluatorComb' object has no attribute 'url_service'` text in verdicts is the released evaluator's debug-info
  path only; scores are real (task 105 passes with the same text). Never-solved state-change tasks: the agent reaches
  the right form, edits it, and declares completion without saving (task 100: all 17 episodes typed "Enjoying life" in
  the profile status field and none clicked a save/Set-status button; the authors' demo episode clicked it); 38–54% of
  those episodes end on a confident wrong "done", the rest time out. Never-solved info tasks: wrong scope (task 0
  reads the lifetime dashboard widget instead of the Jan-2023 report; task 2 answers 351 = unfiltered total) or the
  URL-match target page is never reached (task 25); 73–80% end with a confident wrong answer. Repeated identical
  actions: 9–14% of steps; 18–29% of episodes contain a run of ≥3, and those succeed at ~12% vs 35–45% otherwise.
- 2026-11-03 (user asked whether the appendix gives the prompt): the paper has no policy prompt (Appendix D holds only the
  judge prompt; §5.2.1 says N = 5, knockout tournament, GPT-4o-mini and GPT-4o, nothing else). Appendix J analyses two
  failure cases on our tasks 9 and 100 in bracket syntax with large bids. All 28,563 training prompts use a system text
  without the five-criteria block ("...based on the judging criteria... you need to refer to determine the better
  response") and have no Prior Knowledge / State Difference; WebPRMBench prompts and the released tournament prompt do
  carry the criteria block. A14 tests the training prompt: offline (training_prompt_check.py, WebPRMBench WebArena pairs
  + the 400 reference-judge comparisons, no API) and end to end (tag A14-r1, WebArbiter, GitLab + CMS, ~$20).
- 2026-11-03 external data point: SURFPRM (arXiv 2609.32351, Sept 2026, different group) ran reward-guided search on
  WebArena-Lite with a Qwen3.5-9B judge trained on WebArbiter's 10k SFT data (not the released 7B): GPT-4o policy GitLab
  44.12 (15/34), CMS 30.56 (11/36); GPT-4o-mini policy GitLab 38.24, CMS 38.89. Their no-search / mini-PRM / WebShepherd
  rows are copied from the WebArbiter paper; their task counts (34/36/24/46/31) differ from WebArbiter's (30/35/19/45/26),
  so it is a different harness. Their diagnosis of WebArbiter's data: few contrastive pairs where both candidates
  target real on-page elements with the same action type (24%), so the model learns shortcuts — consistent with our
  reference-judge result (no edge over gpt-4o-mini on close calls). Their appendix gives no harness details either.
  Checked a pasted suggestion list: "feed WebArbiter-Trajectories as candidates" is not actionable (the dataset is
  empty and holds outputs, not inputs; search candidates are generated live); "missing principle-induction prompts" is
  false here (2,303 of 2,303 logged WebArbiter replies contain <State> and <Criteria> with weights).
- 2026-11-03 demo-vs-ours comparison (user request), task 100 with the GPT-4o-mini policy: both reach the status dialog
  by the same route (avatar menu -> Set status -> type "Enjoying life"). At the next step the authors' pool has five
  navigation candidates, WebArbiter picks `click [Set status]`, and their detector fires one step later; in all three
  of our episodes the pool holds the same save click (`click [2854]`, policy confidence 0.95) plus the detector's stop
  (conf 1.0, "Your GitLab status has been set"), and WebArbiter picks the stop. The premature-completion failure is
  therefore the D14 stop regime (detector stop competing, detector over-eager) at least as much as the judge. Task 21:
  all our episodes (both policies) succeed in 4 steps by the same route as the demo (3 steps + detector). A15 launched:
  WebArbiter, GitLab + CMS, 1 repeat (~$20), no detector, full action space, quote fix.
- 2026-11-03 A14 offline (job 415033, 7 min, no API): WebPRMBench WebArena pairs — released-format prompt 88.06 / 72.14
  vs training-format 89.05 / 71.14 (44 of 804 verdicts flip). Logged tournament comparisons (n = 400): agreement with
  gpt-4.1 released 56.9%, training-format 51.0%, gpt-4o-mini 60.2%; 182 of 400 verdicts flip between the two formats.
  The paper's prompt does not help and is worse on our close calls; the end-to-end A14 jobs (415038–415041) were
  cancelled before spending. The 45% flip rate between two near-equivalent prompts says WebArbiter's verdicts on
  tournament pairs are close to arbitrary, matching the reference-judge result.
- 2026-11-03 (user: "the released code BUT with the complete action set"): A16 launched, WebArbiter, GitLab + CMS, 1
  repeat (~$20). Differs from A15 only by the step-0 detector check and the absence of the quote fix. Terminal actions
  in the released code are exactly parsed `SEND_MESSAGE_TO_USER` (step_management.py:505), reached via the detector,
  the 30-step `N/A` fallback, the no-candidates failure terminal, or (now) a policy candidate.
- 2026-11-03 grader discrepancy: the released `StringEvaluator.clean_answer` lacks the leading `answer.strip()` of the
  original WebArena evaluator (web-arena-x/webarena main), so `exact_match` against page text with surrounding
  whitespace fails. Task 100's check (`.cover-status` lastChild textContent == "Enjoying life") is the suspected
  victim: 5 of 19 episodes (A15, step-1 r2, mini-PRM r3, mini-PRM A13, no-search) ended with the header showing the
  saved status "💬 Enjoying life" and all scored 0. GitLab renders `.cover-status` as emoji_icon followed by the
  markdown message (users/show.html.haml:77). Other exposed checks: textContent exact_match on tasks 96, 99, 105, 110,
  143, 163 (105/110 pass, so no stray whitespace there); no string-answer exact_match task differs by whitespace only.
  wa_episode.evaluate now restores the original strip (cells graded before this note used the released version); a
  live probe of what the grader reads for task 100 is running (probe_task100.py).
- 2026-11-03 released grader vs original WebArena (function-level diff of evaluators.py / helper_functions.py):
  (1) `clean_answer` lost `strip()`; (2) `must_include` now tokenizes whenever the reference is one word (original
  tokenizes only one-character references) and also splits tokens on '/'; (3) the URL evaluator's default rule is
  "EXACT" (original: "GOLD in PRED") and its "GOLD in PRED" is a raw substring test (original: base path + query
  parameters as sets); (4) fuzzy-match prompt wording ("Only output one of these options") and model; (5) added
  must_exclude / compare_inequality / str_2_int. Offline re-grade of the 168 step-1 string/URL cells (fuzzy and
  program_html excluded): 164 unchanged, 4 flip — all CMS task 32 (must_include single-word tokenization), where the
  original rule passes 5/6 cells and the released 1/6. The URL approximation found no flips. Demo tasks under the new
  regimes: A15 task 100 saved the status (header shows it) and still scored 0 (strip issue, probe pending); A16 task
  100 looped on the avatar menu and ended on a premature policy answer; both solve task 21.
- 2026-11-03 probe (job 415321, probe_task100.py on a live GitLab copy): after setting the status through the dialog,
  `document.querySelector('.cover-status').lastChild.textContent` is "\nEnjoying life\n" (HAML newlines around the
  markdown message). Released grader: exact_match False; original WebArena grader (strip): True. Task 100 was therefore
  unpassable under the released evaluator; the 5 episodes that saved the status (A15, step-1 arbiter r2, mini-PRM r3,
  mini-PRM A13, no-search) were real successes. The authors' demo passes task 100, so Table 5 was graded with a
  stripping evaluator (WebArena's), not the released copy — the evaluator joins the policy prompt and the stop regime
  as released-vs-paper differences. Grader now restores original clean_answer and must_include; page-text
  exact-match tasks 96, 99, 100, 143, 163 and must_include task 32 are rerun for the step-1 rows (both judges, r1–r3)
  and the GPT-4o-mini-policy rows; other rows keep released-grader cells for these six tasks (noted when reported).
- 2026-11-03 GPT-4o-mini-policy rows final (jobs in minipolicy_jobs.txt; released grader; 3 repeats except WebArbiter
  GitLab r3 second half, 15 cells lost to job 414459 whose SGLang server hung 5.4 h on g008 — resubmission refused by
  the $1,000/user budget): mini-PRM GitLab 21.1 ± 3.2 (paper 33.3), CMS 24.8 ± 2.2 (22.9); WebArbiter GitLab
  18.7 ± 3.1 (46.7), CMS 18.3 ± 2.0 (42.9). WebArbiter again below mini-PRM (pooled ~19 vs ~23). 15 cells invalid
  (page-load timeouts), excluded. Six grader-exposed tasks queued for rerun under the corrected grader.
- 2026-11-03 budget: sbatch refused with "Budget for user piotrt on account zixianma: $1000.00"; Slurm billing since
  Oct 1 ≈ $973, Nov 1–3 ≈ $325 (urgent counted ×2 = $261). Billing is max(TRES): 8-CPU MIG jobs bill like a full
  H200. run_cells.sbatch now times out a server that is not healthy within 25 min.

- 2026-11-03 (user: "set up A16, but generate actions like Web-Shepherd"): A17 = A16 + Web-Shepherd candidate generation.
  Web-Shepherd (arXiv 2505.15277) App. F.2: "we sample 20 output sequences using nucleus sampling with a temperature of
  1.0. The top-n most frequent actions across these samples are selected as candidates. We then score each candidate
  action using the reward model and select the one with the highest predicted reward. In cases where multiple actions
  receive the same score, we execute the action that was sampled more frequently"; Sec. 6.2: "Among 5 action candidates
  sampled from the policy, the action that is assigned the highest reward is executed" (n = 5; the nucleus p is not
  stated, so the released top-p 0.9 stays). Implementation: `wa_episode.shepherd_sample` makes one chat call with
  `n=20` (the ~6k-token observation is billed once) on the one-BLOCK prompt (no-search envelope v2 + A12 full action
  block); `wa_protocol.shepherd_candidates` groups the 20 actions by exact string and writes the 5 most frequent as a
  released-format BLOCKS reply (confidence = frequency/20, `frequency` set, first sample's thought), so the released
  parser, element filter, dedup, step-0 detector and knockout tournament run unchanged. Offline check on a logged A16
  prompt (GitLab task 44, step 3; `$CLAUDE_JOB_DIR/tmp/test_shepherd.py`): 8.1 s (inside the released 60 s sampling
  timeout), 20/20 replies parsed, $0.031 for the call (3,071 prompt + 2,323 completion tokens; ~1.5× the 5-BLOCK
  call), and **all 20 samples were the same action** (`click [6527]`, stated confidence 0.95), where the released
  5-BLOCK prompt had produced 5 distinct actions (`click [6527]`, `type [6447] 'better'`, `click [6492]`, `click [6524]`,
  `click [6195]`). Expect many A17 steps to reach the tournament with 1–2 entrants: frequency sampling measures the
  policy's own spread, and the "5 diverse, PARALLEL" prompt forces diversity that T 1.0 sampling does not have. Still
  not Web-Shepherd: knockout tournament instead of independent scores. Launch: GitLab + CMS, WebArbiter, 1 repeat,
  the A16 halves (`jobs/submit_A17.sh`); blocked by the $1,000 budget at the time of writing.
- 2026-11-03 14:41 A17 launched (jobs 415482 415483 415484 415485, normal QoS, pending on priority; `jobs/submit_A17.sh`,
  idempotent). Budget events on the way there (`hyakusage`: piotrt 914 / 1,000 USD, TYPE total, i.e. an all-time cap;
  account zixianma 12,083 / 14,000 total): the 10 held ARM-scaling jobs (394365–394374, $90 of requests, submit lines
  saved in logs/arm_scaling_held_jobs_2026-11-03.txt) were cancelled with the user's agreement but freed nothing (they
  predate the plugin's tracking); reducing TimeLimits of running/pending jobs freed nothing (the plugin keeps submit-time
  estimates until a job ends); the 12 pending mini-PRM grader re-runs (415338–415357, 3 h / 1.5 h requests) were
  cancelled and are resubmitted by `jobs/resubmit_mini_regrade.sh` with 2 h / 45 min limits as room appears (retry
  loop every 5 min). Room appeared once the six CMS grader re-runs and the WebVoyager base control ended below their
  estimates. Paired ablation comparisons now come from `wa_paired.py` (bootstrap over tasks, 2,000 resamples).
- 2026-11-03 15:02 A17 stopped at 9 cells (1 pass; 3 of 9 ended at step 0, 2 by the detector) on the user's instruction after
  the step-0 audit: the detector fires at step 0 in ~1 episode in 10 under every regime (19/195 step-1 WebArbiter cells,
  16/193 mini-PRM, 29/358 GPT-4o-mini policy, 8/50 A16); a step-0 stop almost never passes (3 of 65 overall). With the
  stop in the pool (D14) the judges reject about half the fires (8/19, 9/16 end the episode; ~1.5 points of solvable
  tasks lost); under the released override every fire ends the episode: A16 9/52 cells (8 detector + 1 policy stop),
  2 passed, 4 of the 7 failures (GitLab 9, 39, 62; CMS 40 — "go to page X" URL-match tasks whose landing page shows a
  count or link) are solved by 4–11 other runs, ~8 points of A16's ceiling and about its deficit to step 1 (−6). A17's
  sampler worked as designed (20/20 parsed, 8–14 s, ~$0.04 per step); 12 of its first 24 steps had a single distinct
  action, 8 had five. A18 = A17 with `WA_DETECTOR=off` (`jobs/submit_A18.sh`), placed by a retry loop as budget frees.
- 2026-11-03 15:35 CORRECTION (A16/A17 "step-0 only" was wrong for this harness). The released post-action stage
  (`StepManagementMixin._compute_post_action_async`, step_management.py:381) runs `detect_terminal` after EVERY executed
  action; its stop becomes the next step's `previous_step_terminal_action`, and the released
  `_detect_terminal_if_needed` (`previous_step_terminal_action or current_step == 0`) then re-runs the detector and, under
  WA_DETECTOR=released, short-circuits the tournament. So the released regime is "detector at step 0 and again after any
  step whose post-action check proposed a stop". Measured: A16 34 of 54 valid cells ended on a detector stop without a
  tournament (8 at step 1, 26 later, spread over steps 2–27; 11 passed, 23 failed); only 5 ended via the tournament.
  A17 5 of 9. Under WA_DETECTOR=off (A15, A18) the post-action detector still runs (A15: 1,026 calls in 54 cells, cost
  only) and is ignored, except one leak: when a step yields no selected action, the native
  `TerminalActionHandler.extract_action(fallback_action=previous_terminal)` executes the post-action stop — 3 of 48 A15
  cells (all failed); step 1 (D14) r1 6 of 75. Not fixed mid-run (A18 is running on this code); a fix would drop the
  fallback when WA_DETECTOR=off. A18 first cell: GitLab task 9 passed in 2 steps (click To-Do, then stop chosen by the
  tournament) — the task the step-0 detector killed in A16/A17.
- 2026-11-03 16:35 A19 (user): A18 with 5 independent single-action samples instead of 20 (`WA_SAMPLES=5`, one n=5 call
  at T 1.0 / top-p 0.9), every distinct action kept (exact-string dedup, as the released sampler does), so 1–5 tournament
  entrants; detector off, full action set, no quote fix, greedy judge. Charged to account vlm (zixianma per-user budget
  exhausted: 965 / 1,000). Jobs in jobs/ablation_A19_jobs.txt; g001 excluded (A18 ran ~5x slower there).
- 2026-11-03 17:35 step-1 and P4m reference rows final under the corrected grader (original-WebArena clean_answer strip and
  must_include rule; tasks 96 99 100 143 163 and 32 re-run for both judges, r1–r3). GPT-4o policy: WebArbiter GitLab
  35.6 ± 3.8 / CMS 28.6 ± 8.6, mini-PRM 40.0 ± 3.3 / 28.2 ± 1.5 (paper 56.7 / 42.9 and 40.0 / 37.1); paired over 65
  tasks WebArbiter − mini-PRM −1.8 [−9, +5]. GPT-4o-mini policy: WebArbiter 23.5 ± 6.5 / 19.3 ± 3.7, mini-PRM
  24.4 ± 3.1 / 25.8 ± 3.9; paired −4.2 [−10, +1]. A12 and A13 final (both judges) and A15 / A16 final: none beats step 1.
- 2026-11-03 18:30 A18 CMS failure audit (21 failed valid cells): 10 stopped with a wrong answer / unfinished state, 6 gave up
  (report_infeasible), 5 ran 30 steps. Only 4 of the 21 tasks are ever solved by step 1 (27, 30, 48, 50). In 9 failures
  the final step's winner was a stop/give-up proposed by 1 of 20 samples while a continue action had 5-16 (116 click x16 vs
  four reworded give-ups; 114 submit x15 vs three "sent" claims; 145 apply-filter x5 vs four "displayed" claims): reworded
  stops each take their own top-5 slot and WebArbiter favours stops. A20 merges them into one candidate (jobs/submit_A20.sh).
  A18 GitLab final: 18/30 = 60.0% (step-1 mini-PRM same tasks 40.0, +20.0 [+8, +34]; paper 56.7 / 40.0).
- 2026-11-03 19:30 A18 final on GitLab + CMS (65/65; CMS 88 re-run valid): GitLab 18/30 = 60.0 (step-1 mini-PRM same tasks
  40.0, +20.0 [+8, +34]; paper 56.7 / 40.0), CMS 11/35 = 31.4 (+3.3 [−8, +14] vs mini-PRM; paper 42.9 / 37.1), both
  29/65 = 44.6 (+12.8 [+4, +22] vs step-1 WebArbiter, +11.0 [+2, +20] vs step-1 mini-PRM). A19 (5 samples) final 24/64 =
  37.5, −7.8 [−19, +2] vs A18 (GitLab −20.0, CMS +2.9). Extension (user): A18 and A20 on Reddit (19 tasks, jobs 416211,
  416212) and Shopping (45 tasks, 416348–416351), account vlm, compared with the paper only (no step-1 rows on these
  sites, per the user). Shopping hosting fixed: the pristine base is packed once into sandbox/shopping.tar.zst
  (pack_site.sbatch, 43 GB, 24 min) and unpacked to node-local disk per job (139 s); per-task copies take ~2 min instead
  of 2,239 s (job 410928). The earlier Shopping "login_ok = false" probe was a false negative (Magento renders Sign Out
  client-side; order history, which needs a session, loaded).
- 2026-11-04 07:30 S18 (user): A18's candidates (20 GPT-4o samples, top-5, detector off, policy stops) with Piotr's
  SelectionARM (arm/selection/81b452d on OpenWebRL-4B-SFT, scripts/serve_arm.py) as the selector instead of the WebArbiter
  tournament (wa_episode.py --method selectionarm, wa_selectionarm.py). The ARM sees the 1280x720 actor screenshot and
  each candidate rendered as the SFT actor's own tool call, element ids grounded to point_2d = bbox centre / viewport x
  1000 (off-screen targets kept with their off-grid point, user); history = executed candidates rendered the same way.
  Matched control A18v720 = A18 with WebArbiter at the same 1280x720 viewport (A18 used 1024x2490). GitLab + CMS first,
  account vlm (jobs/selectionarm_jobs.txt). Smoke 418063: GitLab 9 passed (3 steps), 25 failed (18 steps).
