# Baseline experiments (Tillicum, 2026-11-02 to 11-06)

Every number below is recomputed from the raw per-task results by `webarena_repro/baseline_tables.py`
(run it to regenerate). Success is valid-only (successes / valid episodes) unless marked *overall*
(invalid episodes count as failures). Paired differences use tasks valid in both rows; 95% percentile
intervals from 4,000 task resamples (site-stratified on WebArena-Lite). All rows are a single run unless
stated. "Our ADM" is Piotr's SelectionARM (`PTeterwak/OpenWebRL-4B-SelectionARM` @ 81b452d, served by
`scripts/serve_arm.py --mode selection`).

## Summary

| Question | Answer |
|---|---|
| Does our WebArbiter reproduction match the paper? | Yes: A18 gives a 48.3 four-site mean (no Map) against the published 49.2. |
| Our ADM vs WebArbiter-7B with GPT-4o on WebArena-Lite, same viewport? | Ahead, not significant: +6.6 [−2.0, +15.0] at 1280×720, +4.6 [−2.8, +12.0] at the paper's 1024×2490. |
| Our ADM vs WebArbiter-7B with GPT-4o on Online-Mind2Web? | Ahead, significant: +12.2 [+7.2, +17.6] (valid-only), +9.7 [+5.0, +15.0] over all 300 tasks. WebArbiter gives no gain over GPT-4o alone. |
| External selectors on our SFT actor? | Best-of-Q and PRO-CUA give nothing. WebDreamer (n = 10) is level with our ADM overall (44.1 vs 45.7), significantly behind on OM2W, at about 3.4× a single-pass selector's GPU time. |

## 1. WebArena-Lite, GPT-4o actor (WebArbiter Table 5 setting)

**Harness.** The released WebArbiter run path (`webarena_repro/wa_episode.py` over Zixian's
`vendor/native_episode.py`): original WebArena site images hosted per task with a fresh copy
(`wa_site.py`), the released policy prompt and parser, released knockout tournament for WebArbiter-7B,
WebArena evaluators (fuzzy judge gpt-4.1, the retired gpt-4-1106-preview being unavailable). 129 tasks on
Shopping 45, CMS 35, Reddit 19, GitLab 30; Map is not hosted. Decisions and every ablation (A1–A20) are in
`webarena_repro/PROTOCOL.md`.

**A18** (the setting that reproduces the paper, used for every row below): Web-Shepherd candidate
generation (20 single-action samples in one call, T 1.0, top-p 0.9, five most frequent kept,
`WA_SAMPLING=shepherd`); full action space including `send_msg_to_user` / `report_infeasible`
(`WA_POLICY_STOP=full`); no terminal detector (`WA_DETECTOR=off`). The no-selection row executes one sample
per step with the same prompt and action space.

**Our ADM in this harness** (`wa_selectionarm.py`): each released candidate (an element-id action) is rendered
as the SFT actor's tool call, with `point_2d` = the element's bounding-box centre / viewport × 1000, and the
ARM picks from the screenshot. GPT-4o and WebArbiter read only the accessibility tree (current viewport only);
our ADM reads only the screenshot.

| Row | Shopping | CMS | Reddit | GitLab | Mean |
|---|---|---|---|---|---|
| GPT-4o, no selection, 1280×720 | 18.2 | 23.5 | 31.6 | 16.7 | 22.5 |
| + WebArbiter-7B, 1280×720 | 29.5 | 31.2 | 57.9 | 36.7 | 38.8 |
| + our ADM, 1280×720 | 37.2 | 39.4 | 57.9 | 50.0 | 46.1 |
| + WebArbiter-7B, 1024×2490 (paper / released service) | 38.6 | 31.4 | 63.2 | 60.0 | 48.3 |
| + our ADM, 1024×2490, ADM sees a 1024×576 window covering the most candidates | **54.5** | 38.2 | **66.7** | 56.7 | **54.0** |
| published WebArbiter-7B (GPT-4o) | | | | | 49.2 |

| Paired difference | Points |
|---|---|
| our ADM − WebArbiter, both 1280×720 | +6.6 [−2.0, +15.0] |
| our ADM − WebArbiter, both 1024×2490 (identical actor view) | +4.6 [−2.8, +12.0] (Shopping +15.9, CMS +5.9, Reddit 0, GitLab −3.3) |
| our ADM − no selection, 1280×720 | +24.1 [+16.2, +32.3] |
| WebArbiter − no selection, 1280×720 | +17.5 [+9.1, +26.3] |
| WebArbiter 1280×720 − WebArbiter 1024×2490 | −9.4 [−18.3, −0.8] |
| our ADM 1024×2490 crop − our ADM 1280×720 | +7.7 [−0.4, +16.0] |

**Viewport.** The accessibility tree only covers the viewport, so a taller page gives GPT-4o better
candidates; WebArbiter loses 9.4 points at 1280×720, mostly on GitLab and Shopping. WebArbiter's training
data (WebPRM Collection, Mind2Web) records no viewport and its trees are smaller (median 31 lines) than ours
at either size (85 at 1280×720, 117 at 1024×2490), so 1280×720 does not handicap WebArbiter as a judge.
Our ADM was trained on 1280×720 screenshots; GitLab + CMS variants: full tall 1024×2490 screenshot
(pixel cap raised to keep the training scale) 45.8, 1280×2490 with a centred 720 crop 47.9, the 1024×576
best-coverage crop 47.5, vs 44.7 at 1280×720 (all within noise; off-window candidates 0.5–7.4%).

## 2. Online-Mind2Web, GPT-4o actor, A18 harness, 1280×720

`webarena_repro/om2w_episode.py` runs the same A18 harness on live sites (any origin through the cluster
proxy, desktop user agent, CDP screenshots, DOM-ready navigation) and scores each terminal state once with
the OM2W/AgentTrek judge prompt on gpt-5.6-terra (final screenshot + executed actions). All 300 tasks per row;
tasks whose browser hung past the per-task limit were rerun once with a 20-minute limit, and a second hang is
recorded as invalid.

| Row | Tasks | Valid | Valid-only | Overall | Mean steps | OpenAI |
|---|---|---|---|---|---|---|
| GPT-4o, no selection | 300 | 291 | 23.7 | 23.0 | 9.6 | $38 |
| + WebArbiter-7B | 300 | 293 | 24.2 | 23.7 | 12.1 | $138 |
| + our ADM | 300 | 282 | **35.5** | **33.3** | 18.5 | $212 |

| Paired difference | Points |
|---|---|
| our ADM − WebArbiter | **+12.2 [+7.2, +17.6]** valid-only (278 tasks); **+9.7 [+5.0, +15.0]** overall (300) |
| our ADM − no selection | **+11.9 [+6.8, +16.9]** valid-only; **+10.3 [+5.3, +15.3]** overall |
| WebArbiter − no selection | +0.7 [−4.2, +5.6] valid-only; +0.7 [−4.0, +5.3] overall |

Consistent with Zixian's earlier OM2W rerun under her protocol (no search 26.9, SelectionARM 29.5, WebArbiter
24.1, WebShepherd 12.3). Cost per task in the 10-task cost test: no selection $0.19, WebArbiter $0.78, our ADM
$1.07 (longer episodes; the 20-sample GPT-4o call dominates; the judge is ~$0.005).

## 3. External selectors on our actor (OpenWebRL-4B-SFT, 5 candidates, T 1.0)

Same eval harness and judges as our ARM evals (`openwebrl.arm_eval --mode scalar`; DeepShop 150, OM2W 300,
WebVoyager 595). The base row is our single-sample run in the same days; the ADM row is Zixian's 3-run
SelectionARM batch of Nov 3–4 (`controls-sameday-20261103/piotr5`, pooled as in her tables), next to her
same-window SFT row (`sft1`). Cross-day comparisons carry live-site drift.

| Selector | OM2W | WebVoyager | DeepShop | Mean (valid-only) | Mean (overall) |
|---|---|---|---|---|---|
| SFT alone, Zixian's window (3 runs) | 34.6 | 45.6 | 43.1 | 41.1 | 39.6 |
| SFT alone, our single run (same days) | 36.8 | 46.7 | 48.0 | 43.8 | 41.7 |
| + Best-of-Q | 37.7 | 47.0 | 43.8 | 42.8 | 38.2 |
| + PRO-CUA PRM | 37.2 | 44.5 | 45.9 | 42.6 | 36.9 |
| + WebDreamer, 1 value sample | 37.6 | 54.0 | 45.7 | 45.8 | 40.7 |
| + WebDreamer, 10 value samples | 40.1 | 58.3 | 52.9 | 50.4 | 44.1 |
| **+ our ADM** (Zixian, 3 runs) | **43.8** | 52.4 | 50.5 | 48.9 | **45.7** |

Paired, all tasks (invalid = failure), WebDreamer n = 10 − our ADM: OM2W **−5.7 [−10.4, −0.9]**, WebVoyager
+1.8 [−1.8, +5.5], DeepShop −0.9 [−8.0, +6.4]. Against our same-day base actor, WebDreamer n = 10 is
+11.6 [+7.0, +15.9] on WebVoyager (valid-only) and within noise on the other two; Best-of-Q and PRO-CUA are
within noise everywhere (also at T 0.7: valid-only means 37.9 and 41.1 against a 38.9 base).

**Best-of-Q** (arXiv 2601.22701; `best_of_q/`): frozen Qwen2.5-VL-3B embeddings, MLP Q/V trained with IQL,
ε-greedy seed round then argmax-Q rounds, trained on Zixian's 1,367 branch states (one sampled continuation
outcome per candidate). Offline it fits training states (argmax-Q 44.9 vs uniform 39.2) but is at uniform on
held-out tasks (36.3 vs 35.9): scores track task identity (task-mean correlation 0.96 train, 0.51 held out).
The paper trains on the WebVoyager evaluation tasks themselves (WebVoyager has no train split; §4.2, App. A.2,
App. D), and its only unseen-task test (App. E.1) shows a smaller gain.

**PRO-CUA PRM** (`pro_cua/serve_procua_selector.py`): PRO-CUA's released PRM prompt on frozen
Qwen3-VL-4B-Instruct (the base our selector inherits from), screenshot annotated with their
`actions_visual`, score = P(is_correct = true).

**WebDreamer** (arXiv 2411.06559; `webdreamer/`): Dreamer-7B world model predicts each candidate's effect
(their prompt; action description = our tool call in words + the actor's own reasoning; one imagined step),
Qwen3-VL-4B-Instruct scores it with their value prompt (success 1 / on track 0.5 / failure 0, mean of n
samples). Invalid rates 9–21% vs 3–15% for our ADM, mostly page-load failures and aborts; 180 tasks that hit
the 30-minute limit were rerun with 90 minutes (16 still time out).

| Selector cost, three benchmarks | GPU-hours | Per step |
|---|---|---|
| PRO-CUA PRM | 7.5 | 5 greedy judgments |
| WebDreamer, 1 value sample | 9.5 | 5 world-model predictions + 5 judgments |
| WebDreamer, 10 value samples | 25.4 | 5 predictions + 50 judgments, ~5,300 generated tokens, 56 s median |
| our ADM | ≤ 7.5 (not measured in this harness) | 1 forward pass, a short JSON selection |

## Caveats

- Single seed per row except Zixian's ADM/SFT (3 runs); the WebArena ADM − WebArbiter differences are not
  significant. A second seed of both 1024×2490 rows costs about $15 GPU + $65 OpenAI each.
- WebArena-Lite excludes Map (26 tasks); the published 49.2 is the four-site mean over the same sites.
- Section 3's ADM row is from Nov 3–4, the selector rows from Nov 4–6; WebVoyager drifts a few points
  between windows. A same-day ADM run removes this (~$5 GPU + ~$33 judges).
- WebDreamer departs from the paper in its world-model inputs (descriptions from the actor's reasoning),
  imagination depth (1) and value model (Qwen3-VL-4B instead of GPT-4o).

## Spend (vlm account, GPU; OpenAI on Zixian's key)

GPU for this batch (Nov 5–6): OM2W ADM $52, OM2W WebArbiter $38, WebDreamer $34 (including two failed
launches), WebArena-Lite runs (no selection at 1280×720 and the ADM viewport variants) $42, OM2W no selection $6. Per-user total on vlm: $262.54 of
$1,000. OpenAI: OM2W $388, WebArena-Lite ADM 1024×2490 four sites $68, viewport variants ~$80, judges ~$33
per external-selector run.

## Where things are

| What | Path |
|---|---|
| WebArena / OM2W harness, adapters, analysis | `webarena_repro/` (`wa_episode.py`, `om2w_episode.py`, `wa_selectionarm.py`, `PROTOCOL.md`, `baseline_tables.py`) |
| Job scripts | `webarena_repro/jobs/run_cells.sbatch` (WebArena cells), `webarena_repro/jobs/om2w_cells.sbatch` (OM2W) |
| External selectors | `best_of_q/`, `pro_cua/`, `webdreamer/` |
| WebArena cells | `/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells/<site>-<method>-<tag>-<job>/task-*/summary.json` |
| OM2W cells | `/gpfs/scrubbed/piotrt/owrl/webarena/runs/om2w/<method>-<tag>-<job>/task-*/summary.json` |
| External-selector evals | `/gpfs/scrubbed/piotrt/owrl/eval/pass*/base+<selector>/<benchmark>/results/` |
| Job lists | `/gpfs/scrubbed/piotrt/owrl/webarena/jobs/{selectionarm,om2w}_jobs.txt` |

Tags: `A18` WebArbiter at 1024×2490, `A18v720` WebArbiter at 1280×720, `S18` our ADM at 1280×720,
`S18c1024` our ADM at 1024×2490 with the 1024×576 best-coverage window, `S18tall` / `S18crop` / `S18cover`
other ADM viewport variants, `N18v720` no selection. `/gpfs/scrubbed` is purged after 60 days without access.
