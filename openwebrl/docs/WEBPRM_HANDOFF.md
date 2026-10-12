# WebPRM Table 5 handoff

**Table 5 is incomplete. All current jobs are closed, and no supervisor is running.** This handoff records the independently audited state as of 2026-10-12. Read the current execution pointer before acting; this document is not new resource authorization.

The goal is to reconstruct **WebArbiter Table 5 and add Piotr’s released SelectionARM**, then evaluate Online-Mind2Web separately. The fixed comparison contains **155 single-site tasks × 2 actors × 5 selectors = 1,550 episodes**. Exact historical Table 5 membership/settings remain unresolved, so describe this as a **disclosed reconstruction**, not an exact reproduction.

Start with [results and methodology](WEBPRM_BASELINES.md#table5-plan-20261010), the repository [working agreement](../../AGENTS.md), and these runtime records:

```text
R = /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011
N = R/next-stage-preparation
C = N/reddit-table5-190-v3/campaign

R/current_table5_preparation.json
N/reddit-terminal-independent-root-356127-v1/final-review.json
```

`R`, `N`, and `C` above are path abbreviations used throughout this document.

## Verified results

Only the **190-cell Reddit subset** has execution approval. All 12 attempted cells are independently audited; none were resampled.

| Disposition | Attempts | Raw positive verdicts | Guarded successes |
| --- | ---: | ---: | ---: |
| Valid, task 5 | 6 | 5 | 0 |
| Valid, task 14 | 2 | 2 | 2 |
| Wrapper-invalid | 2 | — | — |
| Interrupted, independently audited | 2 | — | — |
| Total | 12 | 7 | 2 |

The task 14 successes are **GPT-4o-mini + Piotr SelectionARM** and **GPT-4o + no-search**. Five task 5 raw positives incorrectly accepted `N/A`; the explicit scoring guard records zero while preserving the original verdicts. Interrupted attempts retain **null native/effective verdicts**.

**178 Reddit cells remain unclaimed.** The other **1,360 cells** still need validated environments and execution budgets. Do not publish selector rankings or cohort success percentages from this partial coverage.

The authoritative collected snapshot is:

```text
N/reddit-terminal-independent-root-356127-v1/activated-campaign-snapshot.json
SHA256: 26c966caa70f06702583d17cbe10925e1624a56c7de73cc5bab0df9207b71bdd
```

## Immediate next actions

1. Read the current pointer and reviewed repair packages; retain `verified_complete=false` until the requested endpoint is independently verified.
2. Prepare a successor Reddit package incorporating the queue repair and an explicit allocation-slot exception request, preserving every cumulative cap and existing claim. **Do not relaunch frozen `reddit-table5-190-v7` unchanged.** A successor launch package incorporating the latest repair has not been prepared.
3. Obtain exact authorization before another submission; likewise resolve the four pending environment requests below. Parallelize independent authorized work.
4. Restart and verify active supervision for any newly approved jobs. The previous watcher exited after all registered work closed; its recorded PID is historical.
5. Complete Reddit and environment validation, then obtain execution budgets for the remaining sites. Keep historical Online-Mind2Web diagnostics separate from the new comparison.

**The earlier “I approve” authorized bundle v2, not pending bundle v4.** Repeated supervision callbacks are stale deliveries and do not grant new authorization. Leftover time or API budget does not authorize another allocation slot.

<details>
<summary>Failures, repairs and validation evidence</summary>

| Job | Scheduler seconds used | Outcome |
| --- | ---: | --- |
| 356022 | 2,614 | Stopped after detecting native terminal-detector wrapper bugs; 10 immutable primary claims preserved |
| 356094 | 47 | Startup failed when a storage scan encountered a disappearing database-build file; no new model calls or primary claims |
| 356127 | 305 | Completed two successful trials, then prematurely stopped admitting work |

Repairs are implemented and tested:

- Native terminal handling follows the pinned parser’s fallback/nonterminal-action behavior: **78 focused tests**.
- Storage scanning retries narrowly identified file-disappearance races, accepts only complete scans, and verifies final measurements after closure: **35 focused tests**.
- The latest queue fix waits for an active worker to close before rechecking a denied resource admission. It also records admission reasons and checks minimum startup capacity: **52 tests plus 7 independent frozen-source checks**.
- Additive interruption audits and collector support passed **58 tests**, with independent byte-identical replays. No primary evidence or verdict was rewritten, and no model call or state disposal was performed by audit activation.

**The queue fix has not been validated live.** Frozen v7 still contains the premature-exit behavior. The precise resource check that rejected admission in 356127 was not logged; storage pressure is plausible, not proven.

Reviewed repair and closure evidence:

```text
N/reddit-admission-drain-repair-v1/readiness.json
N/reddit-admission-drain-repair-v1/root-review.json
N/reddit-stop-audit-356127-v1/audit.json
N/reddit-terminal-independent-root-356127-v1/two-primary-audits.json
N/reddit-terminal-independent-root-356127-v1/closure-audit-review.json
N/reddit-interruption-audit-v3/readiness.json
N/reddit-terminal-independent-root-356127-v1/interruption-activation.json
```

Queue-repair readiness SHA256: `ac61509ed9d52c02fccb74e9f794325b2efa2366bbf8453909baa83eb2a8df91`.

Independent queue-repair review SHA256: `b93620beace6db15c63187be32922dce1f55181b63d7b339f27f25f4b00b26ca`.

Latest terminal/accounting audit SHA256: `1dff895e4e319d6ad17851d3958bbc859ba89d865aa852c5ecd2401faa816587`.

The queue/controller implementation is in `openwebrl/webprm_campaign_queue.py` and `scripts/launch_webprm_table5_campaign.py`. Reporting uses `scripts/collect_webprm_table5_campaign.py`; interruption auditing uses `scripts/audit_webprm_interrupted_cell.py`. Prefer reviewed frozen source packages over assuming every working-tree change belongs to this task.

</details>

<details>
<summary>Reddit resource authority and cumulative accounting</summary>

| Resource | Original cap | Consumed |
| --- | ---: | ---: |
| Allocation attempts | 3 | **3** |
| Cumulative allocation time | 86,400 s | 2,966 s |
| API spending | USD 250 | USD 4.90736970 |
| Paid dispatches | 21,524 | 707 |
| Local dispatches | 11,471 | 489 |
| Total dispatches | 32,995 | 1,196 |
| Cumulative apparent scratch | 549,755,813,888 bytes | 384,763,249,194 bytes |
| Durable campaign artifacts | 68,719,476,736 bytes | 173,209,452 bytes at final job closure |

The actual allocation profile was **1 H200, 8 CPUs, 240 GiB RAM**, with at most two workers. The original request allowed 16 CPUs; the approved operational reduction used 8 because of scheduler constraints. There are **83,434 unused seconds but zero remaining allocation slots**. A fourth allocation requires an explicit exception; neither the standing recovery instruction nor unused spending grants that exception.

All failed attempts, startup controls and the historical unfinished local dispatch remain charged. The financial epoch is closed, with no unresolved paid reservations or overruns. Preserve the shared campaign, journal, queue claims, source versions and all original verdicts. Do not reset a budget on retry or resample a claimed primary cell.

Original authority and financial evidence:

```text
N/table5-next-jobs-bundle-v2.json
N/table5-next-jobs-bundle-v2-approval.json
C/allocation-history/
C/shared-budget.json
C/shared-budget.json.events.jsonl
C/queue/
C/audit-replacements/
```

Original approved bundle SHA256: `e53d165ad01a9a69c777c9de9f28d4e9f76d0514df203aa8708fdcdd3004ebde`.

The financial epoch recovery for 356022 has already been performed. Do not repeat it. Its preserved receipt is `N/reddit-operational-repair-v1/epoch-recovery.json`.

</details>

<details>
<summary>Other-site preparation and four pending exact requests</summary>

These requests are prepared and reviewed, **not authorized to launch**:

| Request | Prepared resources | Status |
| --- | --- | --- |
| Shopping transfer recovery | 2 CPUs, 8 GiB, 7,140 s; one additional allocation and transfer campaign | Unapproved |
| GitLab transfer recovery | 2 CPUs, 8 GiB, 7,140 s; one additional allocation and transfer campaign | Unapproved |
| Map append-only transfer recovery | 2 CPUs, 8 GiB, 13,020 s; preserve the existing partial file; one additional allocation/campaign | Unapproved |
| CMS hosting/reset validation | 4 CPUs, 16 GiB, 7,200 cumulative s, at most two allocations | Unapproved |

All four request **zero GPUs and zero model/API budget**. Transfer recoveries preserve their original cumulative time/byte caps. CMS hosting requests new compute. Shopping, GitLab and CMS use third-party image candidates whose original-state equivalence is unverified.

The pending bundle contains exact limits, evidence references, required approval fields and launch commands:

```text
N/table5-next-jobs-bundle-v4.json
SHA256: 6dad3776d0364e1ae782b292430fe0476f1ae7880aaed7fc78842d4e3b0c93db

N/shopping-thirdparty-oci-transfer-exception-v1/
N/gitlab-thirdparty-oci-transfer-exception-v1/
N/map-original-backend-resume-exception-v1/
N/cms-hosting-validation-v3/
```

Approval-shaped examples within a pending bundle are requirements for a future receipt, not evidence that the user approved them. The bundle itself has `approved=false` and `submitted=false`.

CMS image acquisition **356009 completed and was independently audited**: 118 seconds for that job, 140 cumulative CMS seconds, 31 verified compressed layers. Hosting is unvalidated. Map frontend archives are also verified; backend job **356008 failed after 1,372 seconds**, preserving a 39,862,665,216-byte partial file. Its prepared recovery appends to the original inode; do not truncate, replace, copy or delete that prefix. Shopping **356007 failed after 37 seconds** and GitLab **356010 after 42 seconds** on a file-accounting race; fixes and transfer-exception requests are prepared.

Transfer byte caps remain Shopping 70,000,000,000, GitLab 80,000,000,000 and Map 195,000,000,000, including all prior charged/unknown bytes. Consult the bundle for full storage, temporary-space and reconnect limits. Download completion does not establish website authentication/reset readiness or authorize benchmark execution.

</details>

<details>
<summary>Supervision, documentation publication and workspace constraints</summary>

The supervisor is **stopped**, not actively monitoring or repairing. Its last state records `no_registered_work=true` and `verified_complete=false`. Recorded PID 3095173 is historical. Do not promise unattended supervision until a real continuation mechanism has been restarted and verified for the current approved job IDs.

```text
R/table5-supervisor/latest.json
R/table5-supervisor/finished.json
N/table5-active-supervisor-v2/supervisor.py
```

Routine callback acknowledgment command:

```bash
python3 -B scripts/supervise_webprm_table5.py --root /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/table5-supervisor --ack
```

Results documentation was committed as **`b1dd5351856f34cda5facb7945b568eaf75de78e`** in `/tmp/openwebrl-webprm-public-20261010`. **Push to `arm` failed because HTTPS credentials were unavailable.** That is the results commit before this handoff document; consult `R/publication-corrected-pilot-20261011.json` for subsequent publication receipts. Do not claim unpublished commits are on the remote. The main checkout contains unrelated dirty work; avoid broad staging.

Operational constraints:

- **Never write under `/gpfs/home/zixianma/`**, including subprocess caches or symlink targets. Use approved project/scrubbed paths or `/tmp`, and Python `-B`. Do not override `HOME` or `CODEX_HOME` to evade this boundary.
- Use auto-review for routine permissions. Any command invoking `rm` needs explicit approval of the exact command and targets.
- Do not use SSH, add allocations or API budget without exact authorization, reset cumulative accounting, rerun claimed primary cells, or delete retained historical evidence.
- Preserve frozen science, task identities, actor/selector protocols, raw/effective verdicts, invalid attempts, source lineage and every resource charge.
- Historical Online-Mind2Web diagnostics cannot populate this Table 5 comparison or a new matched Online-Mind2Web comparison.
- Honor the standing requirement to commit/push substantial project documentation to `arm`, excluding unrelated changes and private payloads; verify publication before claiming it succeeded.

</details>
