"""Methodology v2 of the "success rate w/o aborted tasks" metric.

Extends compute_eval_success_rate_woaborted.py (v1) with the forensic
active-tab confirmation step documented in the dashboard's Multi-Action
Atomization Experiments tab:

    COMPLETED + reward < 1 + BLOCKED_RE match on the agent's final done()
    text is only *excluded* from the denominator if the ACTIVE TAB at the
    final step independently shows a genuine block page (Cloudflare
    "Just a moment...", CAPTCHA challenge, "Access Denied", site outage),
    judged from the tab title/URL recorded in the final "Available tabs:"
    observation inside the eval_samples turn file. Unconfirmed candidates
    are recounted as failures.

Requires the eval run to have been executed with eval_samples saving
enabled (eval_samples/turn/<ts>/<task_id>_*.json under the eval dir).
Candidates whose sample file or tabs block cannot be located are counted
as failures (exclusion requires positive confirmation) and reported so
they can be reviewed manually.

Usage:
    python compute_eval_success_rate_woaborted_v2.py <eval_dir_1> [<eval_dir_2> ...] \
        [--audit-out audit.json] [--json]
"""

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Import v1 helpers without triggering the openwebrl package __init__.
_V1_PATH = Path(__file__).parent / "compute_eval_success_rate_woaborted.py"
_spec = importlib.util.spec_from_file_location("_sr_v1", _V1_PATH)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

BLOCKED_RE = _v1.BLOCKED_RE
_load_jsonl_record = _v1._load_jsonl_record
_reward_value = _v1._reward_value
_final_done_text = _v1._final_done_text

# Genuine-block indicators matched against the active tab's "URL - Title"
# line. Deliberately title/URL-level: these phrases essentially only appear
# on interstitial/denial/outage pages. Calibrated against the audited
# candidate lines of this session's 12-run baseline eval plus the
# action-space ablation audit.
BLOCK_LINE_RE = re.compile(
    r"just a moment|attention required|checking your browser|cloudflare"
    r"|captcha|botchallenge|denied|forbidden"
    r"|robot or human|are you a (robot|human)|human verification"
    r"|verify you|verification required|security (check|verification)"
    r"|pardon our interruption|request could not be satisfied"
    r"|unusual traffic|temporarily unavailable|service unavailable"
    r"|under maintenance|be right back|bot detection|automated access"
    r"|chrome-error://|err_(http2|connection|name|timed|ssl|cert)"
    r"|access to this (site|page)|site can.t be reached",
    re.IGNORECASE,
)
# Patterns only safe against the page TITLE (after "URL - "): short codes
# and generic words that also occur inside working URLs' query params
# (e.g. "403" inside a Bing tracking-id parameter).
BLOCK_TITLE_RE = re.compile(
    r"\b40[03]\b|^sorry\b|\bblocked\b|bot or not|bot check|one moment"
    r"|browsing activity has been paused|rate threshold exceeded",
    re.IGNORECASE,
)


def _is_block_page(active_line: str) -> bool:
    if BLOCK_LINE_RE.search(active_line):
        return True
    # Line format is "URL - Title" (URLs contain no spaces).
    title = active_line.split(" - ", 1)[1] if " - " in active_line else active_line
    return bool(BLOCK_TITLE_RE.search(title))

TABS_BLOCK_RE = re.compile(r"Available tabs:\n((?:- Tab .*\n?)+)")
ACTIVE_LINE_RE = re.compile(r"- Tab \d+ \(active\): (.*)")


def _find_sample_file(eval_dir: Path, task_id: str) -> Path | None:
    matches = sorted(eval_dir.glob(f"eval_samples/turn/*/{task_id}_2*.json"))
    return matches[-1] if matches else None


def _final_active_tab_line(sample_path: Path) -> str | None:
    """Active tab's 'URL - Title' from the last observation in the sample."""
    with sample_path.open() as f:
        data = json.load(f)
    text = data.get("llm_input_texts") or ""
    blocks = TABS_BLOCK_RE.findall(text)
    if not blocks:
        return None
    m = ACTIVE_LINE_RE.search(blocks[-1])
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Training-parity abort mode (--abort-mode training)
# ---------------------------------------------------------------------------
# Mirrors openwebrl.generate_browser._should_remove_browser_sample, which is
# what the TRAINING loop uses to drop a rollout from the loss. Kept as literal
# copies of that function's marker tuples so drift is greppable.
#
# Two deliberate differences from v1/v2 scoring:
#   1. Exclusion is driven by terminate_reason markers, NOT by status alone.
#      An ABORTED sample whose reason matches no marker (e.g. "generation_abort")
#      is therefore COUNTED AS A FAILURE here, where v1/v2 would exclude it.
#   2. No BLOCKED_RE / active-tab exclusion at all. Anti-bot blocks are counted
#      as failures. Only infrastructure failures leave the denominator.
# The result is a stricter, more conservative success rate; it is NOT comparable
# to v1/v2 numbers or to published comparators scored with those.
TRAINING_TIMEOUT_MARKERS = (
    "rollout_task_timeout",
    "sglang /generate timed out",
    "sandbox exec timed out",
    "not healthy within",
)
TRAINING_SANDBOX_MARKERS = (
    "env_step_error",
    "env_server /reset error",
    "env_server /step error",
)


def _terminate_reason(record: dict[str, Any]) -> str:
    """run_evaluate.py writes terminate_reason at top level; tolerate metadata too."""
    reason = record.get("terminate_reason")
    if not reason:
        reason = (record.get("metadata") or {}).get("terminate_reason", "")
    return str(reason or "").lower()


def _is_training_system_failure(record: dict[str, Any]) -> bool:
    """Literal mirror of generate_browser._should_remove_browser_sample."""
    reason = _terminate_reason(record)
    status = str(record.get("status", ""))
    if any(marker in reason for marker in TRAINING_TIMEOUT_MARKERS):
        return True
    if "ABORTED" in status and any(marker in reason for marker in TRAINING_SANDBOX_MARKERS):
        return True
    return False


def compute_stats_training(eval_dir: Path) -> dict[str, Any]:
    result_paths = sorted(eval_dir.glob("**/results_task_*.jsonl"))

    n_total = 0
    n_excluded = 0
    n_success = 0
    n_failure = 0
    n_status_aborted = 0
    n_aborted_kept = 0          # ABORTED but no marker -> counted as failure
    excluded: list[dict[str, Any]] = []

    for path in result_paths:
        record = _load_jsonl_record(path)
        if record is None:
            continue
        n_total += 1
        status = str(record.get("status", ""))
        if "ABORTED" in status:
            n_status_aborted += 1

        if _is_training_system_failure(record):
            n_excluded += 1
            excluded.append({
                "task_id": record.get("task_id") or path.stem.replace("results_task_", ""),
                "status": status,
                "terminate_reason": _terminate_reason(record)[:200],
            })
            continue

        if "ABORTED" in status:
            n_aborted_kept += 1

        if "COMPLETED" in status and _reward_value(record) >= 1.0:
            n_success += 1
        else:
            n_failure += 1

    denom = n_success + n_failure
    return {
        "eval_dir": str(eval_dir),
        "abort_mode": "training",
        "n_total_result_files": n_total,
        "n_excluded_system_failure": n_excluded,
        "n_status_aborted": n_status_aborted,
        "n_aborted_counted_as_failure": n_aborted_kept,
        "n_success": n_success,
        "n_failure": n_failure,
        "denominator": denom,
        "success_rate": n_success / denom if denom else 0.0,
        "audit": {"excluded_system_failures": excluded},
    }


def compute_stats_v2(eval_dir: Path) -> dict[str, Any]:
    result_paths = sorted(eval_dir.glob("**/results_task_*.jsonl"))

    n_total = 0
    n_aborted = 0
    n_success = 0
    n_failure = 0
    confirmed: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []

    for path in result_paths:
        record = _load_jsonl_record(path)
        if record is None:
            continue
        n_total += 1
        status = str(record.get("status", ""))

        if "ABORTED" in status:
            n_aborted += 1
            continue

        reward = _reward_value(record)
        if "COMPLETED" in status and reward >= 1.0:
            n_success += 1
        elif "COMPLETED" in status and reward < 1.0:
            if not BLOCKED_RE.search(_final_done_text(record)):
                n_failure += 1
                continue
            # Forensic step: active tab must independently confirm the block.
            task_id = record.get("task_id") or path.stem.replace("results_task_", "")
            sample_path = _find_sample_file(eval_dir, task_id)
            active_line = _final_active_tab_line(sample_path) if sample_path else None
            entry = {"task_id": task_id, "active_line": active_line,
                     "sample_file": str(sample_path) if sample_path else None}
            if active_line is None:
                unverifiable.append(entry)
                n_failure += 1
            elif _is_block_page(active_line):
                confirmed.append(entry)
            else:
                unconfirmed.append(entry)
                n_failure += 1
        else:  # FAILED / TRUNCATED — no forensic check applies
            n_failure += 1

    denom = n_success + n_failure
    return {
        "eval_dir": str(eval_dir),
        "n_total_result_files": n_total,
        "n_excluded_aborted": n_aborted,
        "n_blocked_candidates": len(confirmed) + len(unconfirmed) + len(unverifiable),
        "n_excluded_confirmed_blocked": len(confirmed),
        "n_unconfirmed_recounted_as_failure": len(unconfirmed),
        "n_unverifiable_recounted_as_failure": len(unverifiable),
        "n_success": n_success,
        "n_failure": n_failure,
        "denominator": denom,
        "success_rate": n_success / denom if denom else 0.0,
        "audit": {"confirmed": confirmed, "unconfirmed": unconfirmed,
                  "unverifiable": unverifiable},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dirs", nargs="+")
    parser.add_argument("--audit-out", help="Write per-candidate audit details to this JSON file")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--abort-mode",
        choices=["v2", "training"],
        default="v2",
        help=(
            "v2 (default): exclude ABORTED plus active-tab-confirmed blocks. "
            "training: exclude ONLY infrastructure failures, using the same "
            "terminate_reason markers as generate_browser."
            "_should_remove_browser_sample. Anti-bot blocks count as failures "
            "and an ABORTED with no matching marker counts as a failure, so "
            "the number is stricter and NOT comparable to v2/published figures."
        ),
    )
    args = parser.parse_args()

    scorer = compute_stats_training if args.abort_mode == "training" else compute_stats_v2
    per_run = [scorer(Path(d)) for d in args.eval_dirs]
    rates = [r["success_rate"] for r in per_run]
    mean = statistics.mean(rates) if rates else 0.0
    stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0

    if args.audit_out:
        audit = {r["eval_dir"]: r["audit"] for r in per_run}
        Path(args.audit_out).write_text(json.dumps(audit, indent=2))

    for r in per_run:
        r = dict(r)
        r.pop("audit")
        if args.json:
            print(json.dumps(r))
        elif args.abort_mode == "training":
            print(
                f"{r['eval_dir']}: success_rate={r['success_rate']*100:.1f}% "
                f"(n_success={r['n_success']}, denom={r['denominator']}, "
                f"excluded_system_failure={r['n_excluded_system_failure']}, "
                f"status_aborted={r['n_status_aborted']}, "
                f"aborted_counted_as_failure={r['n_aborted_counted_as_failure']})"
            )
        else:
            print(
                f"{r['eval_dir']}: success_rate={r['success_rate']*100:.1f}% "
                f"(n_success={r['n_success']}, denom={r['denominator']}, "
                f"aborted={r['n_excluded_aborted']}, "
                f"blocked_candidates={r['n_blocked_candidates']}, "
                f"confirmed_excluded={r['n_excluded_confirmed_blocked']}, "
                f"unconfirmed_failed={r['n_unconfirmed_recounted_as_failure']}, "
                f"unverifiable_failed={r['n_unverifiable_recounted_as_failure']})"
            )
    print()
    print(f"mean = {mean*100:.2f}%  std = {stdev*100:.2f}%  (n_runs={len(rates)})")


if __name__ == "__main__":
    main()
