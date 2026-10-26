#!/usr/bin/env python3
"""Re-judge finished Online-Mind2Web evals with a different judge model, without re-running browsers.

The OM2W judge reads only the task, the assistant thoughts/actions in metadata.messages and the
final screenshot (metadata.full_image_list[-1]); every arm_eval result file stores all three. This
rebuilds that input from each <eval>/results/*.json, calls reward_online_mind2web._judge with the
new model and writes <eval>/rejudge-<model>/results/*.json plus summary.json. Originals are never
modified.

Validity follows arm_eval: an episode is valid if it has a judge score and was not ABORTED. Episodes
that never produced a trajectory (no messages) or were ABORTED stay invalid and are not re-judged.

    python scripts/rejudge_om2w.py --model gpt-5.6-terra --parallel 8 <eval dir> [<eval dir> ...]
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace


async def rejudge_dir(eval_dir, model, gate, judge, sample_cls, tool_parser_for, limit=None):
    out = eval_dir / f'rejudge-{model}'
    (out / 'results').mkdir(parents=True, exist_ok=True)
    actor = json.loads((eval_dir / 'manifest.json').read_text())['actor']
    args = SimpleNamespace(judge_api_model=model, judge_timeout_secs=180, judge_api_mode='served',
                           hf_checkpoint=actor)
    tool_parser = tool_parser_for(actor)

    async def one(path):
        dest = out / 'results' / path.name
        if dest.exists():
            return json.loads(dest.read_text())
        old = json.loads(path.read_text())
        meta = old.get('metadata') or {}
        status = str(old.get('status') or '')
        new = dict(task_id=old['task_id'], old_reward=old.get('reward'), old_valid=old.get('valid'),
                   status=status, terminate_reason=old.get('terminate_reason'))
        if not meta.get('messages') or 'COMPLETED' not in status:
            # No trajectory or not completed: arm_eval scored these 0 (or invalid); keep as they were.
            new.update(reward=old.get('reward'), valid=old.get('valid', False), judge_text='not re-judged')
        else:
            async with gate:
                sample = SimpleNamespace(metadata=meta, status=sample_cls.Status.COMPLETED)
                score, text, timed_out = await judge(args, sample, tool_parser)
            new.update(reward=score, valid=score is not None, judge_text=text, judge_timeout=timed_out)
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1))
        tmp.replace(dest)
        return new

    paths = sorted((eval_dir / 'results').glob('*.json'))[:limit]
    records = await asyncio.gather(*(one(p) for p in paths))
    valid = [r for r in records if r.get('valid')]
    succ = sum(r.get('reward') == 1.0 for r in valid)
    both = [r for r in records if r.get('valid') and r.get('old_valid')]
    summary = dict(judge=model, source=str(eval_dir), scheduled=len(records), valid=len(valid), successes=succ,
                   success_rate_all_scheduled=succ / len(records) if records else None,
                   success_rate_valid=succ / len(valid) if valid else None,
                   agreement_with_original=(sum((r['reward'] == 1.0) == (r['old_reward'] == 1.0) for r in both) / len(both))
                   if both else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('evals', nargs='+', type=Path, help='.../<label>/online-mind2web directories')
    ap.add_argument('--model', default='gpt-5.6-terra')
    ap.add_argument('--parallel', type=int, default=8, help='concurrent judge calls (keep small: shared network)')
    ap.add_argument('--limit', type=int, help='pilot: first N results per eval only (resumable)')
    a = ap.parse_args()
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    from openwebrl.eval import reward_online_mind2web as om2w
    from openwebrl.base.utils import ToolParser
    from openwebrl.eval._shared import _TOOLS_INFO, resolve_parser_type
    from slime.utils.types import Sample
    gate = asyncio.Semaphore(a.parallel)
    for d in a.evals:
        await rejudge_dir(d, a.model, gate, om2w._judge, Sample,
                          lambda actor: ToolParser(_TOOLS_INFO, parser_type=resolve_parser_type(actor)), a.limit)


if __name__ == '__main__':
    asyncio.run(main())
