#!/usr/bin/env python3
"""Dataset statistics for mixed-pool Round 2 (sol2-Bmix) vs Round 2 (sol2-Br1), 2026-11-19.

    mix_pool_stats.py --mix <joint-data/sol2-Bmix> --panels S2M42=<panels-s42.jsonl> S2M43=<panels-s43.jsonl> \
        --labels S2M42=<labels-s42.jsonl> S2M43=<labels-s43.jsonl> [--compare <joint-data/sol2-Br1>]

Per dataset: pairs (successful / failed episodes), chosen action-type mix, and the give-up rates of chosen responses
with the regexes of the 2026-11-19 behaviour analysis (eval/redo2/behavior/final.py, GIVE): give-up phrase in the
reasoning, and give-up phrase without a `done` call. For the mix, also the share of Sol picks (all labelled panels
and the training pairs) that are SFT vs Round-1 candidates, overall and at states where a Round-1 candidate in the
panel gives up without calling `done`. Pair ids are <IDPREFIX>:<task>:<turn>; --panels/--labels are keyed by IDPREFIX.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

GIVE = re.compile(r"exhausted (all|the)|one (final|last) (approach|attempt|time|try)|cannot be completed|not possible to (complete|find)"
                  r"|unable to (complete|find)|use the done function|report (that|my findings)", re.I)
TC = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


def parse(resp):
    """Reasoning text and first tool name (as behavior/common.py parse)."""
    resp = resp.replace('<|im_end|>', '')
    think = resp.split('</think>', 1)[0].rsplit('<think>', 1)[-1].strip() if '</think>' in resp else ''
    name = None
    for m in TC.findall(resp):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            break
        if 'name' in d: name = d['name']; break
    return think, name


def gives_up(text):
    think, name = parse(text)
    return bool(GIVE.search(think)), bool(GIVE.search(think)) and name != 'done'


def pct(a, b): return f'{100 * a / b:.1f}%' if b else 'n/a'


def pair_stats(path, label):
    rows = [json.loads(l) for l in (Path(path) / 'joint_pairs.train.jsonl').open()]
    out = {'pairs': len(rows)}
    for name, sel in [('all', rows), ('successful', [r for r in rows if r.get('terminal_success') is True]),
                      ('failed', [r for r in rows if r.get('terminal_success') is False])]:
        g = sum(gives_up(r['chosen']['text'])[0] for r in sel); gn = sum(gives_up(r['chosen']['text'])[1] for r in sel)
        out[name] = dict(pairs=len(sel), chosen_give_up_phrase=pct(g, len(sel)), chosen_give_up_without_done=pct(gn, len(sel)))
    mix = Counter(parse(r['chosen']['text'])[1] for r in rows)
    out['chosen_action_mix'] = {k: pct(v, len(rows)) for k, v in mix.most_common()}
    rmix = Counter(parse(r['rejected']['text'])[1] for r in rows)
    out['rejected_action_mix'] = {k: pct(v, len(rows)) for k, v in rmix.most_common()}
    print(f'== {label}: {path}'); print(json.dumps(out, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mix', required=True); ap.add_argument('--compare')
    ap.add_argument('--panels', nargs='+', required=True); ap.add_argument('--labels', nargs='+', required=True)
    a = ap.parse_args()
    if a.compare: pair_stats(a.compare, 'Round 2 (compare)')
    rows = pair_stats(a.mix, 'mixed pool')
    panels, labels = {}, {}
    for spec in a.panels:
        prefix, path = spec.split('=', 1)
        for line in open(path):
            p = json.loads(line); key = prefix + ':' + p['id'][3:]
            r1_gives_up = any(s.startswith('r1:') and gives_up(c['response'])[1] for s, c in zip(p['sources'], p['raw_candidates']))
            panels[key] = dict(sources=p['sources'], r1_gives_up=r1_gives_up)
    for spec in a.labels:
        prefix, path = spec.split('=', 1)
        for line in open(path):
            r = json.loads(line); labels[prefix + ':' + r['id'][3:]] = r['selected_index']

    def share(keys_and_picks, title):
        c = Counter(); cg = Counter()
        for key, pick in keys_and_picks:
            p = panels[key]; src = p['sources'][pick].split(':')[0]
            c[src] += 1
            if p['r1_gives_up']: cg[src] += 1
        n, ng = sum(c.values()), sum(cg.values())
        print(f'{title}: n={n} SFT {pct(c["sft"], n)} / Round-1 {pct(c["r1"], n)} | at states where a Round-1 candidate gives up '
              f'without done: n={ng} SFT {pct(cg["sft"], ng)} / Round-1 {pct(cg["r1"], ng)}')
    print('== Sol picks by candidate source')
    share(labels.items(), 'all labelled panels')
    pairs = [(r['id'], r['chosen_index']) for r in rows]
    share(pairs, 'training pairs (chosen)')
    share([(r['id'], r['chosen_index']) for r in rows if r.get('terminal_success') is False], 'training pairs, failed episodes')
    share([(r['id'], r['rejected_index']) for r in rows], 'training pairs (rejected)')
    g = Counter()
    for r in rows:
        src = panels[r['id']]['sources'][r['chosen_index']].split(':')[0]
        g[src, gives_up(r['chosen']['text'])[1]] += 1
    print('chosen give-up-without-done by source:', {f'{k[0]}': pct(g[k[0], True], g[k[0], True] + g[k[0], False]) for k in g if k[1]})


if __name__ == '__main__':
    main()
