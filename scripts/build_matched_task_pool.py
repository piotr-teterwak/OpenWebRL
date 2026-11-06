#!/usr/bin/env python3
"""A new task pool matched to an existing one: same websites, same difficulty, unseen tasks.

For each task in --reference (the C2 pool), draw an unused --source task (WebGym train) on the same host
with the same difficulty, else the nearest difficulty on that host, at most as many per host as the
reference has there. The rest -- most reference hosts have a single task -- fall back to an unused task
on a new host in the same WebGym subdomain with the nearest difficulty, one task per new host, which
keeps the reference's long tail of sites (C2: 1,397 hosts for 2,091 tasks). Excludes reference intents,
blacklisted hosts, and anything close to the eval benchmarks (exact intent, or word-overlap Jaccard
>= --max-eval-jaccard against Online-Mind2Web / WebVoyager / DeepShop). Writes the reference's
tasks.jsonl format, so arm_c2 collection and the pair builder read it unchanged.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import re
from urllib.parse import urlparse


def host(url):
    return (urlparse(url if '//' in url else 'https://' + url).netloc or '').lower().removeprefix('www.')


def norm(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--blacklist', type=Path, required=True)
    ap.add_argument('--eval', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--split', default='round2_newpool_train')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-eval-jaccard', type=float, default=0.6)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    reference = [json.loads(l)['metadata'] for l in args.reference.read_text().splitlines()]
    blacklist = {l.strip().lower().removeprefix('www.') for l in args.blacklist.read_text().splitlines() if l.strip() and not l.startswith('#')}
    evals = []
    for path in args.eval:
        for line in path.read_text().splitlines():
            r = json.loads(line); m = r.get('metadata') or {}
            evals.append(m.get('intent') or r.get('ques') or r.get('task_name') or '')
    eval_exact = {norm(e) for e in evals}; eval_words = [words(e) for e in evals if e]
    by_eval_host = collections.defaultdict(list)
    for e in eval_words:
        for w in e: by_eval_host[w].append(e)
    ref_intents = {norm(t['intent']) for t in reference}
    # Difficulty of the reference tasks, looked up in the source by intent.
    source_rows, ref_difficulty, ref_subdomain = [], {}, {}
    for line in args.source.open():
        r = json.loads(line)
        n = norm(r['task_name'])
        if n in ref_intents:
            ref_difficulty[n] = r.get('difficulty'); ref_subdomain[n] = r.get('subdomain'); continue
        source_rows.append(r)
    pools = collections.defaultdict(lambda: collections.defaultdict(list))
    hosts = {host(t['start_url']) for t in reference}
    by_sub = collections.defaultdict(lambda: collections.defaultdict(list))   # fallback: new hosts
    for r in source_rows:
        h = host(r['website'])
        if not h or h in blacklist: continue
        if h in hosts: pools[h][r.get('difficulty')].append(r)
        else: by_sub[r.get('subdomain')][r.get('difficulty')].append(r)
    host_quota = collections.Counter(host(t['start_url']) for t in reference)
    new_host_used = set()
    for table in (pools, by_sub):
        for k in table:
            for d in table[k]: rng.shuffle(table[k][d])

    def near_eval(text):
        n = norm(text)
        if n in eval_exact: return True
        w = words(text)
        if not w: return True
        seen = set()
        for token in w:
            for e in by_eval_host.get(token, ()):
                if id(e) in seen: continue
                seen.add(id(e))
                if len(w & e) / len(w | e) >= args.max_eval_jaccard: return True
        return False

    def draw(table, d, new_hosts=False):
        for level in sorted(table, key=lambda x: (abs((x or 0) - (d or 0)), x or 0)):
            while table[level]:
                r = table[level].pop()
                if r['task_id'] in used or near_eval(r['task_name']): stats['skipped_used_or_near_eval'] += 1; continue
                if new_hosts:
                    if host(r['website']) in new_host_used: continue
                    new_host_used.add(host(r['website']))
                return r
        return None
    everything = collections.defaultdict(list)
    for sub in by_sub:
        for d, rows in by_sub[sub].items(): everything[d].extend(rows)
    for d in everything: rng.shuffle(everything[d])
    chosen, used, stats = [], set(), collections.Counter()
    for t in reference:
        h = host(t['start_url']); d = ref_difficulty.get(norm(t['intent'])); sub = ref_subdomain.get(norm(t['intent']))
        pick, how = (None, None)
        if h in pools and host_quota[h] > 0: pick, how = draw(pools[h], d), 'same_host'
        if pick: host_quota[h] -= 1
        if not pick and sub in by_sub: pick, how = draw(by_sub[sub], d, True), 'new_host_same_subdomain'
        if not pick: pick, how = draw(everything, d, True), 'new_host_any_subdomain'
        if not pick: stats['unmatched'] += 1; continue
        stats[how] += 1
        stats['same_difficulty' if pick.get('difficulty') == d else 'nearest_difficulty'] += 1
        used.add(pick['task_id'])
        tid = f"{pick['benchmark_name']}/{pick['task_id']}"
        chosen.append(dict(task_id=tid, metadata=dict(task_id=tid, intent=pick['task_name'], start_url=pick['website'],
                           source=str(args.source.name), source_task_id=pick['task_id'], difficulty=pick.get('difficulty'),
                           matched_reference=t['task_id'], split=args.split)))
    args.out.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chosen))
    report = dict(reference=len(reference), new_tasks=len(chosen), hosts=len({host(c['metadata']['start_url']) for c in chosen}), **stats)
    args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
