#!/usr/bin/env python3
"""Repoint /gpfs/scrubbed/zixianma/... paths in a built joint-data directory to the working mirror.

    repoint_zixian_paths.py <data dir> [--mirror /gpfs/scrubbed/piotrt/owrl/zixian-mirror]

Zixian's scrubbed runtime is past the 60-day purge (her C2 screenshots went ~2026-11-07), so new pair sets
reference the copies under the mirror instead (made 2026-11-10 with fresh timestamps; the recovered C2
validation screenshots live in /gpfs/scrubbed/piotrt/owrl/zixian-c2-images). Only paths whose mirror copy
exists are rewritten, and each rewritten file must have the same size as the original when that still
exists. Run before verify_arm_joint_data / data_config so the recorded hashes cover the rewritten files.
"""
import argparse
import os
import re
from pathlib import Path

PREFIX = '/gpfs/scrubbed/zixianma/'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', type=Path); ap.add_argument('--mirror', default='/gpfs/scrubbed/piotrt/owrl/zixian-mirror')
    a = ap.parse_args(); mirror = a.mirror.rstrip('/') + '/'
    pat = re.compile(re.escape(PREFIX) + r'[A-Za-z0-9_./+-]+')
    seen = {}

    def sub(m):
        old = m.group(0); new = mirror + old[len(PREFIX):]
        if old not in seen:
            ok = os.path.exists(new) and (not os.path.isfile(old) or os.path.getsize(old) == os.path.getsize(new))
            seen[old] = new if ok else old
        return seen[old]

    for f in sorted(list(a.data.glob('*.jsonl')) + list(a.data.glob('image-manifest.json'))):
        s = f.read_text(); t = pat.sub(sub, s)
        if t != s:
            f.write_text(t)
    moved = sum(v != k for k, v in seen.items()); kept = [k for k, v in seen.items() if v == k]
    print(f'[repoint] {a.data}: {moved} paths -> {mirror}, {len(kept)} left (no mirror copy)', *kept[:3])


if __name__ == '__main__':
    main()
