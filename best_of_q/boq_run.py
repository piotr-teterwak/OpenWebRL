#!/usr/bin/env python3
"""One Best-of-Q collection or evaluation run: openwebrl.arm_eval with its selector replaced by BestOfQSelector.

    boq_run.py --policy epsilon|random|q [--q-endpoint URL] [--epsilon 0.5] [--n 3] -- <arm_eval arguments>

Everything else (frozen actor, prompt, browser, 30 steps, judge, output layout, resumability) is arm_eval's. For epsilon
and random no selector service is needed, so arm_eval runs in its `baseline` mode (no /health check); for q it runs in
`scalar` mode against the Q-function server. The selector settings are written to <output>/boq_config.json, and a run
directory is refused if it was started with different settings.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', choices=['epsilon', 'random', 'q'], required=True)
    ap.add_argument('--q-endpoint', default='')
    ap.add_argument('--epsilon', type=float, default=0.5)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--extra-draws', type=int, default=4)
    ap.add_argument('--alt-temperature', type=float, default=1.0)
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    if a.policy == 'q' and not a.q_endpoint:
        sys.exit('--policy q needs --q-endpoint')
    from openwebrl import arm_eval, arm_inference
    from boq_selector import BestOfQSelector
    out = Path(rest[rest.index('--output') + 1])
    holder = {}

    def factory(mode, endpoint, output, seed=42, **_):
        holder['sel'] = BestOfQSelector(a.policy, a.q_endpoint, output, seed=seed, n=a.n, draws=a.draws,
                                        extra_draws=a.extra_draws, epsilon=a.epsilon, alt_temperature=a.alt_temperature)
        cfg = dict(holder['sel'].config(), actor_eval_mode=mode)
        path = out / 'boq_config.json'
        if path.exists() and json.loads(path.read_text()) != cfg:
            raise ValueError(f'{path} differs from this run\'s selector settings; use a new output directory')
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        return holder['sel']
    arm_inference.ActionSelector = factory  # arm_eval imports it inside run()
    mode = 'scalar' if a.policy == 'q' else 'baseline'
    args = ['--mode', mode] + (['--selector-endpoint', a.q_endpoint] if a.policy == 'q' else []) + rest
    sys.argv = ['arm_eval'] + args
    arm_eval.main()


if __name__ == '__main__':
    main()
