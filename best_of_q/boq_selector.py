"""Best-of-Q candidate proposal and action selection (arXiv 2601.22701 §3.3, App. A.2) for a frozen single-action actor.

The paper prompts its policy for N = 3 diverse candidates in one response, the first being "typically the VLM's greedy
choice". OpenWebRL-4B-SFT emits exactly one tool call, so the closest frozen-actor equivalent is: candidate 1 = a greedy
decode, candidates 2..N = sampled decodes (T = alt_temperature) kept only if they are a *different* action from every
candidate so far (action name, typed text, key, or a click/scroll point more than `tol` away on the 0-1000 grid). The
first round draws `draws` samples; if fewer than N distinct actions result, one more round of `extra_draws` is drawn.
A state can still end with fewer than N candidates (logged as n_distinct).

Selection policies (one per run):
  epsilon  the paper's exploration agent (Fig. 5): candidate 1 with probability 1 - eps, else a uniformly random other
           candidate (eps = 0.5 for the seed runs).
  random   the paper's "Random action" baseline (eps = 1): a uniformly random candidate.
  q        Best-of-Q: POST the candidates to the Q-function server (scalar mode) and execute the argmax (ties -> lower
           index, i.e. towards the greedy candidate).
Drop-in replacement for openwebrl.arm_inference.ActionSelector (same call signature and return value); every step is
appended to <output>/<sha(task_id)[:20]>.jsonl like the released selector traces.
"""
import asyncio
import base64
import hashlib
import json
import math
import random
import re
from pathlib import Path

from openwebrl.arm_inference import candidate_seed, request_selection_result, split_response

TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def action_key(action, tol=30):
    """Identity of an actor action for de-duplication: tool name + normalized arguments, points binned to `tol`."""
    m = TOOL.search(action or '')
    try:
        call = json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        call = None
    if not isinstance(call, dict):
        return ('raw', re.sub(r'\s+', ' ', (action or '').replace('<|im_end|>', '')).strip())
    args = call.get('arguments') or {}
    norm = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            norm.append((k, 'point', v[0], v[1]))
        elif isinstance(v, str):
            norm.append((k, re.sub(r'\s+', ' ', v.strip().strip('"\'').lower())))
        else:
            norm.append((k, json.dumps(v, sort_keys=True)))
    return (call.get('name'), tuple(norm), tol)


def same_action(a, b):
    if a[0] != b[0] or a[0] == 'raw' or len(a[1]) != len(b[1]):
        return a == b
    tol = a[2]
    for x, y in zip(a[1], b[1]):
        if x[0] != y[0]:
            return False
        if len(x) == 4 and len(y) == 4:
            if math.hypot(x[2] - y[2], x[3] - y[3]) > tol:
                return False
        elif x != y:
            return False
    return True


class BestOfQSelector:
    def __init__(self, policy, endpoint, output, seed=42, n=3, draws=3, extra_draws=4, epsilon=0.5,
                 alt_temperature=1.0, alt_top_p=1.0, tol=30, timeout=180):
        if policy not in ('epsilon', 'random', 'q'):
            raise ValueError(policy)
        self.policy, self.endpoint = policy, (endpoint or '').rstrip('/')
        self.output, self.seed, self.n = Path(output), seed, n
        self.draws, self.extra_draws, self.epsilon = draws, extra_draws, epsilon
        self.alt_temperature, self.alt_top_p, self.tol, self.timeout = alt_temperature, alt_top_p, tol, timeout
        self.output.mkdir(parents=True, exist_ok=True)

    def config(self):
        return dict(policy=self.policy, n=self.n, draws=self.draws, extra_draws=self.extra_draws, epsilon=self.epsilon,
                    alt_temperature=self.alt_temperature, alt_top_p=self.alt_top_p, tol=self.tol, seed=self.seed,
                    endpoint=self.endpoint if self.policy == 'q' else None)

    async def __call__(self, *, infer, url, input_text, sampling_params, images, observation, history, task, task_id,
                       turn, timeout):
        def params(j):
            if j == 0:  # the greedy candidate
                return dict(sampling_params, temperature=0.0, top_p=1.0, sampling_seed=candidate_seed(self.seed, task_id, turn, 0))
            return dict(sampling_params, temperature=self.alt_temperature, top_p=self.alt_top_p,
                        sampling_seed=candidate_seed(self.seed, task_id, turn, j))
        outputs = list(await asyncio.gather(*(infer(url, input_text, params(j), images, timeout_secs=timeout)
                                              for j in range(1 + self.draws))))
        keep, keys = [], []

        def admit(start):
            for j in range(start, len(outputs)):
                if len(keep) == self.n:
                    return
                k = action_key(split_response(outputs[j][0])['action'], self.tol)
                if not any(same_action(k, q) for q in keys):
                    keep.append(j); keys.append(k)
        admit(0)
        if len(keep) < self.n and self.extra_draws:
            first = len(outputs)
            outputs += await asyncio.gather(*(infer(url, input_text, params(j), images, timeout_secs=timeout)
                                              for j in range(first, first + self.extra_draws)))
            admit(first)
        candidates = [split_response(outputs[j][0]) for j in keep]
        rng = random.Random(candidate_seed(self.seed, task_id, turn, 999))
        scores = None
        if len(candidates) == 1:
            index = 0
        elif self.policy == 'epsilon':
            index = 0 if rng.random() < 1 - self.epsilon else rng.randrange(1, len(candidates))
        elif self.policy == 'random':
            index = rng.randrange(len(candidates))
        else:
            payload = {'mode': 'scalar', 'task': task, 'url': observation.get('active_tab_url', ''),
                       'history': [split_response(r) for r in history], 'candidates': candidates,
                       'screenshot': base64.b64encode(observation['screenshot']).decode()}
            result = await request_selection_result(self.endpoint, payload, self.timeout)
            scores = result.get('scores')
            if len(scores or []) != len(candidates) or not all(math.isfinite(s) for s in scores):
                raise ValueError('Invalid scalar score vector')
            index = max(range(len(scores)), key=lambda j: (scores[j], -j))
        record = {'mode': f'best_of_q:{self.policy}', 'task_id': task_id, 'turn': turn, 'selected_index': index,
                  'scores': scores, 'candidates': candidates, 'n_distinct': len(candidates), 'n_draws': len(outputs),
                  'draw_actions': [split_response(o[0])['action'] for o in outputs], 'kept_draws': keep,
                  'finish_types': [outputs[j][3] for j in keep], 'url': observation.get('active_tab_url', ''),
                  'prompt_sha256': hashlib.sha256(input_text.encode()).hexdigest(),
                  'screenshot_sha256': hashlib.sha256(observation['screenshot']).hexdigest()}
        path = self.output / (hashlib.sha256(str(task_id).encode()).hexdigest()[:20] + '.jsonl')
        with path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return outputs[keep[index]], {'selected_index': index, 'mode': record['mode'], 'fallback': None,
                                      'trace_path': str(path)}
