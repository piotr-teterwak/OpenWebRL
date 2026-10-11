"""Auditable, CPU-only scalar advantages for the proposed action-credit study.

This is an offline numerical core, not a production trainer. Callers must verify
policy/protocol provenance, replay equality (including policy-visible history),
and response/token masks. State keys are opaque *verified* identities; this module
does not establish browser state equivalence. Invalid/missing outcomes are never
converted into failures. Every planned identity must have an explicit record.

Pinned primary sources (inspected 2026-10-11):
* Tree-GRPO: https://arxiv.org/html/2509.21240v3 and AMAP-ML/Tree-GRPO
  commit 19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59, especially
  verl/trainer/ppo/ray_trainer_ts.py:168-189 and core_algos.py:101-142.
  The launcher uses estimator ``tree``: sum of within-tree normalization and
  normalization over ALL leaves of the same task group. Inter-tree is NOT a
  normalization of tree means. ``compute_tree_grpo_outcome_advantage`` belongs
  to the separate ``tree_2norm`` ablation and is deliberately not reproduced.
  Both normalizations use sample std (ddof=1), epsilon=1e-6, singleton mean=0,
  std=1. The full leaf paths repeat shared prefixes. Official state_masking
  excludes environment tokens; the web launcher uses token-mean dual-clip PPO.
  These scalar results alone do not reproduce its sampler or loss reduction.
* GiGPO: https://arxiv.org/html/2505.10978v3 and langfengQ/verl-agent
  commit 20bd331bdbc9026a5668e11362178e10ab7400c8, gigpo/core_gigpo.py.
  Exact anchors match within task groups, including repeat visits in a trajectory.
  Step return is sum gamma**offset * reward. Singleton step advantage is zero.
  Upstream defaults to mean-only normalization and STEP-WEIGHTED episode stats
  (compute_mean_std_cross_steps=True); its False option weights trajectories
  equally. We expose both and label the returned method; neither is implicit.

All normalized std denominators are std + epsilon (not sqrt(var + epsilon)).
``singleton='upstream'`` preserves the upstream raw-reward singleton behavior;
``'zero'`` is an explicit controlled deviation. Zero-variance nonsingleton groups
produce zero. Python double precision is used, not upstream float32.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Outcome:
    rollout_id: str
    task_id: str
    group_id: str
    reward: Optional[float]
    status: str = "valid"
    tree_id: Optional[str] = None


@dataclass(frozen=True)
class CandidateDraw:
    draw_id: str
    task_id: str
    state_id: str
    candidate_id: str
    reward: Optional[float]
    status: str = "valid"
    partition: str = "train"


@dataclass(frozen=True)
class Step:
    step_id: str
    state_key: str
    reward: float
    position: int


@dataclass(frozen=True)
class Episode:
    trajectory_id: str
    task_id: str
    group_id: str
    reward: Optional[float]
    steps: Tuple[Step, ...]
    status: str = "valid"


@dataclass(frozen=True)
class AdvantageValue:
    key: Tuple[str, ...]
    advantage: Optional[float]
    components: Tuple[Tuple[str, float], ...]
    valid_count: int
    invalid_count: int
    missing_count: int


@dataclass(frozen=True)
class GroupStats:
    """Counts and the actual normalization baseline/scale, not a confidence CI.

    In singleton mode mean/std include the documented upstream overrides and
    need not equal empirical moments. normalization_count may exceed valid_count
    for explicitly step-weighted episode statistics.
    """
    key: Tuple[str, ...]
    planned_count: int
    valid_count: int
    invalid_count: int
    missing_count: int
    normalization_count: int
    mean: Optional[float]
    std: Optional[float]
    normalizer: Optional[float]
    ddof: int
    singleton_rule: str


@dataclass(frozen=True)
class AdvantageResult:
    method: str
    values: Tuple[AdvantageValue, ...]
    groups: Tuple[GroupStats, ...]


def _id(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        out = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is outside float range") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _status(record):
    if record.status not in ("valid", "invalid", "missing"):
        raise ValueError("status must be valid, invalid, or missing")
    if record.status == "valid":
        _finite(record.reward, "reward")
    elif record.reward is not None:
        raise ValueError("excluded records must have reward=None; no imputation")


def _manifest(records, expected_ids, field, cls):
    if isinstance(records, (str, bytes)) or isinstance(expected_ids, (str, bytes)):
        raise ValueError("records and expected_ids must be sequences")
    records, expected_ids = tuple(records), tuple(expected_ids)
    if not records or not expected_ids:
        raise ValueError("empty collection/manifest is not an estimable batch")
    for value in expected_ids:
        _id(value, "expected ID")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate expected IDs")
    actual = []
    for record in records:
        if not isinstance(record, cls):
            raise ValueError(f"expected {cls.__name__} records")
        value = getattr(record, field)
        _id(value, field)
        _id(record.task_id, "task_id")
        _status(record)
        actual.append(value)
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate record IDs")
    if set(actual) != set(expected_ids):
        raise ValueError("manifest mismatch: represent missing records explicitly")
    return records


def _options(normalization, ddof, epsilon, singleton):
    if normalization not in ("mean", "mean_std"):
        raise ValueError("normalization must be mean or mean_std")
    if type(ddof) is not int or ddof not in (0, 1):
        raise ValueError("ddof must be explicitly 0 or 1")
    epsilon = _finite(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if singleton not in ("upstream", "zero"):
        raise ValueError("singleton must be upstream or zero")


def _moments(values, ddof):
    """Stable sum and deterministic ordering; fail closed on arithmetic overflow."""
    try:
        values = sorted(float(v) for v in values)
        mean = math.fsum(v / len(values) for v in values)
        std = (math.sqrt(math.fsum((v - mean) ** 2 for v in values) /
                         (len(values) - ddof)) if len(values) > 1 else 0.0)
    except (OverflowError, ValueError) as exc:
        raise ValueError("nonfinite group arithmetic") from exc
    return _finite(mean, "group mean"), _finite(std, "group std")


def _stats(key, records, normalization, ddof, epsilon, singleton, values=None):
    valid = sum(r.status == "valid" for r in records)
    invalid = sum(r.status == "invalid" for r in records)
    missing = sum(r.status == "missing" for r in records)
    if values is None:
        values = [r.reward for r in records if r.status == "valid"]
    count = len(values)
    mean = std = normalizer = None
    if count:
        mean, std = _moments(values, ddof)
        if count == 1 and singleton == "upstream":
            mean, std = 0.0, 1.0
        # GiGPO singleton step std is 1, though its centered result is zero.
        elif count == 1:
            std = 1.0
        normalizer = std + epsilon if normalization == "mean_std" else 1.0
        _finite(normalizer, "normalizer")
    return GroupStats(key, len(records), valid, invalid, missing, count,
                      mean, std, normalizer, ddof, singleton)


def _normalize(value, stats):
    return _finite((value - stats.mean) / stats.normalizer, "advantage")


def _value(key, advantage, components, stats):
    return AdvantageValue(key, advantage, tuple(components), stats.valid_count,
                          stats.invalid_count, stats.missing_count)


def _scoped(records, field):
    groups, owners = defaultdict(list), {}
    for r in records:
        identity = getattr(r, field)
        _id(identity, field)
        if identity in owners and owners[identity] != r.task_id:
            raise ValueError(f"{field} reused across tasks: possible leakage")
        owners[identity] = r.task_id
        groups[(r.task_id, identity)].append(r)
    return groups


def outcome_group_advantages(records: Sequence[Outcome], *, expected_ids,
                             normalization="mean_std", ddof=1, epsilon=1e-6,
                             singleton="upstream") -> AdvantageResult:
    """Equal rollout weights within each explicit task/initial-condition group.

    Also supplies the tree-sampling outcome-only control: tree IDs do not change
    its formula. Excluded records yield None, not a trainable zero advantage.
    Values are sorted by ID. Value counts refer to the task group.
    """
    _options(normalization, ddof, epsilon, singleton)
    records = _manifest(records, expected_ids, "rollout_id", Outcome)
    values, stats = [], []
    for key, group in sorted(_scoped(records, "group_id").items()):
        stat = _stats(("outcome",) + key, group, normalization, ddof, epsilon, singleton)
        stats.append(stat)
        for r in group:
            a = _normalize(r.reward, stat) if r.status == "valid" else None
            values.append(_value((r.rollout_id,), a, (("outcome", a),) if a is not None else (), stat))
    return AdvantageResult("outcome_group", tuple(sorted(values, key=lambda v: v.key)), tuple(stats))


def tree_grpo_advantages(records: Sequence[Outcome], *, expected_ids, ddof=1,
                         epsilon=1e-6, singleton="upstream") -> AdvantageResult:
    """Official estimator 'tree': A_inter(task-pooled leaves) + A_intra(tree).

    Each leaf has equal weight in both groups even when trees have unequal leaf
    counts. No normalization of the sum. Repeated prefixes are a loss concern,
    not additional observations for these scalar statistics. Invalid exclusion
    and explicit zero singleton mode are controlled extensions to upstream.
    """
    _options("mean_std", ddof, epsilon, singleton)
    records = _manifest(records, expected_ids, "rollout_id", Outcome)
    task_groups = _scoped(records, "group_id")
    trees, owners = defaultdict(list), {}
    for r in records:
        _id(r.tree_id, "tree_id")
        owner = (r.task_id, r.group_id)
        if r.tree_id in owners and owners[r.tree_id] != owner:
            raise ValueError("tree_id reused across task groups: possible leakage")
        owners[r.tree_id] = owner
        trees[owner + (r.tree_id,)].append(r)
    inter = {k: _stats(("inter",) + k, v, "mean_std", ddof, epsilon, singleton)
             for k, v in task_groups.items()}
    intra = {k: _stats(("intra",) + k, v, "mean_std", ddof, epsilon, singleton)
             for k, v in trees.items()}
    values = []
    for r in records:
        s = inter[(r.task_id, r.group_id)]
        if r.status == "valid":
            a = _normalize(r.reward, s)
            b = _normalize(r.reward, intra[(r.task_id, r.group_id, r.tree_id)])
            values.append(_value((r.rollout_id,), _finite(a + b, "advantage"),
                                 (("inter", a), ("intra", b)), s))
        else:
            values.append(_value((r.rollout_id,), None, (), s))
    return AdvantageResult("tree_grpo", tuple(sorted(values, key=lambda v: v.key)),
                           tuple(sorted((*inter.values(), *intra.values()), key=lambda s: s.key)))


def candidate_q_advantages(draws: Sequence[CandidateDraw], *, expected_ids,
                           partition="train") -> AdvantageResult:
    """Q(a)=mean(valid continuations); A(a)=Q(a)-equal-candidate mean Q.

    Draws have equal weight WITHIN each candidate; candidates have equal weight
    in centering regardless of unequal draw counts. If any planned candidate has
    zero valid draws the entire state's advantages are None: dropping that
    candidate would silently change the comparison. Available Q estimates remain
    in components. A missing/invalid draw does not invalidate a candidate that
    has other valid draws. No standardization or shrinkage is applied.

    Value counts are draw counts for that candidate. ``candidate`` stats count
    draws; ``candidate_center`` stats count candidates (a missing candidate has
    no Q and at least one missing draw, otherwise it is invalid). The manifest
    must declare at least one slot for every intended candidate. ``partition``
    must be train: continuation holdout/evaluation records cannot enter targets.
    """
    if partition != "train":
        raise ValueError("only training continuation draws may form training targets")
    records = _manifest(draws, expected_ids, "draw_id", CandidateDraw)
    states = _scoped(records, "state_id")
    values, stats = [], []
    for state, group in sorted(states.items()):
        candidates = defaultdict(list)
        for r in group:
            _id(r.candidate_id, "candidate_id")
            if r.partition != partition:
                raise ValueError("held-out or mixed-partition continuation leakage")
            candidates[r.candidate_id].append(r)
        candidate_stats, pseudo = {}, []
        for candidate, rows in sorted(candidates.items()):
            s = _stats(("candidate",) + state + (candidate,), rows,
                       "mean", 0, 1e-6, "zero")
            candidate_stats[candidate] = s
            stats.append(s)
            status = "valid" if s.valid_count else ("missing" if s.missing_count else "invalid")
            pseudo.append(Outcome(candidate, state[0], state[1], s.mean, status))
        center = _stats(("candidate_center",) + state, pseudo,
                        "mean", 0, 1e-6, "zero")
        complete = center.valid_count == center.planned_count
        if not complete:
            center = GroupStats(center.key, center.planned_count, center.valid_count,
                                center.invalid_count, center.missing_count,
                                center.normalization_count, None, None, None, 0, "zero")
        stats.append(center)
        for candidate, s in sorted(candidate_stats.items()):
            components = (("q", s.mean),) if s.mean is not None else ()
            a = _normalize(s.mean, center) if complete else None
            if complete:
                components += (("candidate_mean", center.mean),)
            values.append(_value(state + (candidate,), a, components, s))
    return AdvantageResult("continuation_q", tuple(values), tuple(stats))


def gigpo_advantages(episodes: Sequence[Episode], *, expected_ids, gamma=1.0,
                     step_weight=1.0, normalization="mean", ddof=1, epsilon=1e-6,
                     episode_weighting="trajectory", singleton="upstream") -> AdvantageResult:
    """Episode-relative advantage + weight * matched-state return advantage.

    ``episode_weighting='trajectory'`` is equal trajectories (paper contract /
    upstream compute_mean_std_cross_steps=False). ``'step'`` reproduces the
    pinned upstream default by repeating each episode reward per action step.
    Episode reward must equal the sum of immediate step rewards. Step positions
    must be unique contiguous zero-based integers; input storage order need not
    be chronological. Exact state matching is scoped to task AND group.

    Output key is (trajectory_id, step_id); counts refer to matched-state steps.
    Episode stats count episodes but normalization_count exposes repeated-step
    weighting. Missing/invalid episodes with no steps remain visible in episode
    statistics, with no fabricated action targets. Excluded partial steps have
    None advantages and are excluded from discounted-return computations.
    """
    _options(normalization, ddof, epsilon, singleton)
    gamma, step_weight = _finite(gamma, "gamma"), _finite(step_weight, "step_weight")
    if not 0 <= gamma <= 1 or step_weight < 0:
        raise ValueError("gamma must be in [0,1] and step_weight nonnegative")
    if episode_weighting not in ("trajectory", "step"):
        raise ValueError("episode_weighting must be trajectory or step")
    records = _manifest(episodes, expected_ids, "trajectory_id", Episode)
    groups = _scoped(records, "group_id")
    states, returns, ordered = defaultdict(list), {}, {}
    for e in records:
        if not isinstance(e.steps, tuple):
            raise ValueError("episode steps must be an immutable tuple")
        seen = set()
        for s in e.steps:
            if not isinstance(s, Step):
                raise ValueError("expected Step")
            _id(s.step_id, "step_id")
            _id(s.state_key, "state_key")
            _finite(s.reward, "immediate reward")
            if s.step_id in seen:
                raise ValueError("duplicate step identity in trajectory")
            seen.add(s.step_id)
            if type(s.position) is not int or s.position < 0:
                raise ValueError("step position must be a nonnegative integer")
        steps = tuple(sorted(e.steps, key=lambda s: s.position))
        if [s.position for s in steps] != list(range(len(steps))):
            raise ValueError("step positions must be contiguous and unique")
        if e.status == "valid":
            if not steps:
                raise ValueError("valid episode must contain at least one step")
            try:
                total = math.fsum(s.reward for s in steps)
            except OverflowError as exc:
                raise ValueError("nonfinite episode reward sum") from exc
            if not math.isclose(total, e.reward, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("episode reward differs from sum of immediate rewards")
            running = 0.0
            for s in reversed(steps):
                running = _finite(s.reward + gamma * running, "discounted return")
                returns[(e.trajectory_id, s.step_id)] = running
        ordered[e.trajectory_id] = steps
        for s in steps:
            reward = returns.get((e.trajectory_id, s.step_id))
            states[(e.task_id, e.group_id, s.state_key)].append(
                Outcome(e.trajectory_id + "/" + s.step_id, e.task_id, e.group_id, reward, e.status))
    episode_stats = {}
    for key, group in groups.items():
        rewards = [e.reward for e in group if e.status == "valid"
                   for _ in range(len(e.steps) if episode_weighting == "step" else 1)]
        episode_stats[key] = _stats(("episode",) + key, group, normalization,
                                    ddof, epsilon, singleton, rewards)
    state_stats = {key: _stats(("state",) + key, rows, normalization, ddof, epsilon, "zero")
                   for key, rows in states.items()}
    values = []
    for e in records:
        for s in ordered[e.trajectory_id]:
            key = (e.trajectory_id, s.step_id)
            stat = state_stats[(e.task_id, e.group_id, s.state_key)]
            if e.status == "valid":
                a = _normalize(e.reward, episode_stats[(e.task_id, e.group_id)])
                b = _normalize(returns[key], stat)
                values.append(_value(key, _finite(a + step_weight * b, "advantage"),
                                     (("episode", a), ("step", b), ("return", returns[key])), stat))
            else:
                values.append(_value(key, None, (), stat))
    stats = tuple(sorted((*episode_stats.values(), *state_stats.values()), key=lambda s: s.key))
    return AdvantageResult("gigpo_episode_" + episode_weighting,
                           tuple(sorted(values, key=lambda v: v.key)), stats)
