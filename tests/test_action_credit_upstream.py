"""Optional CPU oracle parity against hash-pinned upstream source files.

No network calls and no upstream package imports occur in these tests. Set
ACTION_CREDIT_UPSTREAM_FIXTURE_DIR to a directory containing tree_core.py,
tree_trainer.py and gigpo.py from the exact commit/path pairs below. SHA256 is
checked BEFORE parsing/executing only selected function definitions. Large
third-party libraries are not vendored. Absent fixture/torch is an explicit skip,
not a passed upstream-reproduction check.
"""
import ast
from collections import defaultdict
import contextlib
import hashlib
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

from openwebrl.action_credit_advantages import (
    Episode, Outcome, Step, gigpo_advantages, outcome_group_advantages,
    tree_grpo_advantages,
)


# Source URL: https://raw.githubusercontent.com/{repository}/{commit}/{path}
SOURCES = {
    "tree_core": ("AMAP-ML/Tree-GRPO", "19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59",
                  "verl/trainer/ppo/core_algos.py",
                  "6d62c3a6c20e3621c9cddef1a5ab7252867489aa44028274304124bc37813363"),
    "tree_trainer": ("AMAP-ML/Tree-GRPO", "19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59",
                     "verl/trainer/ppo/ray_trainer_ts.py",
                     "be85b049d3d05fef367c8525d4a3a3342eb0b9efb570b843d16cd9625e818e39"),
    "gigpo": ("langfengQ/verl-agent", "20bd331bdbc9026a5668e11362178e10ab7400c8",
              "gigpo/core_gigpo.py",
              "fa709b91baf2331c4857fee32a249c4ca8d8d69e864d1e71f08730cc8c5cc456"),
}


def load_functions(directory, name, functions, namespace):
    path = Path(directory) / (name + ".py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCES[name][3]:
        raise ValueError(f"upstream source hash mismatch: {name}")
    nodes = [n for n in ast.parse(data, filename=str(path)).body
             if isinstance(n, ast.FunctionDef) and n.name in functions]
    if {n.name for n in nodes} != set(functions):
        raise ValueError("required upstream functions absent")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


@unittest.skipUnless(torch is not None and os.environ.get("ACTION_CREDIT_UPSTREAM_FIXTURE_DIR"),
                     "requires torch and explicit pinned offline upstream fixtures")
class PinnedUpstreamParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ["ACTION_CREDIT_UPSTREAM_FIXTURE_DIR"]
        base = dict(torch=torch, np=np, defaultdict=defaultdict,
                    DataProto=SimpleNamespace, uuid=uuid)
        cls.core = dict(base)
        cls.core["verl_F"] = SimpleNamespace(masked_mean=lambda x, m: (x * m).sum() / m.sum())
        load_functions(directory, "tree_core", {"compute_grpo_outcome_advantage",
                       "compute_policy_loss_dual_clip", "agg_loss"}, cls.core)
        cls.trainer = dict(base, core_algos=SimpleNamespace(**cls.core))
        load_functions(directory, "tree_trainer", {"compute_advantage"}, cls.trainer)
        cls.gigpo = dict(base)
        load_functions(directory, "gigpo", {"compute_step_discounted_returns", "to_hashable",
                       "build_step_group", "compute_gigpo_outcome_advantage", "episode_norm_reward",
                       "step_norm_reward"}, cls.gigpo)

    def assert_values(self, result, upstream, keys):
        ours = {v.key: v.advantage for v in result.values}
        self.assertEqual(set(ours), set(keys))
        for key, value in zip(keys, upstream[:, 0].tolist()):
            self.assertAlmostEqual(ours[key], value, delta=2e-6, msg=str(key))

    def test_outcome_actual_upstream_function(self):
        for rewards in ([0., 1.], [1.], [0.], [1., 1., 1.], [0., 0.], [0., .25, 1.]):
            with self.subTest(rewards=rewards):
                rows = [Outcome(str(i), "t", "g", r) for i, r in enumerate(rewards)]
                upstream, _ = self.core["compute_grpo_outcome_advantage"](
                    torch.tensor(rewards)[:, None], torch.ones((len(rows), 1)), np.array(["g"]*len(rows)))
                ours = outcome_group_advantages(rows, expected_ids=[r.rollout_id for r in rows])
                self.assert_values(ours, upstream, [(r.rollout_id,) for r in rows])

    def test_tree_actual_trainer_dispatch_not_two_stage_ablation(self):
        cases = [([0, 1, 1, 1], ["a", "a", "b", "b"]),
                 ([0, 1, 1, 1], ["a", "b", "b", "b"]), ([1], ["a"]),
                 ([0, 0, 0], ["a", "a", "b"])]
        for rewards, trees in cases:
            with self.subTest(rewards=rewards, trees=trees):
                rows = [Outcome(str(i), "t", "g", r, tree_id=tree)
                        for i, (r, tree) in enumerate(zip(rewards, trees))]
                data = SimpleNamespace(batch={"token_level_rewards": torch.tensor(rewards, dtype=torch.float32)[:, None],
                        "responses": torch.ones((len(rows), 1)), "attention_mask": torch.ones((len(rows), 1))},
                        non_tensor_batch={"uid": np.array(["g"]*len(rows)), "tree_uid": np.array(trees)})
                self.trainer["compute_advantage"](data, "tree")
                ours = tree_grpo_advantages(rows, expected_ids=[r.rollout_id for r in rows])
                self.assert_values(ours, data.batch["advantages"], [(r.rollout_id,) for r in rows])

    def test_token_mean_dual_clip_loss_and_shared_prefix_gradient_in_normal_ratio_range(self):
        # The upstream source clamps log ratios to [-20,20]; this parity check
        # intentionally does not claim equivalence outside that interval.
        from openwebrl.action_credit import (
            CollectionContract, Leaf, ResponseEdge, ResponseGraph, clipped_loss,
            digest, leaf_loss_plan,
        )
        contract = CollectionContract(0, *[digest(x) for x in
            ("policy", "tokenizer", "processor", "tasks", "judge", "generation")],
            .8, ("task",), ("test",))
        def edge(key, parent):
            return ResponseEdge(key, "task", "tree", parent, contract.sha256,
                (0, 1), (), (2, 3), (True, True), (-2., -2.), contract.policy_sha256,
                digest("state" + str(parent)), digest("CPU synthetic receipt"),
                30 if parent is None else 29)
        graph = ResponseGraph(contract, (edge("prefix", None), edge("good", "prefix"), edge("bad", "prefix")),
            (Leaf("a", "task", "tree", ("prefix", "good"), "valid", 1., group_id="g"),
             Leaf("b", "task", "tree", ("prefix", "bad"), "valid", 0., group_id="g")), ("a", "b"))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="token_mean")
        for ratio in (.6, 1., 1.5, 4.):
            with self.subTest(ratio=ratio):
                values = {key: torch.full((2,), -2 + math.log(ratio), dtype=torch.float64, requires_grad=True)
                          for key in ("prefix", "good", "bad")}
                actual = clipped_loss(plan, values, clip_low=.2, clip_high=.28, dual_clip=3.)
                leaves = torch.stack([values[k] for k in ("prefix", "good", "prefix", "bad")])
                expected, *_ = self.core["compute_policy_loss_dual_clip"](
                    torch.full((4, 2), -2., dtype=torch.float64), leaves,
                    torch.tensor([[1.], [1.], [-1.], [-1.]], dtype=torch.float64).expand(4, 2),
                    torch.ones((4, 2)), cliprange_low=.2, cliprange_high=.28,
                    clip_ratio_c=3., loss_agg_mode="token-mean")
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
                ag = torch.autograd.grad(actual, tuple(values.values()), retain_graph=True)
                eg = torch.autograd.grad(expected, tuple(values.values()))
                for a, e in zip(ag, eg):
                    torch.testing.assert_close(a, e, rtol=0, atol=1e-12)

    def make_episode_data(self, episodes):
        rows = [(e, s) for e in episodes for s in sorted(e.steps, key=lambda x: x.position)]
        data = SimpleNamespace(batch={"input_ids": torch.ones((len(rows), 1), dtype=torch.long)},
            non_tensor_batch={"rewards": np.array([s.reward for e, s in rows]),
                              "traj_uid": np.array([e.trajectory_id for e, s in rows]),
                              "active_masks": np.ones(len(rows))})
        returns = self.gigpo["compute_step_discounted_returns"](data, .5)
        kwargs = dict(token_level_rewards=torch.tensor([e.reward for e, s in rows])[:, None],
                      step_rewards=returns, response_mask=torch.ones((len(rows), 1)),
                      anchor_obs=np.array([s.state_key for e, s in rows]),
                      index=np.array([e.group_id for e, s in rows]),
                      traj_index=np.array([e.trajectory_id for e, s in rows]))
        return rows, returns, kwargs

    def episode_cases(self):
        def ep(identity, rs, states):
            return Episode(identity, "t", "g", float(sum(rs)),
                           tuple(Step(str(i), s, float(r), i) for i, (r, s) in enumerate(zip(rs, states))))
        return [[ep("a", [0, 1], ["start", "end"]), ep("b", [0, 0], ["start", "end"])],
                [ep("short", [1], ["same"]), ep("long", [0, 0, 0], ["same"]*3)],
                [ep("single", [1], ["one"])], [ep("repeat", [0, 1], ["same"]*2)],
                [ep("zero", [0, 0], ["start", "end"])]]

    def test_gigpo_actual_upstream_default_step_weighting_both_normalizers(self):
        for episodes in self.episode_cases():
            rows, returns, kwargs = self.make_episode_data(episodes)
            for mode, normalization in (("mean_norm", "mean"), ("mean_std_norm", "mean_std")):
                with self.subTest(episodes=episodes, mode=mode), contextlib.redirect_stdout(io.StringIO()):
                    upstream, _ = self.gigpo["compute_gigpo_outcome_advantage"](**kwargs, mode=mode)
                    ours = gigpo_advantages(episodes, expected_ids=[e.trajectory_id for e in episodes], gamma=.5,
                                            normalization=normalization, episode_weighting="step")
                    self.assert_values(ours, upstream, [(e.trajectory_id, s.step_id) for e, s in rows])
                    got = {v.key: dict(v.components)["return"] for v in ours.values}
                    for (e, s), ret in zip(rows, returns.tolist()):
                        self.assertAlmostEqual(got[(e.trajectory_id, s.step_id)], ret, delta=2e-6)

    def test_gigpo_actual_upstream_equal_trajectory_option(self):
        for episodes in self.episode_cases():
            rows, returns, k = self.make_episode_data(episodes)
            with contextlib.redirect_stdout(io.StringIO()):
                groups = self.gigpo["build_step_group"](k["anchor_obs"], k["index"])
            for normalization in ("mean", "mean_std"):
                remove_std = normalization == "mean"
                episode = self.gigpo["episode_norm_reward"](k["token_level_rewards"], k["response_mask"],
                    k["index"], k["traj_index"], remove_std=remove_std, compute_mean_std_cross_steps=False)
                step = self.gigpo["step_norm_reward"](returns, k["response_mask"], groups, remove_std=remove_std)
                ours = gigpo_advantages(episodes, expected_ids=[e.trajectory_id for e in episodes], gamma=.5,
                                        normalization=normalization, episode_weighting="trajectory")
                self.assert_values(ours, episode + step, [(e.trajectory_id, s.step_id) for e, s in rows])


if __name__ == "__main__":
    unittest.main()
