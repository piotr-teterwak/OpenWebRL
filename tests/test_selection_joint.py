import unittest

import torch

from openwebrl.selection_adapter import add_scoring_adapter, score_candidates
from openwebrl.selection_head import CandidateSetHead, HeadConfig
from openwebrl.selection_joint import JointScorer
from tests.test_selection_adapter import prefix_inputs
from tests.test_selection_head import tiny_qwen3_vl

POOLS = ("thought_mean", "action_mean", "last")
PANEL = [([200, 201, 202, 203], {"thought": (0, 2), "action": (2, 3)}),
         ([300, 301], {"thought": None, "action": (0, 1)}),
         ([400, 401, 402], {"thought": (0, 1), "action": (1, 2)})]


def build():
    torch.manual_seed(0)
    model = add_scoring_adapter(tiny_qwen3_vl(), top_layers=1, rank=4, alpha=8, dropout=0.)
    head = CandidateSetHead(HeadConfig(hidden_size=64, layers=(1, 3), d_model=16, n_heads=4, dropout=0., listwise=False)).eval()
    return model, head, JointScorer(model, head, top_layers=1, pointer_dim=8)


class Joint(unittest.TestCase):
    def test_starts_exactly_at_the_one_at_a_time_head(self):
        model, head, scorer = build()
        logits = scorer(prefix_inputs(), [PANEL], batch_size=2)
        state, feats = score_candidates(model, prefix_inputs(), PANEL, (1, 3), POOLS, batch_size=2)
        expected = head(feats.reshape(1, 3, -1), state.reshape(1, -1))[0]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)

    def test_candidates_stay_isolated_and_order_cannot_matter(self):
        model, head, scorer = build()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.5)
            scorer.gate.fill_(1.0); scorer.decide_offset.normal_()
        # Isolation: with the adapter on, joint candidate states equal one-at-a-time encoding.
        _, feats = score_candidates(model, prefix_inputs(), PANEL, (1, 3), POOLS, batch_size=1)
        state, cache, delta, P, lower = scorer._lower(prefix_inputs(), PANEL, 2, 0)
        top, _, _ = scorer._joint(cache, delta, P, [r[scorer.start] for r in lower], state[scorer.start])
        for i, (ids, spans) in enumerate(PANEL):
            joint = top[i]
            self.assertEqual(joint.shape[0], len(ids))
            torch.testing.assert_close(joint[-1], feats[i, -1], atol=1e-4, rtol=1e-4)  # layer-3 end token
        # Permuting candidates permutes the scores and nothing else.
        logits = scorer(prefix_inputs(), [PANEL])[0]
        perm = [2, 0, 1]
        permuted = scorer(prefix_inputs(), [[PANEL[i] for i in perm]])[0]
        torch.testing.assert_close(permuted, logits[perm], atol=1e-4, rtol=1e-4)

    def test_gradients_reach_adapter_pointer_and_head_only(self):
        model, head, scorer = build()
        with torch.no_grad():
            scorer.gate.fill_(0.5)
        scorer(prefix_inputs(), [PANEL, PANEL[::-1]]).sum().backward()
        trainable = {n: p for n, p in scorer.named_parameters() if p.requires_grad}
        for name in ("gate", "decide_offset", "q.weight", "k.weight"):
            self.assertIsNotNone(trainable[name].grad, name)
        lora = [p for n, p in trainable.items() if "lora_" in n]
        self.assertTrue(lora and all(p.grad is not None for p in lora))
        frozen = [n for n, p in scorer.named_parameters() if not p.requires_grad and p.grad is not None]
        self.assertEqual(frozen, [])


class PointerLearnsWhileGated(unittest.TestCase):
    def test_auxiliary_pointer_loss_reaches_the_pointer_at_zero_gate(self):
        from openwebrl.selection_head import selection_loss
        _, _, scorer = build()
        logits, pointer = scorer(prefix_inputs(), [PANEL], return_pointer=True)
        target = torch.tensor([[0., 1., 0.]]); every = torch.ones_like(target, dtype=torch.bool)
        selection_loss(logits, target, every).backward(retain_graph=True)
        self.assertEqual(scorer.q.weight.grad.abs().sum().item(), 0.0)   # gated main loss alone: no signal
        selection_loss(pointer, target, every).backward()
        self.assertGreater(scorer.q.weight.grad.abs().sum().item(), 0.0)


class JointBf16(unittest.TestCase):
    def test_runs_in_bfloat16_like_the_real_actor(self):
        torch.manual_seed(0)
        model = add_scoring_adapter(tiny_qwen3_vl().to(torch.bfloat16), top_layers=1, rank=4, alpha=8, dropout=0.)
        head = CandidateSetHead(HeadConfig(hidden_size=64, layers=(1, 3), d_model=16, n_heads=4, dropout=0., listwise=False))
        scorer = JointScorer(model, head, top_layers=1, pointer_dim=8)
        inputs = prefix_inputs(); inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        logits = scorer(inputs, [PANEL, PANEL[::-1]], batch_size=2)
        logits.float().sum().backward()
        self.assertEqual(logits.shape, (2, 3))
        self.assertIsNotNone(scorer.decide_offset.grad)


if __name__ == "__main__":
    unittest.main()
