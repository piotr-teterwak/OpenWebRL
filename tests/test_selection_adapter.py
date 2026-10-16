import unittest

import torch

from openwebrl.selection_adapter import add_scoring_adapter, score_candidates
from openwebrl.selection_head_features import extract
from tests.test_selection_head import tiny_qwen3_vl


def prefix_inputs():
    ids = [100, 101, 151652] + [151655] * 4 + [151653, 102, 103]
    torch.manual_seed(1)
    return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.randn(16, 3 * 2 * 16 * 16), "image_grid_thw": torch.tensor([[1, 4, 4]])}


CANDS = [([200, 201, 202, 203], {"thought": (0, 2), "action": (2, 3)}),
         ([300, 301], {"thought": None, "action": (0, 1)})]


class ScoringAdapter(unittest.TestCase):
    def setUp(self):
        self.base = tiny_qwen3_vl()
        _, self.frozen = extract(self.base, prefix_inputs(), CANDS, layers=(1, 3), batch_size=2)
        self.model = add_scoring_adapter(tiny_qwen3_vl(), top_layers=1, rank=4, alpha=8, dropout=0.)

    def test_adapter_starts_as_identity_and_touches_only_top_language_layer(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.assertTrue(names and all("lora_" in n and ".layers.2." in n for n in names), names)
        _, feats = score_candidates(self.model, prefix_inputs(), CANDS, (1, 3), ("thought_mean", "action_mean", "last"),
                                    batch_size=2)
        torch.testing.assert_close(feats, self.frozen, atol=1e-5, rtol=1e-4)

    def test_prefix_is_unadapted_and_gradients_reach_only_the_adapter(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.5)
        state, feats = score_candidates(self.model, prefix_inputs(), CANDS, (1, 3),
                                        ("thought_mean", "action_mean", "last"), batch_size=2)
        base_state, _ = extract(self.base, prefix_inputs(), CANDS, layers=(1, 3))
        torch.testing.assert_close(state, base_state)          # prompt encoded with the adapter off
        # Layer 1 sits below the adapted layer: unchanged. Layer 3 output: changed.
        torch.testing.assert_close(feats[:, :3], self.frozen[:, :3], atol=1e-5, rtol=1e-4)
        self.assertGreater((feats[:, 3:] - self.frozen[:, 3:]).abs().max().item(), 1e-3)
        feats.sum().backward()
        grads = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in grads.values()))
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if not p.requires_grad))


if __name__ == "__main__":
    unittest.main()
