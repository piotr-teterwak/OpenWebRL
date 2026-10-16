import unittest

import torch

from openwebrl.selection_adapter import add_scoring_adapter
from openwebrl.selection_kev import KevStyleScorer, _vl
from tests.test_selection_adapter import prefix_inputs
from tests.test_selection_head import tiny_qwen3_vl

DECIDE = 151662
PANEL = [([200, 201, 202, 203], None), ([300, 301], None), ([400, 401, 402], None)]
OTHER = [([500, 501], None), ([600, 601, 602, 603, 604], None), ([700], None)]


def build(randomize=False):
    torch.manual_seed(0)
    model = add_scoring_adapter(tiny_qwen3_vl(), top_layers=3, rank=4, alpha=8, dropout=0.)
    scorer = KevStyleScorer(model, DECIDE, pointer_dim=8)
    if randomize:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.3)
            scorer.decide_delta.normal_()
    return model, scorer


def alone_end_state(model, candidate):
    """Final normed state of a candidate's last token when the actor reads prompt + candidate only."""
    vl, inputs = _vl(model), prefix_inputs()
    ids = torch.cat([inputs["input_ids"], torch.tensor([candidate])], 1)
    out = vl(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inputs["pixel_values"],
             image_grid_thw=inputs["image_grid_thw"], use_cache=False)
    return out.last_hidden_state[0, -1]


class KevStyle(unittest.TestCase):
    def test_lora_spans_every_language_layer_but_no_vision_block(self):
        model, _ = build()
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        for layer in range(3):
            self.assertTrue(any(f".layers.{layer}." in n for n in names), layer)
        self.assertFalse(any(".visual." in n for n in names))

    def test_candidates_are_isolated_with_generation_positions(self):
        model, scorer = build(randomize=True)
        inputs = prefix_inputs()
        input_ids, position, allow, decide_at, ends = scorer.pack(inputs, [PANEL, OTHER])
        dtype = next(model.parameters()).dtype
        mask = torch.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)
        hidden = _vl(model)(input_ids=input_ids, attention_mask=mask[None, None], position_ids=position,
                            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                            use_cache=False).last_hidden_state[0]
        for panel, panel_ends in zip([PANEL, OTHER], ends):
            for (tok, _), end in zip(panel, panel_ends):
                torch.testing.assert_close(hidden[end], alone_end_state(model, tok), atol=1e-4, rtol=1e-4)

    def test_order_and_packing_cannot_change_scores(self):
        _, scorer = build(randomize=True)
        both = scorer(prefix_inputs(), [PANEL, OTHER])
        torch.testing.assert_close(both[0], scorer(prefix_inputs(), [PANEL])[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(both[1], scorer(prefix_inputs(), [OTHER])[0], atol=1e-4, rtol=1e-4)
        perm = [2, 0, 1]
        permuted = scorer(prefix_inputs(), [[PANEL[i] for i in perm]])[0]
        torch.testing.assert_close(permuted, both[0][perm], atol=1e-4, rtol=1e-4)

    def test_gradients_reach_lora_pointer_and_decide_only(self):
        model, scorer = build()
        with torch.no_grad():
            scorer.q.weight.normal_(); scorer.k.weight.normal_()
        scorer(prefix_inputs(), [PANEL, OTHER]).sum().backward()
        for name in ("decide_delta", "q.weight", "k.weight"):
            self.assertIsNotNone(dict(scorer.named_parameters())[name].grad, name)
        lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
        self.assertTrue(lora_b and all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))


class KevStyleBf16(unittest.TestCase):
    def test_runs_in_bfloat16_like_the_real_actor(self):
        torch.manual_seed(0)
        model = add_scoring_adapter(tiny_qwen3_vl().to(torch.bfloat16), top_layers=3, rank=4, alpha=8, dropout=0.)
        scorer = KevStyleScorer(model, DECIDE, pointer_dim=8)
        inputs = prefix_inputs(); inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        logits = scorer(inputs, [PANEL, OTHER])
        logits.sum().backward()
        self.assertEqual(logits.shape, (2, 3))
        self.assertIsNotNone(scorer.decide_delta.grad)


if __name__ == "__main__":
    unittest.main()
