"""Kev-style decision pass on the actor's own weights (a separate selector sharing the base).

Mirrors Kev (jaredpalmer/kev, option-isolation mode) on our data: one packed sequence

    [prompt + screenshot] [panel 1: cand_1 … cand_5, <decide>] [panel 2: …] …

runs through every layer with LoRA on all language layers. Masks: the prompt is causal; each
candidate sees the prompt and itself (causally); each panel's <decide> sees the prompt and its own
panel's candidates; panels never see each other (Kev's isolated questions). Every candidate, and
every panel, restarts at the same positions after the prompt, so candidate order cannot matter.
Readout is Kev's pointer: q(<decide>) · k(candidate end token) on the final normed states.

<decide> is an otherwise unused special token whose embedding gets a trainable offset.
Unlike the joint top-layer scorer, nothing is reused from generation and there is no warm start.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _vl(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model


class KevStyleScorer(nn.Module):
    def __init__(self, model, decide_id, pointer_dim=256):
        super().__init__()
        vl = _vl(model)
        hidden = vl.language_model.config.hidden_size
        self.model, self.decide_id = model, decide_id
        self.decide_delta = nn.Parameter(torch.zeros(hidden))
        self.q, self.k = nn.Linear(hidden, pointer_dim), nn.Linear(hidden, pointer_dim)
        self.scale = pointer_dim ** -0.5
        vl.language_model.embed_tokens.register_forward_hook(self._decide_embedding)

    def _decide_embedding(self, module, inputs, output):
        marks = (inputs[0] == self.decide_id).unsqueeze(-1)
        return output + marks.to(output.dtype) * self.decide_delta.to(output.dtype)

    def pack(self, prefix_inputs, panels):
        """Token ids, 3-D positions, additive mask and readout indices for one state."""
        vl = _vl(self.model)
        prompt = prefix_inputs["input_ids"][0]
        device, P = prompt.device, prompt.shape[0]
        pos, delta = vl.get_rope_index(prefix_inputs["input_ids"], prefix_inputs.get("image_grid_thw"), None,
                                       prefix_inputs.get("attention_mask"))
        ids, offsets, panel_of, cand_of = [], [], [], []
        decide_at, ends = [], []
        for p, panel in enumerate(panels):
            longest = max(len(tok) for tok, _ in panel)
            panel_ends = []
            for c, (tok, _) in enumerate(panel):
                ids += tok; offsets += range(len(tok)); panel_of += [p] * len(tok); cand_of += [c] * len(tok)
                panel_ends.append(P + len(ids) - 1)
            ids.append(self.decide_id); offsets.append(longest); panel_of.append(p); cand_of.append(-1)
            decide_at.append(P + len(ids) - 1); ends.append(panel_ends)
        T = len(ids)
        off = torch.tensor(offsets, device=device)
        text = (P + off + delta.to(device).reshape(-1)[0]).view(1, 1, T).expand(3, 1, T)
        position = torch.cat([pos.to(device), text], -1)
        pan, cand = torch.tensor(panel_of, device=device), torch.tensor(cand_of, device=device)
        same_panel = pan[:, None] == pan[None, :]
        own = same_panel & (cand[:, None] == cand[None, :]) & (off[None, :] <= off[:, None])
        decide_row = (cand[:, None] == -1) & same_panel
        tail = own | decide_row
        allow = torch.zeros(P + T, P + T, dtype=torch.bool, device=device)
        allow[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))
        allow[P:, :P] = True
        allow[P:, P:] = tail
        input_ids = torch.cat([prompt, torch.tensor(ids, device=device)])[None]
        return input_ids, position, allow, decide_at, ends

    def forward(self, prefix_inputs, panels):
        """panels: list of candidate lists [(ids, spans)] -> logits [len(panels), n]."""
        vl = _vl(self.model)
        input_ids, position, allow, decide_at, ends = self.pack(prefix_inputs, panels)
        dtype = next(self.model.parameters()).dtype
        mask = torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)
        extra = {k: v for k, v in prefix_inputs.items() if k not in ("input_ids", "attention_mask")}
        hidden = vl(input_ids=input_ids, attention_mask=mask[None, None], position_ids=position,
                    use_cache=False, **extra).last_hidden_state[0].float()
        logits = []
        for d, panel_ends in zip(decide_at, ends):
            logits.append((self.k(hidden[panel_ends]) @ self.q(hidden[d])) * self.scale)
        return torch.stack(logits)
