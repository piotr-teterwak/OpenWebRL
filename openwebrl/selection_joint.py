"""Joint top-layer scorer: Kev-style comparison of the five candidates inside the actor.

Lower layers stay exactly as the frozen actor computed them (one candidate at a time after the
cached prompt). The top `top_layers` language layers (LoRA-adapted) are then run once over
[candidate 1 … candidate 5, <decide>], packed with a custom mask:

* each candidate attends to the prompt and causally to itself only (its states equal the
  one-at-a-time adapter path);
* <decide> attends to the prompt and every candidate token, so comparison happens in all
  adapted layers;
* every candidate reuses the same position ids and <decide> sits after the longest one, so
  scores cannot depend on candidate order (Kev's option isolation; possible here because the
  actor is attention-only).

Scores = warm-started pooled head (identical to the one-at-a-time head at initialization)
+ gate × pointer(<decide>, candidate end token), with the gate initialized to zero.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from openwebrl.selection_head_features import pool_candidate


def _parts(model):
    """(Qwen3VLModel, text model) under an optional PEFT wrapper."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.model.language_model


class JointScorer(nn.Module):
    def __init__(self, model, head, top_layers, pointer_dim=256):
        super().__init__()
        vl, text = _parts(model)
        hidden = text.config.hidden_size
        self.model, self.head = model, head
        self.total = text.config.num_hidden_layers
        self.start = self.total - top_layers           # index of the first adapted layer
        self.layers, self.pools = head.config.layers, head.config.pools
        if not all(l <= self.start or l == self.total for l in self.layers):
            raise ValueError("Feature layers must be below the adapted stack or its final output")
        self.decide_offset = nn.Parameter(torch.zeros(hidden))
        self.q, self.k = nn.Linear(hidden, pointer_dim), nn.Linear(hidden, pointer_dim)
        self.gate = nn.Parameter(torch.zeros(()))
        self.scale = pointer_dim ** -0.5

    # ---------------------------------------------------------------- lower part (frozen)
    @torch.no_grad()
    def _lower(self, prefix_inputs, candidates, batch_size, pad_id):
        """Prompt cache and per-candidate states entering the adapted stack (adapter off)."""
        model, (vl, _) = self.model, _parts(self.model)
        device = prefix_inputs["input_ids"].device
        P = prefix_inputs["input_ids"].shape[1]
        keep = sorted({*[l for l in self.layers if l <= self.start], self.start})
        with model.disable_adapter():
            out = model(**prefix_inputs, use_cache=True, output_hidden_states=True,
                        cache_position=torch.arange(P, device=device))
            state = {l: out.hidden_states[l][0, -1] for l in {*self.layers, self.start}}
            cache, delta = out.past_key_values, vl.rope_deltas
            states = []
            for s in range(0, len(candidates), batch_size):
                chunk = candidates[s:s + batch_size]
                width = max(len(ids) for ids, _ in chunk)
                ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
                for r, (tok, _) in enumerate(chunk):
                    ids[r, :len(tok)] = torch.tensor(tok, device=device)
                past = copy.deepcopy(cache)
                if len(chunk) > 1:
                    past.batch_repeat_interleave(len(chunk))
                step = model(input_ids=ids, past_key_values=past, use_cache=True, output_hidden_states=True,
                             cache_position=torch.arange(P, P + width, device=device))
                for r, (tok, _) in enumerate(chunk):
                    states.append({l: step.hidden_states[l][r, :len(tok)] for l in keep})
                del past, step
        return state, cache, delta, P, states

    # ---------------------------------------------------------------- adapted joint stack
    def _joint(self, cache, delta, P, panel_states, decide_in):
        """Run the adapted layers over [cand_1 … cand_n, <decide>].

        Returns per-candidate final-layer states before the final norm (as in the extracted
        features the head was trained on), and normed candidate end / <decide> states for the pointer.
        """
        vl, text = _parts(self.model)
        device = decide_in.device
        lengths = [s.shape[0] for s in panel_states]
        T, longest = sum(lengths) + 1, max(lengths)
        x = torch.cat([*panel_states, decide_in[None]], 0)[None]
        offsets = torch.cat([torch.arange(n, device=device) for n in lengths] +
                            [torch.tensor([longest], device=device)])
        position = (P + offsets + delta.to(device).reshape(-1)[0]).view(1, 1, T).expand(3, 1, T)
        owner = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(lengths)] +
                          [torch.tensor([-1], device=device)])
        q_owner, k_owner = owner[:, None], owner[None, :]
        q_off, k_off = offsets[:, None], offsets[None, :]
        allow = ((q_owner == k_owner) & (k_off <= q_off)) | (q_owner == -1)   # own causal span; <decide> sees all
        allow = torch.cat([torch.ones(T, P, dtype=torch.bool, device=device), allow], 1)
        dtype = x.dtype
        mask = torch.zeros(T, P + T, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[None, None]
        past = copy.deepcopy(cache)
        embeddings = text.rotary_emb(x, position)
        cache_position = torch.arange(P, P + T, device=device)
        for layer in text.layers[self.start:]:
            x = layer(x, attention_mask=mask, position_ids=position[0], past_key_values=past,
                      cache_position=cache_position, position_embeddings=embeddings)
        normed = text.norm(x)[0]
        x = x[0]
        bounds = torch.tensor([0, *lengths]).cumsum(0).tolist()
        spans = list(zip(bounds[:-1], bounds[1:]))
        return [x[a:b] for a, b in spans], torch.stack([normed[b - 1] for _, b in spans]), normed[-1]

    def forward(self, prefix_inputs, panels, batch_size=8, pad_id=0, return_pointer=False):
        """panels: list of candidate lists [(ids, spans)] (one list per panel) -> logits [len(panels), n].

        return_pointer also returns the pointer's own logits: the gate starts at zero, so the pointer
        needs its own loss to learn before the gate has a reason to open.
        """
        flat = [c for panel in panels for c in panel]
        state, cache, delta, P, lower = self._lower(prefix_inputs, flat, batch_size, pad_id)
        state_vec = torch.stack([state[l] for l in self.layers]).reshape(1, -1)
        decide_in = state[self.start] + self.decide_offset.to(state[self.start].dtype)
        logits, pointers, start = [], [], 0
        for panel in panels:
            rows = lower[start:start + len(panel)]; start += len(panel)
            top, ends, decide = self._joint(cache, delta, P, [r[self.start] for r in rows], decide_in)
            feats = []
            for (ids, spans), low, hi in zip(panel, rows, top):
                hidden = {**low, self.total: hi}
                feats.append(pool_candidate(hidden, spans, self.layers, self.pools))
            feats = torch.stack(feats).reshape(1, len(panel), -1)
            pooled = self.head(feats, state_vec)[0][0]
            pointer = (self.k(ends.float()) @ self.q(decide.float())) * self.scale
            logits.append(pooled + self.gate * pointer); pointers.append(pointer)
        return (torch.stack(logits), torch.stack(pointers)) if return_pointer else torch.stack(logits)
