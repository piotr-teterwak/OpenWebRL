"""Scoring adapter: LoRA on the actor's top language layers, used only to score candidates.

Generation keeps the frozen SFT actor. At scoring time the prompt KV cache is the one
the actor already produced (adapter off); only candidate tokens pass through the adapted
top layers, so serving needs no second model and recomputes a fraction of one forward.
"""
from __future__ import annotations

import torch

from openwebrl.selection_head_features import encode_candidates, encode_prefix

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def language_layer_count(model):
    config = getattr(model.config, "text_config", model.config)
    return config.num_hidden_layers


def add_scoring_adapter(model, top_layers=8, rank=16, alpha=32, dropout=0.05):
    """Wrap `model` with LoRA on its last `top_layers` language layers only.

    Vision blocks use different module names (qkv/proj), so they are never adapted.
    """
    from peft import LoraConfig, get_peft_model
    total = language_layer_count(model)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                        layers_to_transform=list(range(total - top_layers, total)), layers_pattern="layers",
                        bias="none")
    model = get_peft_model(model, config)
    adapted = {n for n, _ in model.named_parameters() if "lora_" in n}
    if not adapted or any(".visual." in n for n in adapted):
        raise ValueError("Scoring adapter must touch only language layers")
    return model


def score_candidates(model, prefix_inputs, candidates, layers, pools, batch_size=8, pad_id=0):
    """State [L,H] and candidate features [N,L*P,H]; gradients reach only the adapter.

    The prefix is encoded with the adapter disabled and without grad, as at generation.
    """
    prefix_len = prefix_inputs["input_ids"].shape[1]
    with torch.no_grad(), model.disable_adapter():
        state, cache = encode_prefix(model, prefix_inputs, layers)
    features = encode_candidates(model, cache, prefix_len, candidates, layers, pools, batch_size, pad_id)
    return state, features
