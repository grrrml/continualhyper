"""Learned identifier tokens (option C): textual-inversion-style per-concept embeddings.

Each concept's identifier (e.g. "<V1>") is added to the CLIP tokenizer as a new token whose
input-embedding row is LEARNED during that concept's task (all other rows stay frozen; the
text-encoder transformer stays frozen). This makes both the pooled conditioning (hyper input)
and the cross-attn context genuinely discriminative between same-class concepts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def add_learned_tokens(bundle, idents_classes: List[Tuple[str, str]],
                       init_from_class: bool = True) -> Dict[str, int]:
    """Register identifier tokens in the tokenizer/text encoder; returns {token: row_id}.

    Only identifiers of the "<...>" form are registered. With `init_from_class` each new row
    is initialised from the first sub-token of the concept's class word (standard TI init).
    """
    pairs = [(t, c) for t, c in idents_classes if t.startswith("<") and t.endswith(">")]
    if not pairs:
        return {}
    tokens = [t for t, _ in pairs]
    bundle.tokenizer.add_tokens(tokens)
    bundle.text_encoder.resize_token_embeddings(len(bundle.tokenizer))
    emb = bundle.text_encoder.get_input_embeddings().weight
    emb.requires_grad_(False)   # the trainer re-enables (with row-masked grads) when training
    ids = bundle.tokenizer.convert_tokens_to_ids(tokens)
    if init_from_class:
        with torch.no_grad():
            for (tok, cls), tid in zip(pairs, ids):
                cls_ids = bundle.tokenizer(cls, add_special_tokens=False).input_ids
                emb[tid] = emb[cls_ids[0]].detach().clone()
    return dict(zip(tokens, ids))


@torch.no_grad()
def apply_learned_tokens(bundle, rows: Dict[str, torch.Tensor]) -> None:
    """Overwrite the embedding rows of registered identifier tokens (inference-time load)."""
    emb = bundle.text_encoder.get_input_embeddings().weight
    for tok, vec in rows.items():
        tid = bundle.tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == bundle.tokenizer.unk_token_id:
            raise RuntimeError(f"learned token {tok!r} not registered in the tokenizer")
        emb[tid] = vec.to(dtype=emb.dtype, device=emb.device)


def token_span_mask(tokenizer, prompts, phrase, max_length=None):
    """[B, L] float mask: 1.0 at the sub-token positions of `phrase` inside each prompt
    (all occurrences), 0.0 elsewhere. Used to apply the LoRA delta ONLY to the concept's
    tokens ("V7 dog") so the rest of the context (e.g. "in the swimming pool") keeps the
    base K/V -- recovers prompt-following without touching concept identity.

    Fallback: if the phrase does not occur in a prompt, that row is all-ones (old behavior).
    """
    max_length = max_length or tokenizer.model_max_length
    enc = tokenizer(prompts, padding="max_length", max_length=max_length,
                    truncation=True)["input_ids"]
    pat = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    mask = torch.zeros(len(prompts), max_length)
    if not pat:
        return mask + 1.0
    for b, ids in enumerate(enc):
        for i in range(max_length - len(pat) + 1):
            if ids[i:i + len(pat)] == pat:
                mask[b, i:i + len(pat)] = 1.0
        if mask[b].sum() == 0:
            mask[b] = 1.0
    return mask
