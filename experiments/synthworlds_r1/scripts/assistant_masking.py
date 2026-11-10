#!/usr/bin/env python3
"""Chat-template encoding with a fail-closed assistant-only loss mask."""
from __future__ import annotations


def encode_assistant_only(tokenizer, messages):
    """Encode once, masking tokens that overlap the prompt by character offsets.

    Separately tokenizing the prompt and completed conversation is unsafe for
    subword tokenizers: the final prompt token may change when answer text is
    appended. We instead render both forms, verify their textual prefix, and
    tokenize the complete rendered conversation once with offsets.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected a conversation with a prompt and assistant answer")
    if messages[-1].get("role") != "assistant":
        raise ValueError("The final chat message must be an assistant answer")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        raise ValueError("Tokenizer chat template did not return rendered strings")
    if not full_text.startswith(prompt_text):
        raise ValueError("Rendered prompt is not a prefix of the full conversation")

    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except (TypeError, KeyError, NotImplementedError) as exc:
        raise ValueError(
            "A fast tokenizer with offset_mapping support is required for safe assistant-only masking"
        ) from exc

    # Hugging Face may return a tensor/array in some tokenizer configurations;
    # this routine deliberately requests ordinary, unbatched Python lists.
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    input_ids = [int(token_id) for token_id in input_ids]
    offsets = [(int(start), int(end)) for start, end in offsets]
    if len(input_ids) != len(offsets) or not input_ids:
        raise ValueError("Tokenizer returned missing or misaligned token offsets")

    boundary = len(prompt_text)
    labels = [token_id if start >= boundary else -100 for token_id, (start, _end) in zip(input_ids, offsets)]
    if not any(label != -100 for label in labels):
        raise ValueError("No assistant answer tokens remain after applying the loss mask")
    return input_ids, labels
