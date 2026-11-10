"""Mask entity names and numbers without placing digits inside placeholders."""
from __future__ import annotations

import re
from collections import Counter

NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")


def alpha_label(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, value = divmod(index - 1, 26)
        label = chr(65 + value) + label
    return label


def mask_triplet(row: dict) -> tuple[str, str, str]:
    texts = [row["text_en"], row["text_translation"], row["backtranslation_en"]]
    if any(re.search(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", text) for text in texts):
        raise ValueError("Input contains a reserved review placeholder")
    names = sorted({name for name in row["protected_entity_names"] if name and name in texts[0]},
                   key=lambda name: (texts[0].index(name), -len(name)))
    for name in names:
        if any(text.count(name) != texts[0].count(name) for text in texts[1:]):
            raise ValueError("Protected entity counts differ")
    mapping = {name: f"[ENTITY_{alpha_label(i)}]" for i, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        texts = [text.replace(name, mapping[name]) for text in texts]
    numbers = NUMBER.findall(texts[0])
    if any(Counter(NUMBER.findall(text)) != Counter(numbers) for text in texts[1:]):
        raise ValueError("Protected number counts differ")
    number_mapping = {number: f"[NUMBER_{alpha_label(i)}]" for i, number in enumerate(dict.fromkeys(numbers))}
    return tuple(NUMBER.sub(lambda match: number_mapping[match.group()], text) for text in texts)
