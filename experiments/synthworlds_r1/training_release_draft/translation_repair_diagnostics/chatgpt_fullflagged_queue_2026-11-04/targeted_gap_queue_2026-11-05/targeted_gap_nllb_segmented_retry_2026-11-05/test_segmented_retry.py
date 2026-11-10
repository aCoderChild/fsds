#!/usr/bin/env python3
"""Offline checks for the actual marker-span helper in the Kaggle template."""
import ast
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    for filename in ("prepare_segmented_retry.py", "segmented_retry_template.py", "segmented_retry.py", "verify_segmented_retry.py"):
        ast.parse((HERE / filename).read_text(encoding="utf-8"), filename=filename)

    tree = ast.parse((HERE / "segmented_retry_template.py").read_text(encoding="utf-8"))
    marker_assign = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MARKER" for target in node.targets)
    )
    pattern = ast.literal_eval(marker_assign.value.args[0])
    helper_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "translate_marker_spans"
    )
    namespace = {"MARKER": re.compile(pattern)}
    module = ast.Module(body=[helper_node], type_ignores=[])
    exec(compile(module, "<actual-segmented-retry-template>", "exec"), namespace)
    helper = namespace["translate_marker_spans"]

    source = "Who is [ENTITY_A]'s sibling of [ENTITY_B] on [NUMBER_C]?"
    calls = []
    candidate = helper(source, lambda fragment: calls.append(fragment) or fragment.upper())
    assert candidate == "WHO IS [ENTITY_A]'S SIBLING OF [ENTITY_B] ON [NUMBER_C]?", candidate
    assert calls == ["Who is", "'s sibling of", "on", "?"], calls
    assert [match.group(0) for match in namespace["MARKER"].finditer(candidate)] == [
        "[ENTITY_A]", "[ENTITY_B]", "[NUMBER_C]"
    ]
    try:
        helper("Named entity without a placeholder?", str.upper)
    except ValueError as exc:
        assert "no protected entity placeholder" in str(exc)
    else:
        raise AssertionError("Marker-free question must fail closed")

    payload = json.loads((HERE / "segmented_retry_input.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "segmented_retry_input_manifest.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "AI_ONLY_DIAGNOSTIC_SEGMENTED_RETRY_NO_RELEASE_APPROVAL"
    assert len(payload["rows"]) == manifest["rows"] == 125
    assert len({row["review_key"] for row in payload["rows"]}) == 125
    assert dict(collections.Counter(row["language"] for row in payload["rows"])) == {
        "de": 2, "hi": 52, "th": 71
    }
    assert payload["answers_or_split_labels_included"] is False
    assert payload["human_translation_reviewed"] is False
    assert payload["release_approved"] is False and payload["training_eligible"] is False
    print("PASS: marker segmentation, marker-free fail-closed path, input scope, exact row selection, and script syntax")


if __name__ == "__main__":
    main()
