"""Production-rule animal recognizer using interpretable image statistics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Rule:
    name: str
    conditions: frozenset[str]
    conclusion: str
    weight: float


RULES = [
    Rule("R1", frozenset({"smooth_texture", "horizontal_body"}), "candidate_cat", 0.55),
    Rule("R2", frozenset({"warm_color", "slender_shape"}), "candidate_dog", 0.56),
    Rule("R3", frozenset({"blue_background", "low_center_contrast"}), "candidate_deer", 0.70),
    Rule("R4", frozenset({"rough_texture", "horizontal_body"}), "candidate_horse", 0.72),
    Rule("R5", frozenset({"candidate_cat", "center_bright"}), "cat", 0.85),
    Rule("R6", frozenset({"candidate_dog", "center_bright"}), "dog", 0.82),
    Rule("R7", frozenset({"candidate_deer", "smooth_texture"}), "deer", 0.88),
    Rule("R8", frozenset({"candidate_horse", "center_bright"}), "horse", 0.88),
    Rule("R9", frozenset({"warm_color", "center_bright", "smooth_texture"}), "cat", 0.68),
    Rule("R10", frozenset({"cool_color", "high_edge", "rough_texture"}), "dog", 0.66),
]


def image_statistics(path: str | Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB").resize((64, 64))
    x = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    center = gray[16:48, 16:48].mean()
    border = np.concatenate([gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()])
    return {
        "red": float(x[..., 0].mean()),
        "green": float(x[..., 1].mean()),
        "blue": float(x[..., 2].mean()),
        "edge": float(gx + gy),
        "texture": float(gray.std()),
        "center_contrast": float(center - border.mean()),
        "horizontal_ratio": float((gx + 1e-6) / (gy + 1e-6)),
    }


def facts_from_statistics(s: dict[str, float]) -> set[str]:
    facts: set[str] = set()
    if s["red"] > s["blue"] + 0.025:
        facts.add("warm_color")
    if s["blue"] >= s["red"] - 0.01:
        facts.add("cool_color")
    if s["green"] > (s["red"] + s["blue"]) / 2 + 0.015:
        facts.add("green_background")
    if s["blue"] > s["red"] + 0.08:
        facts.add("blue_background")
    if s["red"] > 0.34 and s["red"] > s["green"] + 0.015:
        facts.add("brown_dominant")
    facts.add("high_edge" if s["edge"] > 0.20 else "low_edge")
    facts.add("rough_texture" if s["texture"] > 0.21 else "smooth_texture")
    if s["center_contrast"] > 0.015:
        facts.add("center_bright")
    else:
        facts.add("low_center_contrast")
    if s["horizontal_ratio"] > 1.05:
        facts.add("horizontal_body")
    else:
        facts.add("slender_shape")
    return facts


def forward_chain(initial_facts: set[str]) -> dict:
    memory = set(initial_facts)
    trace: list[dict] = []
    scores = {label: 0.0 for label in ("cat", "dog", "deer", "horse")}
    changed = True
    while changed:
        changed = False
        for rule in RULES:
            if rule.conditions <= memory and rule.conclusion not in memory:
                memory.add(rule.conclusion)
                trace.append({"rule": rule.name, "if": sorted(rule.conditions), "then": rule.conclusion})
                changed = True
            if rule.conditions <= memory and rule.conclusion in scores:
                scores[rule.conclusion] = max(scores[rule.conclusion], rule.weight)
    best = max(scores, key=scores.get)
    status = "recognized" if scores[best] > 0 else "unknown"
    return {"status": status, "prediction": best if status == "recognized" else None,
            "scores": scores, "facts": sorted(memory), "trace": trace}


def recognize(path: str | Path) -> dict:
    stats = image_statistics(path)
    result = forward_chain(facts_from_statistics(stats))
    result["statistics"] = stats
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(recognize(args.image), ensure_ascii=False, indent=2))

