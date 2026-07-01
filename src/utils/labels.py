import re
from typing import List

DEFAULT_CLASS_NAMES: List[str] = ["glass", "hazardous", "metal", "organic", "paper", "plastic", "textile"]


def canonical_label(raw_label: str) -> str:
    if raw_label is None:
        return ""

    normalized = str(raw_label).strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    aliases = {
        "paper_cardboard": "paper",
        "cardboard": "paper",
        "plastic_bottle": "plastic",
        "bottle": "plastic",
        "organic_waste": "organic",
        "food_waste": "organic",
        "hazardous_waste": "hazardous",
        "glass_bottle": "glass",
        "metal_can": "metal",
        "textile_waste": "textile",
        "cloth": "textile",
    }
    return aliases.get(normalized, normalized)


def display_label(raw_label: str) -> str:
    label = canonical_label(raw_label)
    if not label:
        return "Unknown"
    return label.replace("_", " ").title()
