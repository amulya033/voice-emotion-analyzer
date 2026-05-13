import torch
from transformers import pipeline
from .config import MODEL_ID, LABEL_EXPAND, EMA_ALPHA


def load_model() -> object:
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=MODEL_ID, device=device)
    backend = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"Model ready on {backend}.")
    return clf


def classify(clf, audio) -> list[dict]:
    results = clf(audio, top_k=None)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def smooth_emotions(
    previous: dict[str, float] | None,
    current: list[dict],
    alpha: float = EMA_ALPHA,
) -> list[dict]:
    """Apply EMA smoothing so bar values don't jump each hop."""
    if previous is None:
        return current
    smoothed = []
    prev_map = {e["label"]: e["score"] for e in previous.get("raw", current)}
    for e in current:
        lbl = e["label"]
        prev_score = prev_map.get(lbl, e["score"])
        smoothed.append({
            "label": lbl,
            "score": alpha * e["score"] + (1 - alpha) * prev_score,
        })
    smoothed.sort(key=lambda x: x["score"], reverse=True)
    return smoothed


def confidence_level(emotions: list[dict]) -> str:
    """Return a label describing model certainty."""
    top = emotions[0]["score"] if emotions else 0.0
    if top >= 0.60:
        return "HIGH"
    if top >= 0.40:
        return "MODERATE"
    return "LOW"
