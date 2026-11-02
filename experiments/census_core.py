"""Base-policy error census: CPU-analysis core (master plan §11, §9.3).

Pure functions, no torch/transformers: the GPU job classifies completions and
then calls these. A generations frame has columns `prompt_id, a, b, completion`.

Outputs ordered exactly as §11's list of census artifacts:
  q_ref.json (point estimate + 95% cluster-bootstrap CI, clustered by prompt),
  per-mode output-length distribution, per-prompt accuracy histogram,
  per-prompt mode entropy (B7), split-half TV distance (stability),
  coverage and ambiguity, and the §9.3 gate evaluation.

Conventions:
  - `q` weights live over *incorrect* classes only (C excluded), matching
    verifiers/stats.py. Classes: ECHO, CARRY, TRUNC, TRANS, DIGIT, U, I.
  - Primary label means the first matching rule in PRIORITY order; every
    matching rule is retained (multi-hot) for the ambiguity statistic.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from error_modes.classify_errors import ALL_CLASSES, DESIGNED_MODES, TAXONOMY_VERSION, classify

INCORRECT_CLASSES = [c for c in ALL_CLASSES if c != "C"]

GATE_COVERAGE_MIN = 0.70      # §9.3.1 designed-mode coverage of incorrect outputs
GATE_MASS_COUNT = 3           # §9.3.2 at least N designed modes with >= MASS_MIN
GATE_MASS_MIN = 0.05
GATE_MERGE_BELOW = 0.02       # §9.3.2 designed modes below this merge into U
GATE_AMBIGUITY_MAX = 0.10     # §9.3.3
GATE_TV_MAX = 0.03            # §9.3.4
GATE_ACC_RANGE = (0.25, 0.60)  # §9.1 base-accuracy window used to choose n


# ---------------------------------------------------------------- classification

def classify_generations(gens: pd.DataFrame) -> pd.DataFrame:
    """Annotate generations with taxonomy v1 labels. Returns a new frame with
    per-row: mode, matches, correct, ambiguous, value, parse_status, n_chars."""
    required = {"prompt_id", "a", "b", "completion"}
    missing = required - set(gens.columns)
    if missing:
        raise ValueError(f"census frame missing columns: {sorted(missing)}")
    rows = []
    for r in gens.itertuples(index=False):
        cl = classify(int(r.a), int(r.b), str(r.completion))
        rows.append(
            {
                "prompt_id": r.prompt_id,
                "a": int(r.a),
                "b": int(r.b),
                "completion": str(r.completion),
                "mode": cl.primary,
                "matches": "|".join(cl.matches),
                "correct": cl.correct,
                "ambiguous": cl.ambiguous,
                "value": cl.value,
                "parse_status": cl.parse_status,
                "n_chars": len(str(r.completion)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- q_ref estimate

def mode_counts(cls: pd.DataFrame) -> pd.Series:
    """Counts of primary classes among *incorrect* generations, ordered INCORRECT_CLASSES."""
    bad = cls.loc[~cls["correct"], "mode"]
    vc = bad.value_counts()
    return pd.Series([int(vc.get(c, 0)) for c in INCORRECT_CLASSES], index=INCORRECT_CLASSES, dtype=int)


def q_ref_point(cls: pd.DataFrame) -> dict[str, float]:
    counts = mode_counts(cls)
    n_bad = int(counts.sum())
    if n_bad == 0:
        raise ValueError("census contains no incorrect generations: q_ref is undefined")
    return {c: float(counts[c] / n_bad) for c in INCORRECT_CLASSES}


def cluster_bootstrap_q(cls: pd.DataFrame, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict[str, list[float]]:
    """Percentile CI for each class mass, resampling *prompts* with replacement
    (cluster bootstrap). Prompts contribute their samples unaltered."""
    bad = cls.loc[~cls["correct"], ["prompt_id", "mode"]]
    pids = bad["prompt_id"].to_numpy()
    modes = bad["mode"].to_numpy()
    uniq = np.unique(pids)
    if uniq.size < 2:
        raise ValueError("need ≥ 2 prompts with errors for a cluster bootstrap")
    counts = {c: np.zeros(n_boot) for c in INCORRECT_CLASSES}
    rng = np.random.default_rng(seed)
    idx_by_prompt: dict[Any, np.ndarray] = {}
    for uid in uniq:
        idx_by_prompt[uid] = np.where(pids == uid)[0]
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_prompt[u] for u in chosen])
        cm = pd.Series(modes[sel]).value_counts().reindex(INCORRECT_CLASSES).fillna(0)
        total = cm.sum()
        for c in INCORRECT_CLASSES:
            counts[c][b] = cm[c] / total
    lo, hi = (1.0 - ci) / 2 * 100, (1.0 + ci) / 2 * 100
    return {c: [float(np.percentile(counts[c], lo)), float(np.percentile(counts[c], hi))] for c in INCORRECT_CLASSES}


# ---------------------------------------------------------------- §11 summaries

def half_of_prompt(prompt_id: str) -> int:
    """Deterministic split-half assignment (hash-based, order-free)."""
    h = hashlib.sha256(("split-half|" + prompt_id).encode()).digest()
    return h[0] & 1


def split_half_tv(cls: pd.DataFrame) -> float:
    """TV distance between the q estimates of two disjoint prompt halves."""
    cls = cls.copy()
    cls["half"] = [half_of_prompt(p) for p in cls["prompt_id"]]
    qs = {}
    for h in (0, 1):
        sub = cls[cls["half"] == h].reset_index(drop=True)
        qs[h] = q_ref_point(sub)  # raises if a half has zero errors (census too sparse)
    return 0.5 * sum(abs(qs[0][c] - qs[1][c]) for c in INCORRECT_CLASSES)


def coverage_and_ambiguity(cls: pd.DataFrame) -> dict[str, float]:
    bad = cls[~cls["correct"]]
    n_bad = len(bad)
    if n_bad == 0:
        raise ValueError("no incorrect generations")
    n_designed = int(bad["mode"].isin(DESIGNED_MODES).sum())
    designed = bad[bad["mode"].isin(DESIGNED_MODES)]
    n_amb = int(designed["ambiguous"].sum())
    return {
        "coverage": n_designed / n_bad,
        "ambiguity_rate": (n_amb / n_designed) if n_designed else 0.0,
        "n_incorrect": n_bad,
        "n_designed": n_designed,
    }


def per_mode_lengths(cls: pd.DataFrame) -> pd.DataFrame:
    g = cls.groupby("mode")["n_chars"]
    out = g.agg(["count", "mean", "std"]).reset_index()
    return out


def per_prompt_summary(cls: pd.DataFrame) -> pd.DataFrame:
    """Per-prompt accuracy, error-mode entropy (incl. C; B7), and n_samples."""
    rows = []
    for pid, grp in cls.groupby("prompt_id", sort=True):
        vc = grp["mode"].value_counts(normalize=True)
        p = vc.to_numpy()
        ent = float(-(p * np.log(p[p > 0])).sum())
        rows.append(
            {
                "prompt_id": pid,
                "a": int(grp["a"].iloc[0]),
                "b": int(grp["b"].iloc[0]),
                "n_samples": int(len(grp)),
                "accuracy": float(grp["correct"].mean()),
                "mode_entropy": ent,
                "n_errors": int((~grp["correct"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def accuracy_histogram(per_prompt: pd.DataFrame) -> pd.Series:
    """Histogram of per-prompt accuracy (0..1, G bins of the sample count)."""
    g = int(per_prompt["n_samples"].iloc[0])
    bins = [i / g for i in range(g + 1)]
    return per_prompt["accuracy"].value_counts(bins=bins).sort_index()


# ---------------------------------------------------------------- §9.3 gate

def evaluate_construct_gate(
    q: dict[str, float], coverage: float, ambiguity_rate: float, tv: float,
    base_accuracy: float | None = None,
) -> list[dict[str, Any]]:
    """Criteria §9.3.1–4 (plus the §9.1 accuracy window when supplied).

    Criterion 5 (synthetic recovery ≥ 99 %) is `tests/test_classifier.py` and is
    reported by the caller; criterion 6 (200-prompt manual audit) is human work
    and is emitted as a sample file by `check_construct_gate.py`.
    """
    masses_ordered = sorted(
        ((c, m) for c, m in q.items() if c in DESIGNED_MODES), key=lambda t: -t[1]
    )
    n_heavy = sum(1 for _, m in masses_ordered if m >= GATE_MASS_MIN)
    light = [c for c, m in q.items() if c in DESIGNED_MODES and m < GATE_MERGE_BELOW]
    checks = [
        {"id": "9.3.1 coverage", "value": coverage, "threshold": f">= {GATE_COVERAGE_MIN}",
         "pass": coverage >= GATE_COVERAGE_MIN},
        {"id": "9.3.2 mass", "value": n_heavy, "threshold": f">= {GATE_MASS_COUNT} designed modes at >= {GATE_MASS_MIN}",
         "pass": n_heavy >= GATE_MASS_COUNT},
        {"id": "9.3.3 ambiguity", "value": ambiguity_rate, "threshold": f"<= {GATE_AMBIGUITY_MAX}",
         "pass": ambiguity_rate <= GATE_AMBIGUITY_MAX},
        {"id": "9.3.4 split-half TV", "value": tv, "threshold": f"<= {GATE_TV_MAX}", "pass": tv <= GATE_TV_MAX},
    ]
    if base_accuracy is not None:
        checks.append(
            {
                "id": "9.1 base accuracy window",
                "value": base_accuracy,
                "threshold": f"in [{GATE_ACC_RANGE[0]}, {GATE_ACC_RANGE[1]}]",
                "pass": GATE_ACC_RANGE[0] <= base_accuracy <= GATE_ACC_RANGE[1],
            }
        )
    if light:
        checks.append(
            {"id": "9.3.2 merge note", "value": light, "threshold": "designed modes < 0.02 merge into U before configs",
             "pass": True}  # informational; the merge happens at config generation (Phase 5)
        )
    return checks


def gate_passes(checks: list[dict[str, Any]]) -> bool:
    return all(c["pass"] for c in checks if c["id"] != "9.3.2 merge note")
