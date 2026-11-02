"""Deterministic pseudo-policy for census smoke tests and CPU end-to-end runs.

Realizes each taxonomy mode by construction; the real classifier then labels the
resulting completions, so the census pipeline is exercised end to end without
torch. Distribution config: {mode: prob} with 'C' = correct rate; remaining
mass renormalised.

Honesty note (D6): a CARRY realization legitimately matches additional rules
(a dropped carry often looks like a digit slip or a truncation). The default
smoke policy (configs/mock_base_policy.json) therefore omits CARRY so the §9.3
gate can pass on mock data; injecting CARRY exercises the ambiguity accounting
truthfully but lowers the mock's designed coverage.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _rng(seed: int, prompt_id: str):
    h = hashlib.sha256(f"mock-policy|{seed}|{prompt_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def realize(mode: str, a: int, b: int, rng) -> str:
    s = a + b
    st = str(s)
    if mode == "C":
        return f"Answer: {s}"
    if mode == "ECHO":
        return f"Answer: {rng.choice([a, b, int(f'{a}{b}'), int(f'{b}{a}')])}"
    if mode == "CARRY":
        from error_modes.classify_errors import carry_drop_answers
        candidates = sorted(carry_drop_answers(a, b) - {s})
        if not candidates:
            return f"Answer: {s + 1337}"  # no carry existed; falls to U
        return f"Answer: {candidates[int(rng.integers(len(candidates)))]}"
    if mode == "TRUNC":
        return f"Answer: {st[:-1]}"  # proper prefix
    if mode == "TRANS":
        # first adjacent pair of distinct digits anywhere in the sum
        for i in range(len(st) - 1):
            if st[i] != st[i + 1]:
                return f"Answer: {st[:i] + st[i + 1] + st[i] + st[i + 2:]}"
        return f"Answer: {s + 1337}"  # all digits equal: no transposition exists
    if mode == "DIGIT":
        d = (int(st[-1]) + 3) % 10
        return f"Answer: {st[:-1] + str(d)}"
    if mode == "I":
        return "I think the result is roughly the two numbers taken together."
    return f"Answer: {s + 1337}"  # U


def mock_completions(a: int, b: int, seed: int, prompt_id: str, probs: dict, samples: int) -> list[str]:
    rng = _rng(seed, prompt_id)
    items = [(k, v) for k, v in probs.items() if not k.startswith("_") and isinstance(v, (int, float))]
    modes, p = zip(*items)
    total = sum(p)
    p = [x / total for x in p]
    drawn = rng.choice(len(modes), size=samples, p=p)
    return [realize(modes[i], a, b, rng) for i in drawn]
