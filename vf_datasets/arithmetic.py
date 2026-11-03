"""Two-operand integer addition prompts with disjoint splits (master plan §9.1, §11).

Named `vf_datasets` to avoid shadowing Hugging Face `datasets`.

- Operands are uniform n-digit integers (leading digit non-zero), independently per operand.
- Splits are disjoint on the *unordered* operand pair. Precedence when de-duplicating is
  eval > eval_mode > census > train: a pair drawn for a higher-precedence split is removed from
  every lower one, so the eval set is never contaminated.
- Each split uses its own seed stream derived from (base_seed, split name, n).
- The prompt template is frozen; its SHA-256 is recorded in every manifest.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# v1 (2026-11-02) failed Gate 3 at every n, mostly through missing Answer:
# lines (DECISIONS.md D8). v2 is the §9.3 ladder's answer-only prompt (D9).
PROMPT_TEMPLATE_V1 = (
    "Compute {a} + {b}.\n"
    "You may show brief working. On the last line, write the final answer exactly in the form\n"
    "Answer: <integer>"
)
PROMPT_VERSION = "v2"
PROMPT_TEMPLATE = (
    "Compute {a} + {b}.\n"
    "Reply with only the final answer, exactly in the form\n"
    "Answer: <integer>"
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()

# Subtraction (§9.3 ladder step 2, D13): the v2 wording with only the operator
# changed. Operands satisfy a > b, both n-digit, so a - b >= 1.
SUB_PROMPT_TEMPLATE = (
    "Compute {a} - {b}.\n"
    "Reply with only the final answer, exactly in the form\n"
    "Answer: <integer>"
)
SUB_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(SUB_PROMPT_TEMPLATE.encode()).hexdigest()
TASKS = ("add", "sub")


def prompt_template_sha256(task: str = "add") -> str:
    return {"add": PROMPT_TEMPLATE_SHA256, "sub": SUB_PROMPT_TEMPLATE_SHA256}[task]

SPLIT_PRECEDENCE = ("eval", "eval_mode", "census", "train")
DEFAULT_SIZES = {"eval": 1000, "eval_mode": 1000, "census": 2000, "train": 50000}


def _split_seed(base_seed: int, split: str, n_digits: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{split}|{n_digits}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def pair_key(a: int, b: int) -> tuple:
    return (a, b) if a <= b else (b, a)


def _draw(rng: np.random.Generator, n_digits: int, size: int):
    lo, hi = 10 ** (n_digits - 1), 10**n_digits
    # integers() handles up to int64; n_digits ≤ 18 is enforced by the caller
    return rng.integers(lo, hi, size=size, dtype=np.int64), rng.integers(lo, hi, size=size, dtype=np.int64)


def make_prompt(a: int, b: int, task: str = "add") -> str:
    return (SUB_PROMPT_TEMPLATE if task == "sub" else PROMPT_TEMPLATE).format(a=a, b=b)


def make_splits(n_digits: int, sizes: dict | None = None, base_seed: int = 20261102,
                task: str = "add") -> dict[str, pd.DataFrame]:
    """Return {split: DataFrame[prompt_id, a, b, answer, prompt, n_digits]} with disjoint unordered pairs.

    task="sub" draws from its own seed namespace, orders each pair so a > b and
    drops a == b; answer = a - b. task="add" is unchanged bit for bit."""
    if not 1 <= n_digits <= 18:
        raise ValueError("n_digits must be in [1, 18]")
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    taken: set = set()
    out = {}
    for split in SPLIT_PRECEDENCE:
        want = sizes.get(split, 0)
        if want == 0:
            continue
        rng = np.random.default_rng(_split_seed(base_seed, split if task == "add" else f"sub|{split}", n_digits))
        rows = []
        seen_here: set = set()
        attempts = 0
        while len(rows) < want:
            attempts += 1
            if attempts > 200:
                raise RuntimeError(f"cannot draw {want} unique pairs for split {split} at n={n_digits}")
            A, B = _draw(rng, n_digits, 2 * (want - len(rows)) + 16)
            for a, b in zip(A.tolist(), B.tolist()):
                if task == "sub":
                    if a == b:
                        continue
                    a, b = max(a, b), min(a, b)
                k = pair_key(a, b)
                if k in taken or k in seen_here:
                    continue
                seen_here.add(k)
                rows.append((a, b))
                if len(rows) == want:
                    break
        taken |= seen_here
        df = pd.DataFrame(rows, columns=["a", "b"])
        tag = "" if task == "add" else "sub-"
        df.insert(0, "prompt_id", [f"{split}-{tag}n{n_digits}-{i:06d}" for i in range(len(df))])
        df["answer"] = df["a"] - df["b"] if task == "sub" else df["a"] + df["b"]
        df["prompt"] = [make_prompt(a, b, task) for a, b in zip(df["a"], df["b"])]
        df["n_digits"] = n_digits
        out[split] = df
    return out


def frame_sha256(df: pd.DataFrame) -> str:
    """Order-sensitive content hash of a split (columns prompt_id, a, b, answer, prompt)."""
    cols = ["prompt_id", "a", "b", "answer", "prompt"]
    payload = df[cols].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def check_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any unordered operand pair appears in two splits or twice in one split."""
    seen: dict = {}
    for name, df in splits.items():
        for a, b in zip(df["a"].tolist(), df["b"].tolist()):
            k = pair_key(a, b)
            if k in seen:
                raise AssertionError(f"pair {k} appears in {seen[k]} and {name}")
            seen[k] = name
