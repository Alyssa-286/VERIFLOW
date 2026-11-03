"""Deterministic error-mode taxonomy (v2: bare-integer parse, D11) for two-operand integer addition (master plan §9.2).

Hypothesis matching: compute counterfactual answers for each error mechanism and
match the parsed answer. Every rule that matches is recorded (multi-hot); the
primary class is the first match in priority order:

  0 C      parsed answer == a + b
  1 I      no parseable answer (see parse_answer)
  2 ECHO   answer ∈ {a, b} or the digit concatenation of a and b (either order)
  3 CARRY  equals the column-wise sum with a non-empty subset of carries dropped
  4 TRUNC  digits form a proper prefix or proper suffix of the correct digits
  5 TRANS  correct digits with exactly one adjacent pair of *distinct* digits swapped
  6 DIGIT  same number of digits, exactly one position differs
  7 U      any other integer (including negative numbers)

Operands must be non-negative integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from error_modes.parse_answer import ParseResult, parse_answer

TAXONOMY_VERSION = (Path(__file__).with_name("TAXONOMY_VERSION")).read_text().strip()
PRIORITY = ("C", "I", "ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT", "U")
DESIGNED_MODES = ("ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT")
ALL_CLASSES = PRIORITY

# Subtraction (§9.3 ladder step 2, DECISIONS.md D13): a - b with a > b. A strict
# mirror of the addition taxonomy with BORROW in CARRY's place.
SUB_TAXONOMY_VERSION = "sub-v1"
PRIORITY_SUB = ("C", "I", "ECHO", "BORROW", "TRUNC", "TRANS", "DIGIT", "U")
DESIGNED_MODES_SUB = ("ECHO", "BORROW", "TRUNC", "TRANS", "DIGIT")
# Diagnostic tags on U only (never a designed mode, never in a gate statistic).
SUB_DIAGNOSTICS = ("OPSWAP", "NEG", "SFL")
TASKS = ("add", "sub")

# Taxonomy v3 (Amendment 1, D15): an exhaustive, disjoint partition of wrong
# addition answers by how they differ from the truth. Mechanism labels are kept
# only as a descriptive annotation (Classification.diag).
V3_TAXONOMY_VERSION = "v3"
PRIORITY_V3 = ("C", "I", "NEG", "LEN", "ONE_HI", "ONE_LO", "MULTI")
DESIGNED_MODES_V3 = ("NEG", "LEN", "ONE_HI", "ONE_LO", "MULTI")
SCHEMES = ("add", "sub", "add_v3")  # classification scheme: task + taxonomy


def task_for(scheme: str) -> str:
    return "sub" if scheme == "sub" else "add"


def classes_for(scheme: str = "add") -> tuple[tuple, tuple]:
    """(priority, designed modes) for a classification scheme."""
    if scheme == "add":
        return PRIORITY, DESIGNED_MODES
    if scheme == "sub":
        return PRIORITY_SUB, DESIGNED_MODES_SUB
    if scheme == "add_v3":
        return PRIORITY_V3, DESIGNED_MODES_V3
    raise ValueError(f"unknown scheme {scheme!r}")


def taxonomy_version_for(scheme: str = "add") -> str:
    return {"add": TAXONOMY_VERSION, "sub": SUB_TAXONOMY_VERSION, "add_v3": V3_TAXONOMY_VERSION}[scheme]


_ALL_DESIGNED = frozenset(DESIGNED_MODES) | frozenset(DESIGNED_MODES_SUB) | frozenset(DESIGNED_MODES_V3)


@dataclass(frozen=True)
class Classification:
    primary: str
    matches: tuple  # every rule that matched (subset of PRIORITY, in priority order)
    value: int | None
    parse_status: str
    diag: tuple = ()  # subtraction-only diagnostic tags on U (SUB_DIAGNOSTICS)

    @property
    def correct(self) -> bool:
        return self.primary == "C"

    @property
    def ambiguous(self) -> bool:
        """More than one designed-mode rule matched."""
        return sum(m in _ALL_DESIGNED for m in self.matches) > 1


@lru_cache(maxsize=65536)
def carry_drop_answers(a: int, b: int) -> frozenset:
    """All sums obtainable by dropping a non-empty subset of the carries that actually arise.

    Column-by-column from the least significant digit. At each column the carry out is either
    propagated (correct) or dropped (the error). Dropping changes later carries, so the tree is
    enumerated rather than subsets of the correct carry positions. A dropped final carry loses the
    leading 1.
    """
    da = [int(c) for c in str(a)[::-1]]
    db = [int(c) for c in str(b)[::-1]]
    n = max(len(da), len(db))
    da += [0] * (n - len(da))
    db += [0] * (n - len(db))
    out = set()

    def rec(i: int, carry_in: int, digits: list, dropped: bool):
        if i == n:
            final = digits + ([carry_in] if carry_in else [])
            if dropped:
                out.add(int("".join(map(str, final[::-1]))))
            return
        s = da[i] + db[i] + carry_in
        d, c = s % 10, s // 10
        rec(i + 1, c, digits + [d], dropped)
        if c:
            rec(i + 1, 0, digits + [d], True)

    rec(0, 0, [], False)
    return frozenset(out)


def _echo(a: int, b: int):
    return {a, b, int(f"{a}{b}"), int(f"{b}{a}")}


def _is_trunc(v: int, s: int) -> bool:
    ds, dv = str(s), str(v)
    if len(dv) >= len(ds) or len(dv) == 0:
        return False
    return ds.startswith(dv) or ds.endswith(dv)


def _transpositions(s: int):
    ds = str(s)
    out = set()
    for i in range(len(ds) - 1):
        if ds[i] != ds[i + 1]:
            t = ds[:i] + ds[i + 1] + ds[i] + ds[i + 2:]
            if t[0] != "0":
                out.add(int(t))
    return out


def _is_digit_slip(v: int, s: int) -> bool:
    dv, ds = str(v), str(s)
    return len(dv) == len(ds) and sum(x != y for x, y in zip(dv, ds)) == 1


def classify_value(a: int, b: int, value: int | None, parse_status: str = "ok") -> Classification:
    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    s = a + b
    if value is None:
        return Classification("I", ("I",), None, parse_status)
    if value == s:
        return Classification("C", ("C",), value, parse_status)
    matches = []
    if value in _echo(a, b):
        matches.append("ECHO")
    if value >= 0:
        if value in carry_drop_answers(a, b):
            matches.append("CARRY")
        if _is_trunc(value, s):
            matches.append("TRUNC")
        if value in _transpositions(s):
            matches.append("TRANS")
        if _is_digit_slip(value, s):
            matches.append("DIGIT")
    if not matches:
        matches.append("U")
    return Classification(matches[0], tuple(matches), value, parse_status)


@lru_cache(maxsize=65536)
def borrow_drop_answers(a: int, b: int) -> frozenset:
    """All differences a - b obtainable by dropping a non-empty subset of the borrows that arise.

    Mirror of carry_drop_answers: column by column from the least significant digit, a column
    that needs a borrow still adds 10, and the borrow is either taken from the next column
    (correct) or dropped (the error: the next column is not decremented). Dropping changes later
    borrows, so the tree is enumerated. Requires a > b >= 0.
    """
    da = [int(c) for c in str(a)[::-1]]
    db = [int(c) for c in str(b)[::-1]]
    n = max(len(da), len(db))
    da += [0] * (n - len(da))
    db += [0] * (n - len(db))
    out = set()

    def rec(i: int, borrow_in: int, digits: list, dropped: bool):
        if i == n:
            if dropped and not borrow_in:
                out.add(int("".join(map(str, digits[::-1]))))
            return
        x = da[i] - borrow_in - db[i]
        d, borrow = (x + 10, 1) if x < 0 else (x, 0)
        rec(i + 1, borrow, digits + [d], dropped)
        if borrow:
            rec(i + 1, 0, digits + [d], True)

    rec(0, 0, [], False)
    return frozenset(out)


def _smaller_from_larger(a: int, b: int) -> int:
    """Classic buggy-subtraction rule: each column gives |a_i - b_i|, no borrows."""
    da, db = str(a)[::-1], str(b)[::-1].ljust(len(str(a)), "0")
    return int("".join(str(abs(int(x) - int(y))) for x, y in zip(da, db))[::-1])


@lru_cache(maxsize=65536)
def classify_value_sub(a: int, b: int, value: int | None, parse_status: str = "ok") -> Classification:
    if not a > b >= 0:
        raise ValueError("subtraction operands must satisfy a > b >= 0")
    d = a - b
    if value is None:
        return Classification("I", ("I",), None, parse_status)
    if value == d:
        return Classification("C", ("C",), value, parse_status)
    matches = []
    if value in _echo(a, b):
        matches.append("ECHO")
    if value >= 0:
        if value in borrow_drop_answers(a, b):
            matches.append("BORROW")
        if _is_trunc(value, d):
            matches.append("TRUNC")
        if value in _transpositions(d):
            matches.append("TRANS")
        if _is_digit_slip(value, d):
            matches.append("DIGIT")
    diag: tuple = ()
    if not matches:
        matches.append("U")
        diag = tuple(t for t, hit in (("OPSWAP", value == a + b), ("NEG", value == -d),
                                      ("SFL", value == _smaller_from_larger(a, b))) if hit)
    return Classification(matches[0], tuple(matches), value, parse_status, diag)


def v3_label(value: int, s: int) -> str:
    """Partition cell of a wrong, parsed addition answer (Amendment 1)."""
    if value < 0:
        return "NEG"
    dv, ds = str(value), str(s)
    if len(dv) != len(ds):
        return "LEN"
    diff = [i for i, (x, y) in enumerate(zip(dv, ds)) if x != y]
    if len(diff) >= 2:
        return "MULTI"
    return "ONE_HI" if diff[0] < len(ds) // 2 else "ONE_LO"


@lru_cache(maxsize=65536)
def classify_value_v3(a: int, b: int, value: int | None, parse_status: str = "ok") -> Classification:
    """Taxonomy v3: primary = partition cell; diag = the v2 mechanism labels
    (descriptive only, never used by a gate statistic)."""
    mech = classify_value(a, b, value, parse_status)
    if mech.primary in ("C", "I"):
        return mech
    label = v3_label(value, a + b)
    return Classification(label, (label,), value, parse_status, mech.matches)


def classify(a: int, b: int, completion: str, task: str = "add") -> Classification:
    """task is a classification scheme: "add" (v2 mechanisms), "sub", or "add_v3"."""
    pr: ParseResult = parse_answer(completion)
    if task == "sub":
        return classify_value_sub(a, b, pr.value, pr.status)
    if task == "add_v3":
        return classify_value_v3(a, b, pr.value, pr.status)
    if task != "add":
        raise ValueError(f"unknown scheme {task!r}")
    return classify_value(a, b, pr.value, pr.status)
