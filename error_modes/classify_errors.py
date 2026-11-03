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


@dataclass(frozen=True)
class Classification:
    primary: str
    matches: tuple  # every rule that matched (subset of PRIORITY, in priority order)
    value: int | None
    parse_status: str

    @property
    def correct(self) -> bool:
        return self.primary == "C"

    @property
    def ambiguous(self) -> bool:
        """More than one designed-mode rule matched."""
        return sum(m in DESIGNED_MODES for m in self.matches) > 1


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


def classify(a: int, b: int, completion: str) -> Classification:
    pr: ParseResult = parse_answer(completion)
    return classify_value(a, b, pr.value, pr.status)
