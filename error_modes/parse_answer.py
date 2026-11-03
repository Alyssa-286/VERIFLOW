"""Final-answer extraction (master plan §9.1).

The prompt asks for a final line `Answer: <integer>`. Extraction is strict:
- every line matching `Answer:` (case-insensitive, optional markdown bold/backticks) is collected;
- a candidate is an optionally-signed integer, allowing thousands separators (`,` `_` thin/normal spaces);
- taxonomy v2: with no `Answer:` line, a completion consisting of exactly one integer
  (after stripping whitespace, `*`, backticks, `$`, `\\boxed{}` and one trailing period)
  is the answer, status "ok_bare";
- otherwise zero `Answer:` lines, an unparseable value, or several `Answer:` lines with
  *different* values give a status without a value (class I, never U).
Formatting differences (commas, spaces, leading zeros, a leading '+') are normalised, not errors (§9.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANSWER_LINE = re.compile(r"^[\s>*_`#-]*answer\s*[:：]\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_INT = re.compile(r"^[+-]?\d{1,3}(?:[,_   ]\d{3})+$|^[+-]?\d+$")


@dataclass(frozen=True)
class ParseResult:
    value: int | None
    status: str  # ok | ok_bare | no_answer_line | unparseable | conflicting
    raw: str | None = None


def _clean(s: str) -> str:
    s = s.strip()
    s = s.strip("*`$ ").rstrip(".").strip()
    s = s.replace("\\,", ",")
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[len("\\boxed{"):-1].strip()
    return s


def parse_integer(token: str) -> int | None:
    t = _clean(token)
    if not _INT.match(t):
        return None
    digits = re.sub(r"[,_   ]", "", t)
    return int(digits)


def parse_answer(completion: str) -> ParseResult:
    matches = _ANSWER_LINE.findall(completion)
    if not matches:
        # Taxonomy v2 (DECISIONS.md D11): a completion that is nothing but one
        # integer is that answer. Prose, working or several numbers stay I.
        v = parse_integer(completion)
        if v is not None:
            return ParseResult(v, "ok_bare", completion.strip())
        return ParseResult(None, "no_answer_line")
    values = []
    for m in matches:
        v = parse_integer(m)
        if v is None:
            return ParseResult(None, "unparseable", m.strip())
        values.append(v)
    if len(set(values)) > 1:
        return ParseResult(None, "conflicting", "|".join(map(str, values)))
    return ParseResult(values[0], "ok", matches[-1].strip())
