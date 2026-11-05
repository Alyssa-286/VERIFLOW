"""AI-performed audit labelling for census n5_v3_s20261104_j3640.

Written by Claude (Opus 5.5) at the researcher's request on 2026-11-05, from
audit_instructions.md only. Neither audit_key.csv nor any classifier output
(classification.parquet, gate3.json) was opened. Rows whose completion is not a
bare integer were judged one by one; those judgments are listed in MANUAL with
their reasons and cross-checked against the rule implementation below.
"""
import csv, re, sys

def rule_label(completion, true):
    lines = [l for l in completion.splitlines() if l.strip().startswith("Answer:")]
    if len(lines) > 1:
        vals = {l.split("Answer:", 1)[1].strip() for l in lines}
        if len(vals) > 1:
            return "I"
    if lines:
        v = lines[0].split("Answer:", 1)[1].strip()
    else:
        v = completion.strip()
    if not re.fullmatch(r"-?\d+", v):
        return "I"
    ans = int(v)
    if ans < 0:
        return "NEG"
    s, t = str(ans), str(true)
    if len(s) != len(t):
        return "LEN"
    diff = [i for i in range(len(t)) if s[i] != t[i]]
    if len(diff) == 0:
        return "CORRECT?"  # not a valid audit label; must never occur
    if len(diff) >= 2:
        return "MULTI"
    return "ONE_HI" if diff[0] < len(t) // 2 else "ONE_LO"

MANUAL = {
    "AUD-0003": ("I", "'1,982,300': comma-grouped, not a bare integer, no Answer: line. "
                      "Interpretive call: a lenient reader would strip commas -> 1982300 -> LEN."),
    "AUD-0010": ("I", "'1,52114': malformed grouping, unparseable"),
    "AUD-0031": ("MULTI", "114050 vs 113949: positions 2-5 differ"),
    "AUD-0032": ("I", "equation, no Answer: line, not a bare integer"),
    "AUD-0040": ("ONE_HI", "121785 vs 120785: position 2 < floor(6/2)=3"),
    "AUD-0056": ("ONE_LO", "121352 vs 121342: position 4 >= 3"),
    "AUD-0108": ("ONE_HI", "60676 vs 69676: position 1 < floor(5/2)=2"),
    "AUD-0148": ("I", "equation, no Answer: line, not a bare integer"),
    "AUD-0152": ("LEN", "107256 (6 digits) vs 97256 (5 digits)"),
    "AUD-0153": ("I", "'1,94807': malformed grouping, unparseable"),
    "AUD-0182": ("I", "'1 <integer>': not an integer"),
    "AUD-0199": ("I", "'<64452>': not a bare integer"),
}

def main(src, dst):
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(rows) == 200
    for r in rows:
        auto = rule_label(r["completion"], int(r["answer_true"]))
        if r["row_id"] in MANUAL:
            lab = MANUAL[r["row_id"]][0]
            if lab != auto:
                print(f"{r['row_id']}: manual {lab} vs rule {auto}")
        else:
            assert re.fullmatch(r"-?\d+", r["completion"].strip()), r["row_id"]
            lab = auto
        assert lab in {"NEG", "LEN", "ONE_HI", "ONE_LO", "MULTI", "I"}, (r["row_id"], lab)
        assert int(r["a"]) + int(r["b"]) == int(r["answer_true"]), r["row_id"]
        r["judgment"] = lab
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
