"""Isolation tests (master plan §16.1): VERIFLOW code must never reference the
ErrorEcho project paths, and the preflight layout check must pass."""
from __future__ import annotations

import re

from scripts.preflight import _ISOLATION_PATTERNS, check_isolation, check_layout

# Offending = a real path reference into ErrorEcho's directories (the bare
# project name in docs is legitimate).
OFFENDING_PATTERNS = tuple(re.compile(p.pattern) for p in _ISOLATION_PATTERNS)


def test_no_echo_references_in_code():
    assert check_isolation() == []


def test_layout_complete():
    assert check_layout() == []


def test_isolation_check_catches_real_offenders(tmp_path, monkeypatch):
    import scripts.preflight as pre
    (tmp_path / "evil.py").write_text("import os\nos.chdir('~/ErrorEcho-phase11-20')\n")
    monkeypatch.setattr(pre, "ROOT", tmp_path)
    monkeypatch.setattr(pre, "REQUIRED_PATHS", ())
    offenders = check_isolation(tmp_path)
    assert offenders == ["evil.py"]
