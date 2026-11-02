"""Tests for gpucheck's pure parsing / compatibility logic (no GPU needed)."""
from __future__ import annotations

from scripts.gpucheck import driver_cuda_compatible, parse_nvidia_smi

SAMPLE = """index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%], driver_version
0, NVIDIA L40, 47999 MiB, 21300 MiB, 97 %, 550.54.15
1, NVIDIA L40, 47999 MiB, 0 MiB, 0 %, 550.54.15
"""


def test_parse_nvidia_smi_rows():
    rows = parse_nvidia_smi(SAMPLE)
    assert len(rows) == 2
    assert rows[0]["name"] == "NVIDIA L40"
    assert rows[0]["driver_version"] == "550.54.15"
    assert rows[1]["memory.used"] == "0 MiB"


def test_parse_empty():
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("\n\n") == []


def test_driver_compatibility_rule():
    smi = parse_nvidia_smi(SAMPLE)
    rep = {"cuda_runtime": "12.4"}
    chk = driver_cuda_compatible(smi, rep)
    assert chk["driver"] == "550.54.15"
    assert chk["compatible"] is True
    # pre-baseline driver for CUDA 12 should be incompatible
    old = [dict(smi[0], **{"driver_version": "515.00.00"})]
    assert driver_cuda_compatible(old, rep)["compatible"] is False
    # missing info -> None, never a crash or a false positive
    assert driver_cuda_compatible([], {"cuda_runtime": "12.4"})["compatible"] is None
    assert driver_cuda_compatible(smi, {"cuda_runtime": None})["compatible"] is None
