import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "configs/locked/q_ref.lock").exists(), reason="needs q_ref.lock")


def _build(R):
    from experiments.make_grid_configs import build
    from verifiers.qref import load_q_ref
    b = json.loads((ROOT / "configs/base.json").read_text())["verifier"]
    return build(load_q_ref()["q_ref"], b["Fbar"], b["tau"], R)


@pytest.mark.parametrize("R,n_aux", [(12, 0), (24, 8), (36, 12)])
def test_grid_sizes_and_matching(R, n_aux):
    matched, aux = _build(R)
    assert len(matched) == R and len(aux) == n_aux
    assert matched[0].grid_cell == "homog"
    for c in matched:
        p = c.predictors()
        assert abs(p["FPR"] - 0.3) < 1e-9 and abs(p["TPR"] - 0.9) < 1e-12


def test_grid_is_deterministic():
    a, _ = _build(24)
    b, _ = _build(24)
    assert [c.sha256 for c in a] == [c.sha256 for c in b]
