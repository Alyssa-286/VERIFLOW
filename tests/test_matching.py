import numpy as np
import pytest

from verifiers import matched_verifiers as mv
from verifiers import stats as vs

Q_REF = {"ECHO": 0.06, "CARRY": 0.34, "TRUNC": 0.12, "TRANS": 0.08, "DIGIT": 0.25, "U": 0.10, "I": 0.05}
DESIGNED = ["ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT"]


def test_homogeneous_is_floor_and_matched():
    c = mv.homogeneous_config(Q_REF, 0.3, 0.9)
    assert c.Fbar == pytest.approx(0.3, abs=1e-14)
    assert c.V == pytest.approx(mv.v_floor(Q_REF, 0.3), abs=1e-14)
    assert c.f["U"] == 0.3 and c.f["I"] == 0.0
    designed_vals = {c.f[k] for k in DESIGNED}
    assert len(designed_vals) == 1


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.5, 0.9])
def test_generated_configs_match_exactly(frac):
    rng = np.random.default_rng(1)
    lo, hi = mv.v_floor(Q_REF, 0.3), mv.feasible_v_max(Q_REF, 0.3)
    V = lo + frac * (hi - lo)
    for _ in range(20):
        c = mv.generate_config(Q_REF, DESIGNED, 0.3, V, 0.9, rng)
        assert abs(c.Fbar - 0.3) <= 1e-12
        assert abs(c.V - V) <= 1e-12
        _, _, f = c.vectors()
        assert (f >= 0).all() and (f <= 1).all()
        assert c.f["U"] == 0.3 and c.f["I"] == 0.0 and c.tau == 0.9


def test_fmax_target_is_hit_and_is_the_max():
    rng = np.random.default_rng(2)
    ranges = mv.feasible_v_range_by_mode(Q_REF, 0.3, 0.8)
    assert ranges
    for _ in range(20):
        mode = list(ranges)[int(rng.integers(len(ranges)))]
        lo, hi = ranges[mode]
        c = mv.generate_config(Q_REF, DESIGNED, 0.3, lo + 0.4 * (hi - lo), 0.9, rng, fmax=0.8, fmax_mode=mode)
        assert c.f[mode] == 0.8
        _, _, f = c.vectors()
        assert f.max() == pytest.approx(0.8, abs=1e-12)
        assert abs(c.Fbar - 0.3) <= 1e-12


def test_v_max_is_attainable_and_not_exceeded():
    hi = mv.feasible_v_max(Q_REF, 0.3)
    rng = np.random.default_rng(3)
    mv.generate_config(Q_REF, DESIGNED, 0.3, 0.98 * hi, 0.9, rng, max_tries=200_000)
    with pytest.raises(ValueError):
        mv.generate_config(Q_REF, DESIGNED, 0.3, 1.05 * hi, 0.9, rng, max_tries=2_000)


def test_v_max_brute_force_small_case():
    """Vertex search agrees with a dense grid search on a 3-mode case."""
    q_ref = {"A": 0.5, "B": 0.3, "C": 0.2}
    hi = mv.feasible_v_max(q_ref, 0.3)
    q = np.array([0.5, 0.3, 0.2])
    g = np.linspace(0, 1, 2001)
    fa, fb = np.meshgrid(g, g)
    fc = (0.3 - 0.5 * fa - 0.3 * fb) / 0.2
    ok = (fc >= 0) & (fc <= 1)
    F = np.stack([fa[ok], fb[ok], fc[ok]], 1)
    best = ((F - 0.3) ** 2 @ q).max()
    assert best <= hi + 1e-12
    assert best >= hi - 1e-3


def test_infeasible_V_below_floor_raises():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        mv.generate_config(Q_REF, DESIGNED, 0.3, 0.5 * mv.v_floor(Q_REF, 0.3), 0.9, rng, max_tries=500)


def test_grid_determinism_hashes_and_mode_randomisation():
    g1 = mv.design_grid(Q_REF, 0.3, 0.9, [0.2, 0.6], [None, 0.8], 4, seed=11)
    g2 = mv.design_grid(Q_REF, 0.3, 0.9, [0.2, 0.6], [None, 0.8], 4, seed=11)
    assert [c.sha256 for c in g1] == [c.sha256 for c in g2]
    assert len({c.sha256 for c in g1}) == len(g1)
    # the argmax mode varies across configs (identity randomisation)
    argmax_modes = {max(DESIGNED, key=lambda k: c.f[k]) for c in g1 if c.grid_cell != "homog"}
    assert len(argmax_modes) >= 3
    for c in g1:
        assert abs(c.Fbar - 0.3) <= 1e-12


def test_split_is_stratified_and_keeps_homogeneous():
    g = mv.design_grid(Q_REF, 0.3, 0.9, [0.2, 0.6], [None, 0.8], 6, seed=12)
    train, held = mv.split_heldout(g, 1 / 3, seed=0)
    assert len(train) + len(held) == len(g)
    assert {c.config_id for c in train}.isdisjoint({c.config_id for c in held})
    assert any(c.grid_cell == "homog" for c in train)
    for cell in {c.grid_cell for c in g if c.grid_cell != "homog"}:
        n_cell = sum(c.grid_cell == cell for c in g)
        assert sum(c.grid_cell == cell for c in held) == round(n_cell / 3)


def test_grid_fills_fmax_cells():
    g = mv.design_grid(Q_REF, 0.3, 0.9, [0.2, 0.5, 0.8], [0.6, 0.9], 5, seed=13)
    from collections import Counter

    counts = Counter(c.grid_cell for c in g)
    for fm in (0.6, 0.9):
        for vf in (0.2, 0.5, 0.8):
            assert counts[f"fmax={fm}|Vfrac={vf}"] == 5, counts


def test_designed_targets_bookkeeping():
    S, W = mv.designed_targets(Q_REF, DESIGNED, 0.3, 0.05)
    assert S == pytest.approx(0.3 * 0.9)
    assert W == pytest.approx(0.05 - 0.05 * 0.09)
