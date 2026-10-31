import numpy as np
import pytest

from verifiers import stats as vs

Q = np.array([0.5, 0.3, 0.2])
F = np.array([0.1, 0.4, 0.7])


def test_aggregate_and_variance_by_hand():
    fbar = 0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.7  # 0.31
    assert vs.aggregate_fpr(Q, F) == pytest.approx(fbar, abs=1e-15)
    v = 0.5 * (0.1 - fbar) ** 2 + 0.3 * (0.4 - fbar) ** 2 + 0.2 * (0.7 - fbar) ** 2
    assert vs.weighted_var(Q, F) == pytest.approx(v, abs=1e-15)
    k3 = 0.5 * (0.1 - fbar) ** 3 + 0.3 * (0.4 - fbar) ** 3 + 0.2 * (0.7 - fbar) ** 3
    assert vs.kappa3(Q, F) == pytest.approx(k3, abs=1e-15)


def test_homogeneous_has_zero_dispersion():
    f = np.full(3, 0.3)
    assert vs.weighted_var(Q, f) == 0.0
    assert vs.kappa3(Q, f) == 0.0
    assert vs.v_q2(Q, f) == 0.0
    assert vs.trigger_freq(Q, f) == pytest.approx(1.0)


def test_cgf_derivatives_are_cumulants():
    # K(0)=0, K'(0)=F̄, K''(0)=V, K'''(0)=κ3 (finite differences)
    h = 1e-4
    K = lambda s: float(vs.cgf(Q, F, s))  # noqa: E731
    assert K(0.0) == pytest.approx(0.0, abs=1e-15)
    assert (K(h) - K(-h)) / (2 * h) == pytest.approx(vs.aggregate_fpr(Q, F), abs=1e-8)
    assert (K(h) - 2 * K(0) + K(-h)) / h**2 == pytest.approx(vs.weighted_var(Q, F), abs=1e-6)
    assert vs.cgf_prime(Q, F, 0.0) == pytest.approx(vs.aggregate_fpr(Q, F), abs=1e-14)
    assert vs.cgf_second(Q, F, 0.0) == pytest.approx(vs.weighted_var(Q, F), abs=1e-14)
    # third cumulant: one central difference of the analytic K'' (a triple FD of K is roundoff-dominated)
    k3_fd = (vs.cgf_second(Q, F, h) - vs.cgf_second(Q, F, -h)) / (2 * h)
    assert k3_fd == pytest.approx(vs.kappa3(Q, F), abs=1e-8)


def test_cgf_vectorised_matches_scalar():
    s = np.array([0.0, 0.5, 3.0, 50.0])
    vec = vs.cgf(Q, F, s)
    assert vec.shape == s.shape
    for si, vi in zip(s, vec):
        assert float(vs.cgf(Q, F, si)) == pytest.approx(vi, rel=1e-14)


def test_cgf_large_s_is_stable():
    # K(s) ~ f_max s + log q_argmax
    s = 1e4
    assert float(vs.cgf(Q, F, s)) == pytest.approx(0.7 * s + np.log(0.2), rel=1e-12)


def test_youden_and_worst():
    assert vs.youden(0.9, Q, F) == pytest.approx(0.9 - 0.31)
    assert vs.j_worst(0.9, F) == pytest.approx(0.2)


def test_iqr_and_entropy():
    assert vs.iqr_weighted(Q, F) == pytest.approx(0.4 - 0.1)
    assert vs.acceptance_entropy(Q, np.zeros(3)) == 0.0
    w = Q * F / (Q @ F)
    assert vs.acceptance_entropy(Q, F) == pytest.approx(-(w * np.log(w)).sum())


def test_input_validation():
    with pytest.raises(ValueError):
        vs.aggregate_fpr([0.5, 0.6], [0.1, 0.1])
    with pytest.raises(ValueError):
        vs.aggregate_fpr([0.5, 0.5], [0.1, 1.2])
