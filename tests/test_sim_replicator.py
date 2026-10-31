"""Gate 1 tests (master plan §19.1). Each test names the proposition it checks."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import logit

from simulations import sim_replicator as sr
from verifiers import stats as vs


def _random_case(rng, M=None):
    M = M or int(rng.integers(2, 9))
    q0 = rng.dirichlet(np.ones(M))
    f = rng.uniform(0, 1, M)
    tau = rng.uniform(0.3, 1.0)
    A0 = rng.uniform(0.05, 0.95)
    return A0, q0, tau, f


def test_closed_form_matches_ode_100_random_cases():
    """Prop. 2: ODE vs closed form, ≤1e-10 on 100 random (q0, f, τ, A0)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 21)
    worst = 0.0
    for _ in range(100):
        A0, q0, tau, f = _random_case(rng)
        p = sr.integrate_replicator(A0, q0, tau, f, t)
        worst = max(worst, np.abs(p[:, 0] - sr.closed_form_accuracy(A0, q0, tau, f, t)).max())
    assert worst <= 1e-10, worst


def test_error_mode_closure_prop1():
    """Prop. 1: q(t) is the exponential tilt of q(0), independent of τ and A0."""
    rng = np.random.default_rng(1)
    A0, q0, tau, f = _random_case(rng, M=5)
    t = np.array([0.0, 1.0, 4.0, 9.0])
    for tau_ in (tau, 0.2):
        p = sr.integrate_replicator(A0, q0, tau_, f, t)
        q_t = p[:, 1:] / p[:, 1:].sum(1, keepdims=True)
        for i, ti in enumerate(t):
            np.testing.assert_allclose(q_t[i], vs.tilted(q0, f, ti), atol=1e-10)


def test_homogeneous_verifier_is_linear_in_logit():
    """V = 0 ⇒ logit A(t) = logit A0 + J t exactly (pure rate rescaling)."""
    q0 = np.array([0.2, 0.5, 0.3])
    f = np.full(3, 0.35)
    tau, A0 = 0.8, 0.3
    t = np.linspace(0, 20, 41)
    lg = sr.closed_form_logit_accuracy(A0, q0, tau, f, t)
    np.testing.assert_allclose(lg, logit(A0) + (tau - 0.35) * t, atol=1e-12)
    p = sr.integrate_replicator(A0, q0, tau, f, t)
    np.testing.assert_allclose(logit(p[:, 0]), logit(A0) + (tau - 0.35) * t, atol=1e-8)


def test_initial_fpr_drift_equals_variance_prop3():
    """Prop. 3: dF/dt at 0 = V (finite difference on the ODE)."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        A0, q0, tau, f = _random_case(rng)
        h = 1e-5
        p = sr.integrate_replicator(A0, q0, tau, f, np.array([h, 2 * h]))
        q_h = p[0, 1:] / p[0, 1:].sum()
        q_2h = p[1, 1:] / p[1, 1:].sum()
        F0 = q0 @ f
        # second-order one-sided difference
        dF = (-3 * F0 + 4 * (q_h @ f) - (q_2h @ f)) / (2 * h)
        assert dF == pytest.approx(vs.weighted_var(q0, f), abs=1e-6)


def test_peak_time_and_collapse_prop4():
    """Prop. 4: f_max > τ ⇒ interior peak at K'(t*) = τ and A → 0."""
    q0 = np.array([0.5, 0.3, 0.2])
    f = np.array([0.1, 0.3, 0.95])
    tau, A0 = 0.8, 0.4
    ts = sr.peak_time(q0, f, tau)
    assert np.isfinite(ts) and ts > 0
    assert vs.cgf_prime(q0, f, ts) == pytest.approx(tau, abs=1e-10)
    grid = np.linspace(max(ts - 1, 0), ts + 1, 200001)
    acc = sr.closed_form_accuracy(A0, q0, tau, f, grid)
    assert grid[acc.argmax()] == pytest.approx(ts, abs=1e-4)
    assert sr.closed_form_accuracy(A0, q0, tau, f, 400.0) < 1e-6
    # ODE agrees on the peak location
    t = np.linspace(0, 2 * ts, 4001)
    p = sr.integrate_replicator(A0, q0, tau, f, t)
    assert t[p[:, 0].argmax()] == pytest.approx(ts, abs=2 * ts / 4000)


def test_no_collapse_when_worst_mode_below_tau_prop4():
    q0 = np.array([0.6, 0.4])
    f = np.array([0.2, 0.7])
    tau, A0 = 0.8, 0.4
    assert sr.peak_time(q0, f, tau) == float("inf")
    assert sr.closed_form_accuracy(A0, q0, tau, f, 1000.0) > 1 - 1e-6
    # asymptotic logit delay vs ideal is -log q_argmax (Prop. 4)
    t = 2000.0
    lg = sr.closed_form_logit_accuracy(A0, q0, tau, f, t)
    assert lg - (logit(A0) + (tau - 0.7) * t) == pytest.approx(-np.log(0.4), abs=1e-9)


def test_matched_pair_quadratic_divergence_cor3a():
    """Cor. 3a: matched F̄ and τ ⇒ Δlogit ≈ -(V_B - V_A) t²/2 as t → 0."""
    q0 = np.array([0.4, 0.35, 0.25])
    fA = np.full(3, 0.3)
    fB = np.array([0.05, 0.35, 0.63])
    fB = fB - q0 @ fB + 0.3
    assert q0 @ fB == pytest.approx(0.3, abs=1e-15)
    dV = vs.weighted_var(q0, fB) - vs.weighted_var(q0, fA)
    ratios = []
    for t in [1e-1, 1e-2, 1e-3]:
        d = sr.closed_form_logit_accuracy(0.4, q0, 0.9, fB, t) - sr.closed_form_logit_accuracy(0.4, q0, 0.9, fA, t)
        ratios.append(d / (-dV * t**2 / 2))
    assert abs(ratios[-1] - 1) < 1e-3
    assert abs(ratios[-1] - 1) < abs(ratios[0] - 1)


def test_gibbs_optimum_matches_long_run_kl_flow_prop6():
    """Prop. 6: the KL-regularised NPG flow converges to the Gibbs optimum (= closed form at t = 1/β)."""
    rng = np.random.default_rng(3)
    for beta in (0.5, 0.1):
        A0, q0, tau, f = _random_case(rng, M=4)
        p_inf = sr.integrate_kl_npg(A0, q0, tau, f, beta, T=60.0 / beta)
        assert logit(p_inf[0]) == pytest.approx(sr.gibbs_logit_accuracy(A0, q0, tau, f, beta), abs=1e-8)


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=0, max_value=10_000), st.floats(min_value=0.0, max_value=30.0))
def test_permutation_invariance(seed, t):
    """Jointly permuting modes in q and f leaves A(t) unchanged."""
    rng = np.random.default_rng(seed)
    A0, q0, tau, f = _random_case(rng)
    perm = rng.permutation(len(q0))
    a = sr.closed_form_accuracy(A0, q0, tau, f, t)
    b = sr.closed_form_accuracy(A0, q0[perm], tau, f[perm], t)
    assert a == pytest.approx(b, rel=1e-12, abs=1e-300)


def test_tabular_npg_matches_closed_form():
    """The vectorised tabular simulator with rule='npg' reproduces Prop. 2 (Euler error only)."""
    rng = np.random.default_rng(4)
    q0 = rng.dirichlet(np.ones(4))
    fam = sr.random_matched_family(q0, 0.3, 6, rng)
    times, acc = sr.simulate_tabular("npg", 0.4, np.tile(q0, (6, 1)), 0.9, fam, T=4.0, n_steps=8000, record_every=100)
    exact = np.array([sr.closed_form_accuracy(0.4, q0, 0.9, f, times) for f in fam]).T
    assert np.abs(acc - exact).max() < 1e-3


def test_vpg_drift_formula_prop7():
    """Prop. 7: under vanilla PG, dF/dt = P_E Σ q_k² (f_k - F)(f_k - r̄)."""
    rng = np.random.default_rng(5)
    q0 = rng.dirichlet(np.ones(5))
    f = rng.uniform(0, 1, 5)
    tau, A0 = 0.9, 0.4
    h = 1e-6
    _, _ = sr.simulate_tabular("vpg", A0, q0[None], tau, f[None], T=h, n_steps=1, record_every=1)
    # recompute q after one tiny Euler step directly
    rho = np.r_[tau, f]
    p = sr.initial_state(A0, q0)
    th = np.log(p) + h * p * (rho - p @ rho)
    p1 = np.exp(th) / np.exp(th).sum()
    q1 = p1[1:] / p1[1:].sum()
    dF = (q1 @ f - q0 @ f) / h
    rbar = p @ rho
    pred = (1 - A0) * (q0**2) @ ((f - q0 @ f) * (f - rbar))
    assert dF == pytest.approx(pred, rel=1e-4, abs=1e-9)


def test_matched_family_is_matched():
    rng = np.random.default_rng(6)
    q0 = rng.dirichlet(np.ones(6))
    fam = sr.random_matched_family(q0, 0.25, 50, rng)
    np.testing.assert_allclose(fam @ q0, 0.25, atol=1e-14)
    assert (fam >= 0).all() and (fam <= 1).all()
    assert vs.weighted_var(q0, fam[0]) == 0.0


def test_loco_shortcut_matches_brute_force():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(15, 2))
    y = X @ np.array([1.0, -2.0]) + rng.normal(size=15)
    brute = []
    for i in range(15):
        k = np.arange(15) != i
        Xa = np.c_[np.ones(14), X[k]]
        b, *_ = np.linalg.lstsq(Xa, y[k], rcond=None)
        brute.append(y[i] - np.r_[1, X[i]] @ b)
    assert sr._loco_mse(X, y) == pytest.approx(np.mean(np.square(brute)), rel=1e-10)
