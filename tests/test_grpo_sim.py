"""Correctness of the finite-G GRPO engine (Phase 2)."""

import numpy as np
import pytest

from simulations import sim_grpo_tabular as sg


def test_expected_group_gradient_without_std_is_scaled_vanilla_pg():
    """E[Σ_g (R_g - R̄)(e_y - π)]/G = (1 - 1/G) p ⊙ (ρ - r̄) for mean-centred, unscaled advantages."""
    rng = np.random.default_rng(0)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    rho = np.array([0.9, 0.5, 0.1, 0.7])
    G, n = 8, 200_000
    probs = np.tile(p, (1, n, 1))
    rr = np.tile(rho, (1, n, 1))
    grad, _, _ = sg.grpo_group_gradient(probs, rr, rng, G, scale_rewards=False, eps_std=1e-4)
    est = grad[0].sum(0)  # mean over n groups, since grad is divided by B·G with B = n
    expected = (1 - 1 / G) * p * (rho - p @ rho) / G * G  # per-group sum / G
    se = 4e-3
    np.testing.assert_allclose(est, expected, atol=se)


def test_rewards_follow_configured_rates():
    rng = np.random.default_rng(1)
    rho = np.array([0.8, 0.2, 0.6])
    K = 3
    n = 100_000
    probs = np.tile(np.array([1 / 3] * 3), (1, n, 1))
    rr = np.tile(rho, (1, n, 1))
    # replicate the sampling internals
    grad, rbar, tacc = sg.grpo_group_gradient(probs, rr, rng, 4, True, 1e-4)
    assert rbar[0] == pytest.approx(rho.mean(), abs=5e-3)
    assert tacc[0] == pytest.approx(1 / 3, abs=5e-3)


@pytest.mark.parametrize("policy", ["linear", "mlp"])
def test_param_grads_match_finite_differences(policy):
    """Analytic backprop vs finite differences of the surrogate S = Σ g_delta · Δ."""
    rng = np.random.default_rng(2)
    N, K, d, hdim, B = 2, 5, 4, 6, 3
    psi = rng.normal(size=(7, d))
    idx = np.array([0, 3, 5])
    if policy == "linear":
        params = [rng.normal(size=(N, K)), rng.normal(size=(N, K, d))]
    else:
        params = [rng.normal(size=(N, K)), rng.normal(size=(N, K, hdim)), rng.normal(size=(N, hdim, d))]
    g_delta = rng.normal(size=(N, B, K))

    def S(ps):
        dl, _ = sg.policy_delta(policy, ps, idx, psi[idx])
        return (g_delta * dl).sum()

    _, h = sg.policy_delta(policy, params, idx, psi[idx])
    grads = sg.policy_param_grads(policy, params, idx, psi[idx], g_delta, h)
    eps = 1e-6
    for pi, p in enumerate(params):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            j = it.multi_index
            old = p[j]
            p[j] = old + eps
            up = S(params)
            p[j] = old - eps
            dn = S(params)
            p[j] = old
            num[j] = (up - dn) / (2 * eps)
        np.testing.assert_allclose(grads[pi], num, rtol=1e-5, atol=1e-7)


def test_coupling_matrix_shape_and_rows():
    M = sg.coupling_matrix(5, 0.6)
    assert M[0, 0] == 1 and (M[0, 1:] == 0).all()
    np.testing.assert_allclose(M[1:, 1:].sum(1), 1.0)
    np.testing.assert_allclose(sg.coupling_matrix(5, 0.0), np.eye(5))


def test_homogeneous_vs_perfect_verifier_learns():
    """Sanity: with a perfect verifier accuracy rises; a verifier rewarding one error mode above τ collapses it."""
    rng = np.random.default_rng(3)
    pop = sg.make_population(np.array([0.5, 0.3, 0.2]), 0.4, 32, rng)
    cfg = sg.GRPOSimConfig(policy="tabular", optimizer="adam", lr=0.05, steps=150, prompts_per_step=16, G=8)
    rho = np.array([[1.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 1.0]])
    res = sg.simulate(pop, rho, cfg, seed=0)
    assert res["acc"][-1, 0] > 0.9
    assert res["acc"][-1, 1] < res["acc"][0, 1]
    # reward-accuracy divergence in the hacked run
    assert res["train_reward"][-20:, 1].mean() > res["train_true"][-20:, 1].mean() * 0.6 + 0.3


def test_common_random_numbers_give_identical_runs_for_identical_configs():
    rng = np.random.default_rng(4)
    pop = sg.make_population(np.array([0.6, 0.4]), 0.4, 16, rng)
    cfg = sg.GRPOSimConfig(policy="linear", optimizer="adam", lr=0.02, steps=30, prompts_per_step=8, G=4)
    r1 = sg.simulate(pop, np.array([[0.9, 0.3, 0.3]]), cfg, seed=5, batch_seed=9)
    r2 = sg.simulate(pop, np.array([[0.9, 0.3, 0.3]]), cfg, seed=5, batch_seed=9)
    np.testing.assert_array_equal(r1["acc"], r2["acc"])
