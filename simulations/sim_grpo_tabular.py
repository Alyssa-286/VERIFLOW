"""Finite-group GRPO on class-level policies (master plan §12.2-12.3; departures B1-B3, B6-B8).

Level of claim: numerical simulation. Policies act on outcome *classes*
[C, designed modes..., U, I], not tokens; the point is to see how far the
idealized predictions survive finite groups, Bernoulli verifiers, group
normalisation, prompt heterogeneity, shared parameters and Adam.

Policy family (one engine for all variants)
------------------------------------------
logits(x) = L0(x) + M · Δ(x)
  L0(x)  fixed base-policy logits for prompt x ("pretrained model"); prompt-heterogeneous
  M      K×K mode-coupling matrix; on the incorrect block M = (1-κ)I + κ·11ᵀ/n_err
         (κ = 0: modes independent; κ → 1: modes generalise to one another)
  Δ(x)   learned:
         "tabular": free vector per prompt (θ_x)                          — B1-B3, B7 only
         "linear" : b + U ψ(x), shared across prompts                     — adds B6
         "mlp"    : b + U tanh(H ψ(x)), H also learned                    — adds B6 with features
Optimiser: "sgd" or "adam" (per-config moments), optional L2 pull to init (KL proxy, B5).

The engine is vectorised over N independent runs (config × seed) with common
random numbers for prompt selection, so configurations are compared on the same batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logit


@dataclass
class GRPOSimConfig:
    policy: str = "tabular"  # tabular | linear | mlp
    optimizer: str = "adam"  # sgd | adam
    lr: float = 0.05
    steps: int = 300
    prompts_per_step: int = 16
    G: int = 8
    scale_rewards: bool = True  # GRPO std normalisation (False = Dr.GRPO-style)
    coupling: float = 0.0  # κ
    l2_to_init: float = 0.0  # λ
    feat_dim: int = 8
    hidden: int = 16
    eval_every: int = 10
    adam_betas: tuple = (0.9, 0.999)
    eps_std: float = 1e-4


@dataclass
class PromptPopulation:
    """Base-policy class probabilities per prompt, p0 (P, K), and prompt features ψ (P, d)."""

    p0: np.ndarray
    psi: np.ndarray
    class_names: list = field(default_factory=list)

    @property
    def q_ref(self) -> np.ndarray:
        """Pooled conditional error distribution (what a census would estimate)."""
        err = self.p0[:, 1:]
        return err.sum(0) / err.sum()

    @property
    def A0(self) -> float:
        return float(self.p0[:, 0].mean())


def make_population(q_ref: np.ndarray, A0_mean: float, n_prompts: int, rng, prompt_conc: float = 20.0,
                    acc_conc: float = 6.0, feat_dim: int = 8, class_names=None) -> PromptPopulation:
    """Heterogeneous prompts: A0(x) ~ Beta(mean A0_mean), q(x) ~ Dirichlet(prompt_conc · q_ref)."""
    q_ref = np.asarray(q_ref, float)
    a = rng.beta(A0_mean * acc_conc, (1 - A0_mean) * acc_conc, n_prompts)
    a = np.clip(a, 0.02, 0.98)
    if np.isinf(prompt_conc):
        qx = np.tile(q_ref, (n_prompts, 1))
    else:
        qx = rng.dirichlet(prompt_conc * q_ref, n_prompts)
    qx = np.clip(qx, 1e-6, None)
    qx /= qx.sum(1, keepdims=True)
    p0 = np.c_[a, (1 - a)[:, None] * qx]
    psi = rng.normal(size=(n_prompts, feat_dim)) / np.sqrt(feat_dim)
    return PromptPopulation(p0, psi, class_names or [])


def coupling_matrix(K: int, kappa: float) -> np.ndarray:
    M = np.eye(K)
    n_err = K - 1
    M[1:, 1:] = (1 - kappa) * np.eye(n_err) + kappa * np.ones((n_err, n_err)) / n_err
    return M


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class _Adam:
    def __init__(self, shapes, b1, b2, eps=1e-8):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def step(self, grads, lr):
        self.t += 1
        out = []
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            out.append(lr * mh / (np.sqrt(vh) + self.eps))
        return out


def policy_delta(policy: str, params: list, idx: np.ndarray, psi_idx: np.ndarray):
    """Δ(x) for prompts idx: (N, len(idx), K), plus cached hidden activations (mlp only)."""
    if policy == "tabular":
        return params[0][:, idx, :], None
    if policy == "linear":
        b, U = params
        return b[:, None, :] + np.einsum("nkd,bd->nbk", U, psi_idx), None
    b, U, H = params
    h = np.tanh(np.einsum("nhd,bd->nbh", H, psi_idx))
    return b[:, None, :] + np.einsum("nkh,nbh->nbk", U, h), h


def policy_param_grads(policy: str, params: list, idx: np.ndarray, psi_idx: np.ndarray, g_delta: np.ndarray, h):
    """Backpropagate dL/dΔ (N, B, K) to the parameters of each policy family."""
    if policy == "tabular":
        gt = np.zeros_like(params[0])
        np.add.at(gt, (slice(None), idx, slice(None)), g_delta)
        return [gt]
    if policy == "linear":
        return [g_delta.sum(1), np.einsum("nbk,bd->nkd", g_delta, psi_idx)]
    b, U, H = params
    gU = np.einsum("nbk,nbh->nkh", g_delta, h)
    gh = np.einsum("nbk,nkh->nbh", g_delta, U) * (1 - h**2)
    gH = np.einsum("nbh,bd->nhd", gh, psi_idx)
    return [g_delta.sum(1), gU, gH]


def grpo_group_gradient(probs, rho, rng, G, scale_rewards, eps_std):
    """Sample G classes per (run, prompt), Bernoulli rewards, GRPO advantages.

    probs, rho: (N, B, K). Returns (grad wrt logits (N,B,K), mean assigned reward, mean true correctness).
    The loss is averaged over all N·B·G samples per run (sequence-level analogue of 'dapo' normalisation).
    """
    N, B, K = probs.shape
    cdf = np.cumsum(probs, -1)
    u = rng.random((N, B, G, 1))
    y = (u > cdf[:, :, None, :]).sum(-1)  # (N,B,G) class index
    y = np.minimum(y, K - 1)
    r_prob = np.take_along_axis(rho, y, axis=-1)
    R = (rng.random((N, B, G)) < r_prob).astype(float)
    A = R - R.mean(-1, keepdims=True)
    if scale_rewards:
        A = A / (R.std(-1, keepdims=True) + eps_std)
    onehot = np.zeros((N, B, G, K))
    np.put_along_axis(onehot, y[..., None], 1.0, axis=-1)
    grad = (A[..., None] * (onehot - probs[:, :, None, :])).sum(2) / (B * G)
    return grad, R.mean((1, 2)), (y == 0).mean((1, 2))


def simulate(pop: PromptPopulation, rho: np.ndarray, cfg: GRPOSimConfig, seed: int, batch_seed: int | None = None):
    """Run N GRPO simulations in parallel. rho: (N, K) acceptance probabilities [τ, f...].

    batch_seed controls prompt sampling (shared across runs: common random numbers);
    seed controls rollouts and verifier draws.
    Returns dict with eval steps, accuracy (n_eval, N), pooled error-mode distribution (n_eval, N, K-1),
    and training reward/true-accuracy traces.
    """
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed if batch_seed is None else batch_seed)
    prng = np.random.default_rng(12345)  # parameter init, fixed across configs (same 'pretrained' model)
    rho = np.atleast_2d(np.asarray(rho, float))
    N, K = rho.shape
    P, d = pop.psi.shape
    L0 = np.log(pop.p0)  # (P, K)
    M = coupling_matrix(K, cfg.coupling)

    if cfg.policy == "tabular":
        params = [np.zeros((N, P, K))]
    elif cfg.policy == "linear":
        params = [np.zeros((N, K)), np.zeros((N, K, d))]
    elif cfg.policy == "mlp":
        H0 = prng.normal(size=(cfg.hidden, d)) / np.sqrt(d)
        params = [np.zeros((N, K)), np.zeros((N, K, cfg.hidden)), np.tile(H0, (N, 1, 1))]
    else:
        raise ValueError(cfg.policy)
    init = [p.copy() for p in params]
    opt = _Adam([p.shape for p in params], *cfg.adam_betas) if cfg.optimizer == "adam" else None

    def delta(idx):
        return policy_delta(cfg.policy, params, idx, pop.psi[idx])

    def all_probs():
        idx = np.arange(P)
        dl, _ = delta(idx)
        return _softmax(L0[None] + np.einsum("kj,nbj->nbk", M, dl))

    ev_steps, acc_tr, mode_tr, rew_tr, true_tr = [], [], [], [], []

    def record(step):
        pr = all_probs()
        ev_steps.append(step)
        acc_tr.append(pr[:, :, 0].mean(1))
        err = pr[:, :, 1:].sum(1)
        mode_tr.append(err / err.sum(-1, keepdims=True))

    record(0)
    for step in range(1, cfg.steps + 1):
        idx = brng.choice(P, cfg.prompts_per_step, replace=False)
        dl, h = delta(idx)
        probs = _softmax(L0[idx][None] + np.einsum("kj,nbj->nbk", M, dl))
        g_logit, rbar, tacc = grpo_group_gradient(probs, rho[:, None, :].repeat(len(idx), 1), rng, cfg.G, cfg.scale_rewards, cfg.eps_std)
        rew_tr.append(rbar)
        true_tr.append(tacc)
        g_delta = np.einsum("kj,nbk->nbj", M, g_logit)  # chain rule through M (Mᵀ g)
        grads = policy_param_grads(cfg.policy, params, idx, pop.psi[idx], g_delta, h)
        if cfg.l2_to_init > 0:
            grads = [g - cfg.l2_to_init * (p - p0) for g, p, p0 in zip(grads, params, init)]
        updates = opt.step(grads, cfg.lr) if opt is not None else [cfg.lr * g for g in grads]
        for p, u in zip(params, updates):
            p += u  # gradient ascent on expected reward
        if step % cfg.eval_every == 0:
            record(step)

    return {
        "steps": np.array(ev_steps),
        "acc": np.array(acc_tr),
        "modes": np.array(mode_tr),
        "train_reward": np.array(rew_tr),
        "train_true": np.array(true_tr),
    }


def window_outcome(steps: np.ndarray, acc: np.ndarray, T: int) -> np.ndarray:
    """Primary-outcome analogue: logit of mean accuracy over evals in [T/2, T]. acc: (n_eval, N)."""
    w = (steps >= T / 2) & (steps <= T)
    return logit(np.clip(acc[w].mean(0), 1e-9, 1 - 1e-9))
