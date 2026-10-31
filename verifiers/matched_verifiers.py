"""Exact matched verifier configurations (master plan §10.2).

Given the frozen reference distribution q_ref over incorrect classes
(designed modes + "U" unclassified + "I" invalid), generate per-mode
false-acceptance vectors that hit target (F̄, V[, f_max]) *exactly*, with
f_U = F̄ and f_I = 0 fixed (conventions D1.2).

Geometry. Write the designed-mode values as x = F̄ + a·1 + z with Σ_d q_d z_d = 0.
The linear constraint fixes the scalar a; the variance constraint fixes the
q-weighted norm of z. So for any direction z we can rescale to satisfy both
constraints in closed form, then accept if 0 ≤ x ≤ 1 (and x ≤ f_max). This makes
matching exact up to floating point and lets random directions randomise which
modes are accepted more (mode-identity randomisation).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from verifiers import stats as vs

TOL = 1e-12


@dataclass(frozen=True)
class VerifierConfig:
    config_id: str
    tau: float
    f: dict  # class name -> false-acceptance prob (designed modes, "U", "I")
    q_ref: dict  # class name -> reference mass (sums to 1)
    target_Fbar: float
    target_V: float
    target_fmax: float | None
    grid_cell: str
    generator_seed: int

    def vectors(self):
        names = list(self.q_ref)
        return names, np.array([self.q_ref[k] for k in names]), np.array([self.f[k] for k in names])

    @property
    def Fbar(self) -> float:
        _, q, f = self.vectors()
        return vs.aggregate_fpr(q, f)

    @property
    def V(self) -> float:
        _, q, f = self.vectors()
        return vs.weighted_var(q, f)

    def canonical_json(self) -> str:
        d = asdict(self)
        d["f"] = {k: repr(float(v)) for k, v in sorted(self.f.items())}
        d["q_ref"] = {k: repr(float(v)) for k, v in sorted(self.q_ref.items())}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def predictors(self, s: float | None = None) -> dict:
        _, q, f = self.vectors()
        return vs.all_predictors(self.tau, q, f, s)


def _split(q_ref: dict, designed: list[str]):
    for k in designed:
        if k in ("U", "I"):
            raise ValueError("U and I are not designed modes")
    missing = set(designed) - set(q_ref)
    if missing:
        raise ValueError(f"designed modes missing from q_ref: {missing}")
    q = np.array([q_ref[k] for k in designed], float)
    qU = float(q_ref.get("U", 0.0))
    qI = float(q_ref.get("I", 0.0))
    total = q.sum() + qU + qI
    if abs(total - 1) > 1e-9 or set(q_ref) - set(designed) - {"U", "I"}:
        raise ValueError("q_ref must cover exactly designed + U + I and sum to 1")
    return q, qU, qI


def designed_targets(q_ref: dict, designed: list[str], Fbar: float, V: float, fixed: dict | None = None):
    """Return (S, W): required q-weighted sum and q-weighted sq. deviation over the *free* designed modes.

    Over all incorrect classes: Σ q f = F̄ and Σ q (f-F̄)² = V, with f_U = F̄ (contributes 0 to both
    deviations) and f_I = 0 (contributes -q_I F̄ to the sum and q_I F̄² to V). `fixed` pins some designed
    modes (e.g. the f_max mode).
    """
    q, qU, qI = _split(q_ref, designed)
    fixed = fixed or {}
    # Σ_d q_d f_d = F̄ - q_U F̄ - q_I·0  =  F̄ (1 - q_U)
    S = Fbar * (1 - qU)
    W = V - qI * Fbar**2
    for k, val in fixed.items():
        qk = q_ref[k]
        S -= qk * val
        W -= qk * (val - Fbar) ** 2
    return S, W


def v_floor(q_ref: dict, Fbar: float) -> float:
    """Minimum attainable V: all designed modes equal (= value making F̄ exact), plus the I contribution."""
    designed = [k for k in q_ref if k not in ("U", "I")]
    q, qU, qI = _split(q_ref, designed)
    x = Fbar * (1 - qU) / q.sum()  # common designed value
    return float(q.sum() * (x - Fbar) ** 2 + qI * Fbar**2)


def _solve_direction(qf: np.ndarray, S: float, W: float, z: np.ndarray, Fbar: float):
    """Map an arbitrary direction z to x with Σ q x = S and Σ q (x-F̄)² = W exactly. None if impossible."""
    Q = qf.sum()
    mean_x = S / Q  # q-weighted mean of x over free modes
    base = (mean_x - Fbar) ** 2 * Q  # Σ q (x - F̄)² contributed by the mean shift
    need = W - base
    if need < -1e-14:
        return None
    z = z - (qf @ z) / Q  # centre: Σ q z = 0
    nz = qf @ z**2
    if need <= 1e-15:
        return np.full_like(qf, mean_x)
    if nz <= 0:
        return None
    return mean_x + z * np.sqrt(need / nz)


def generate_config(
    q_ref: dict,
    designed: list[str],
    Fbar: float,
    V: float,
    tau: float,
    rng: np.random.Generator,
    fmax: float | None = None,
    fmax_mode: str | None = None,
    config_id: str = "cfg",
    grid_cell: str = "",
    generator_seed: int = 0,
    max_tries: int = 20_000,
) -> VerifierConfig:
    """Random configuration hitting (F̄, V[, f_max]) exactly. Raises ValueError if not found (likely infeasible)."""
    if not (0 <= Fbar <= 1 and 0 <= tau <= 1):
        raise ValueError("rates must lie in [0,1]")
    fixed = {}
    free = list(designed)
    ub = 1.0
    if fmax is not None:
        fmax_mode = fmax_mode or designed[int(rng.integers(len(designed)))]
        fixed[fmax_mode] = fmax
        free.remove(fmax_mode)
        ub = fmax
    S, W = designed_targets(q_ref, designed, Fbar, V, fixed)
    qf = np.array([q_ref[k] for k in free], float)
    if len(free) == 0:
        raise ValueError("no free modes")
    for _ in range(max_tries):
        # mixture of direction types: dense Gaussian and sparse (one or two big modes)
        if rng.random() < 0.5:
            z = rng.normal(size=len(free))
        else:
            z = np.zeros(len(free))
            z[rng.integers(len(free))] = 1.0
            if rng.random() < 0.5:
                z[rng.integers(len(free))] -= rng.uniform(0, 1)
        x = _solve_direction(qf, S, W, z, Fbar)
        if x is None:
            continue
        if (x >= -TOL).all() and (x <= ub + TOL).all():
            x = np.clip(x, 0.0, ub)
            f = {**dict(zip(free, x.tolist())), **fixed, "U": Fbar, "I": 0.0}
            cfg = VerifierConfig(config_id, float(tau), f, dict(q_ref), Fbar, V, fmax, grid_cell, generator_seed)
            # exactness check (clipping only moved values by ≤ TOL)
            if abs(cfg.Fbar - Fbar) <= 1e-11 and abs(cfg.V - V) <= 1e-11:
                return cfg
    raise ValueError(f"could not reach F̄={Fbar}, V={V}, fmax={fmax}; target may be infeasible")


def homogeneous_config(q_ref: dict, Fbar: float, tau: float, config_id="cfg_homog") -> VerifierConfig:
    """Minimum-dispersion configuration (V = v_floor): all designed modes share one value."""
    designed = [k for k in q_ref if k not in ("U", "I")]
    q, qU, _ = _split(q_ref, designed)
    x = Fbar * (1 - qU) / q.sum()
    if not 0 <= x <= 1:
        raise ValueError("infeasible homogeneous config")
    f = {**{k: x for k in designed}, "U": Fbar, "I": 0.0}
    return VerifierConfig(config_id, float(tau), f, dict(q_ref), Fbar, v_floor(q_ref, Fbar), None, "homog", 0)


def v_min_with_mode(q_ref: dict, Fbar: float, fmax: float, mode: str) -> float:
    """Smallest V attainable when `mode` is pinned at f_max (other designed modes equal, if within [0, f_max])."""
    designed = [k for k in q_ref if k not in ("U", "I")]
    S, _ = designed_targets(q_ref, designed, Fbar, 0.0, {mode: fmax})
    free = [k for k in designed if k != mode]
    qs = sum(q_ref[k] for k in free)
    x = S / qs
    if not (-1e-12 <= x <= fmax + 1e-12):
        return float("inf")
    names = list(q_ref)
    f = {**{k: x for k in free}, mode: fmax, "U": Fbar, "I": 0.0}
    return vs.weighted_var(np.array([q_ref[k] for k in names]), np.array([f[k] for k in names]))


def feasible_v_range_by_mode(q_ref: dict, Fbar: float, fmax: float) -> dict:
    """{mode: (V_min, V_max)} when that mode carries f_max. Modes with an empty range are omitted."""
    designed = [k for k in q_ref if k not in ("U", "I")]
    out = {}
    for m in designed:
        lo = v_min_with_mode(q_ref, Fbar, fmax, m)
        hi = feasible_v_max(q_ref, Fbar, fmax, modes=[m])
        if np.isfinite(lo) and hi >= lo:
            out[m] = (lo, hi)
    return out


def feasible_v_max(q_ref: dict, Fbar: float, fmax: float | None = None, modes: list[str] | None = None) -> float:
    """Largest V attainable at F̄ (and optional f_max carried by one of `modes`), by vertex search.

    Maximising a convex function over the polytope {0≤x≤ub, Σ q x = S} attains its max at a vertex,
    where all but at most one coordinate sit at a bound. We enumerate those vertices directly.
    """
    designed = [k for k in q_ref if k not in ("U", "I")]
    q, qU, qI = _split(q_ref, designed)
    best = -np.inf
    if fmax is None:
        candidates = [(None, designed, 1.0)]
    else:
        candidates = [(m, [k for k in designed if k != m], fmax) for m in (modes or designed)]
    for m, free, ub in candidates:
        fixed = {m: fmax} if m is not None else {}
        S, _ = designed_targets(q_ref, designed, Fbar, 0.0, fixed)
        qf = np.array([q_ref[k] for k in free])
        n = len(free)
        if n > 14:
            raise ValueError("too many modes for exhaustive vertex search")
        for mask in range(2**n):
            hi = np.array([(mask >> i) & 1 for i in range(n)], bool)
            for j in range(n):  # coordinate j is the "free" one
                x = np.where(hi, ub, 0.0).astype(float)
                rest = S - (qf * x).sum() + qf[j] * x[j]
                xj = rest / qf[j]
                if -1e-12 <= xj <= ub + 1e-12:
                    x[j] = min(max(xj, 0.0), ub)
                    f = {**dict(zip(free, x)), **fixed, "U": Fbar, "I": 0.0}
                    names = list(q_ref)
                    qq = np.array([q_ref[k] for k in names])
                    ff = np.array([f[k] for k in names])
                    if abs(qq @ ff - Fbar) < 1e-9:
                        best = max(best, vs.weighted_var(qq, ff))
    return float(best)


def design_grid(
    q_ref: dict,
    Fbar: float,
    tau: float,
    V_fracs: list[float],
    fmax_levels: list[float | None],
    n_per_cell: int,
    seed: int,
    prefix: str = "cfg",
    include_homogeneous: bool = True,
) -> list[VerifierConfig]:
    """De-confounding (V, f_max) grid (§10.2).

    For f_max = None, V = floor + frac·(V_max - floor). For a numeric f_max, the feasible V range depends on
    which mode carries f_max; V = lo* + frac·(hi* - lo*) over the union of per-mode ranges, and the f_max mode
    is drawn at random among modes whose range contains V (mode-identity randomisation). Configs that cannot
    be generated are skipped, so callers must check cell sizes.
    """
    designed = [k for k in q_ref if k not in ("U", "I")]
    rng = np.random.default_rng(seed)
    out = []
    if include_homogeneous:
        out.append(homogeneous_config(q_ref, Fbar, tau, f"{prefix}000"))
    idx = len(out)
    for fm in fmax_levels:
        if fm is None:
            ranges = None
            lo, hi = v_floor(q_ref, Fbar), feasible_v_max(q_ref, Fbar)
        else:
            ranges = feasible_v_range_by_mode(q_ref, Fbar, fm)
            if not ranges:
                continue
            lo, hi = min(r[0] for r in ranges.values()), max(r[1] for r in ranges.values())
        for vf in V_fracs:
            V = lo + vf * (hi - lo)
            cell = f"fmax={fm}|Vfrac={vf}"
            eligible = None if ranges is None else [m for m, (a, b) in ranges.items() if a - 1e-12 <= V <= b + 1e-12]
            if eligible == []:
                continue
            for _ in range(n_per_cell):
                mode = None if eligible is None else eligible[int(rng.integers(len(eligible)))]
                try:
                    cfg = generate_config(
                        q_ref, designed, Fbar, V, tau, rng, fmax=fm, fmax_mode=mode,
                        config_id=f"{prefix}{idx:03d}", grid_cell=cell, generator_seed=seed,
                    )
                except ValueError:
                    continue
                out.append(cfg)
                idx += 1
    return out


def split_heldout(configs: list[VerifierConfig], heldout_frac: float = 1 / 3, seed: int = 0):
    """Stratified (by grid cell) random split into (train_configs, heldout_configs)."""
    rng = np.random.default_rng(seed)
    cells: dict[str, list] = {}
    for c in configs:
        cells.setdefault(c.grid_cell, []).append(c)
    train, held = [], []
    for cell in sorted(cells):
        group = cells[cell]
        order = rng.permutation(len(group))
        k = int(round(heldout_frac * len(group)))
        if cell == "homog":
            k = 0  # the homogeneous control always stays in training
        for r, i in enumerate(order):
            (held if r < k else train).append(group[i])
    return train, held


def save_configs(configs: list[VerifierConfig], path) -> None:
    import pandas as pd
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in configs:
        (path / f"{c.config_id}.json").write_text(c.canonical_json())
        rows.append({"config_id": c.config_id, "sha256": c.sha256, "grid_cell": c.grid_cell, **c.predictors()})
    pd.DataFrame(rows).to_csv(path / "index.csv", index=False)
