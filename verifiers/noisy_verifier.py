"""Programmatic mode-specific noisy verifier (master plan §10.1).

decide(): classify the completion (taxonomy v1), look up its acceptance probability
ρ(class) from a VerifierConfig, and accept iff u < ρ, where u is a uniform drawn
from a counter-based PRF keyed on
    (verifier_seed, run_seed, step, prompt_id, sample_idx).
The PRF is SHA-256 (D4). Decisions are therefore:
  - bit-reproducible and replayable offline from the log alone;
  - independent of every other RNG in the process (no global numpy/torch state is touched);
  - independent across calls (per-call Bernoulli draws: the primary design).
A deterministic-per-string variant (key on the completion hash instead of step/sample) is
available for the extended-design ablation.

Design credit: the per-class confusion-matrix idea follows the TARGETED bucket strategy of
eth-sri/llm-verifier-noise (Mixup.mixup_rewards, commit 913e3bf, MIT licence). This is an
independent reimplementation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass

from error_modes.classify_errors import TAXONOMY_VERSION, classify
from verifiers.matched_verifiers import VerifierConfig


def prf_uniform(*key) -> float:
    """Uniform in [0,1) from SHA-256 of the canonical key (53 bits of mantissa)."""
    h = hashlib.sha256("\x1f".join(map(str, key)).encode()).digest()
    (x,) = struct.unpack("<Q", h[:8])
    return (x >> 11) * (1.0 / (1 << 53))


@dataclass(frozen=True)
class VerifierRecord:
    run_seed: int
    step: int
    prompt_id: str
    sample_idx: int
    completion_sha256: str
    mode: str
    modes_multi: str  # '|'-joined matching rules
    parsed_value: str  # str to keep large ints exact in logs
    parse_status: str
    correct: bool
    rho: float
    u: float
    reward: float
    verifier_config_id: str
    verifier_config_sha256: str
    taxonomy_version: str


class NoisyVerifier:
    def __init__(self, config: VerifierConfig, verifier_seed: int, run_seed: int, per_string: bool = False):
        missing = {"C"} | set(config.f)
        self.config = config
        self.verifier_seed = int(verifier_seed)
        self.run_seed = int(run_seed)
        self.per_string = per_string
        self._rho = {"C": float(config.tau), **{k: float(v) for k, v in config.f.items()}}
        if not set(self._rho) >= missing:
            raise ValueError("config lacks some classes")
        self._sha = config.sha256

    def rho(self, mode: str) -> float:
        try:
            return self._rho[mode]
        except KeyError:
            raise KeyError(f"verifier config {self.config.config_id} has no rate for class {mode!r}") from None

    def decide(self, a: int, b: int, completion: str, step: int, prompt_id: str, sample_idx: int) -> VerifierRecord:
        cl = classify(a, b, completion)
        csha = hashlib.sha256(completion.encode()).hexdigest()
        rho = self.rho(cl.primary)
        if self.per_string:
            u = prf_uniform("veriflow-v1-str", self.verifier_seed, self.run_seed, prompt_id, csha)
        else:
            u = prf_uniform("veriflow-v1", self.verifier_seed, self.run_seed, step, prompt_id, sample_idx)
        return VerifierRecord(
            run_seed=self.run_seed, step=int(step), prompt_id=str(prompt_id), sample_idx=int(sample_idx),
            completion_sha256=csha, mode=cl.primary, modes_multi="|".join(cl.matches),
            parsed_value="" if cl.value is None else str(cl.value), parse_status=cl.parse_status,
            correct=cl.correct, rho=rho, u=u, reward=1.0 if u < rho else 0.0,
            verifier_config_id=self.config.config_id, verifier_config_sha256=self._sha,
            taxonomy_version=TAXONOMY_VERSION,
        )


def make_reward_fn(verifier: NoisyVerifier, log: list | None = None):
    """TRL-style reward function (TRL ≥1.x passes dataset columns as kwargs).

    Requires dataset columns `a`, `b`, `prompt_id`. `completions` may be strings or chat-message lists.
    Sample index = position within the prompt's group, recovered by counting repeats of prompt_id in the call.
    Every decision is appended to `log` (list of dicts) when given; true correctness is logged separately
    from the assigned reward.
    """

    def reward_fn(prompts=None, completions=None, a=None, b=None, prompt_id=None, trainer_state=None, **kwargs):
        step = int(getattr(trainer_state, "global_step", 0) or 0)
        counts: dict = {}
        rewards = []
        for comp, ai, bi, pid in zip(completions, a, b, prompt_id):
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            j = counts.get(pid, 0)
            counts[pid] = j + 1
            rec = verifier.decide(int(ai), int(bi), text, step, pid, j)
            rewards.append(rec.reward)
            if log is not None:
                log.append(asdict(rec))
        return rewards

    reward_fn.__name__ = f"veriflow_{verifier.config.config_id}"
    return reward_fn
