"""Noisy verifier tests (§10.4)."""

import random

import numpy as np
import pytest

from verifiers import matched_verifiers as mv
from verifiers.noisy_verifier import NoisyVerifier, make_reward_fn, prf_uniform

Q_REF = {"ECHO": 0.06, "CARRY": 0.34, "TRUNC": 0.12, "TRANS": 0.08, "DIGIT": 0.25, "U": 0.10, "I": 0.05}


def _cfg():
    rng = np.random.default_rng(0)
    return mv.generate_config(Q_REF, ["ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT"], 0.3, 0.05, 0.85, rng, config_id="t1")


# completions of each class for a = 58, b = 67 (s = 125)
EXAMPLES = {
    "C": "Answer: 125", "CARRY": "Answer: 25", "ECHO": "Answer: 58", "TRUNC": "Answer: 12",
    "TRANS": "Answer: 152", "DIGIT": "Answer: 129", "U": "Answer: 999", "I": "I think it is big",
}


def test_example_classes():
    v = NoisyVerifier(_cfg(), 1, 1)
    for cls, text in EXAMPLES.items():
        assert v.decide(58, 67, text, 0, "p", 0).mode == cls


@pytest.mark.parametrize("cls", list(EXAMPLES))
def test_realised_rates_match_configured(cls):
    cfg = _cfg()
    v = NoisyVerifier(cfg, verifier_seed=7, run_seed=3)
    n = 40_000
    acc = sum(v.decide(58, 67, EXAMPLES[cls], step, "p0", 0).reward for step in range(n)) / n
    rho = cfg.tau if cls == "C" else cfg.f[cls]
    se = max(np.sqrt(rho * (1 - rho) / n), 1e-12)
    assert abs(acc - rho) <= 4.5 * se + 1e-12, (cls, acc, rho)


def test_invalid_never_rewarded_and_record_fields():
    v = NoisyVerifier(_cfg(), 1, 1)
    for s in range(2000):
        r = v.decide(58, 67, "nonsense", s, "p", 0)
        assert r.reward == 0.0 and not r.correct and r.mode == "I"
    r = v.decide(58, 67, "Answer: 125", 5, "p", 2)
    assert r.correct and r.taxonomy_version == "v1" and len(r.verifier_config_sha256) == 64


def test_replay_determinism_and_independence_from_global_rngs():
    v = NoisyVerifier(_cfg(), 11, 22)
    np.random.seed(0)
    random.seed(0)
    state_np = np.random.get_state()[1].copy()
    first = [v.decide(58, 67, "Answer: 25", s, "p", j).u for s in range(50) for j in range(4)]
    assert (np.random.get_state()[1] == state_np).all()  # global numpy RNG untouched
    np.random.seed(123)
    random.seed(123)
    second = [NoisyVerifier(_cfg(), 11, 22).decide(58, 67, "Answer: 25", s, "p", j).u for s in range(50) for j in range(4)]
    assert first == second


def test_stream_keys_matter_and_draws_are_uncorrelated():
    u = np.array([prf_uniform("veriflow-v1", 1, 1, s, "p", j) for s in range(5000) for j in range(2)]).reshape(5000, 2)
    r = np.corrcoef(u[:, 0], u[:, 1])[0, 1]
    assert abs(r) < 3 / np.sqrt(5000) + 0.01
    assert abs(u.mean() - 0.5) < 0.01
    assert prf_uniform("veriflow-v1", 1, 1, 0, "p", 0) != prf_uniform("veriflow-v1", 2, 1, 0, "p", 0)
    assert prf_uniform("veriflow-v1", 1, 1, 0, "p", 0) != prf_uniform("veriflow-v1", 1, 2, 0, "p", 0)


def test_per_string_variant_is_deterministic_per_completion():
    v = NoisyVerifier(_cfg(), 1, 1, per_string=True)
    us = {v.decide(58, 67, "Answer: 25", s, "p", j).u for s in range(20) for j in range(3)}
    assert len(us) == 1


def test_reward_fn_trl_signature_and_logging():
    class State:
        global_step = 4

    log = []
    fn = make_reward_fn(NoisyVerifier(_cfg(), 1, 1), log)
    comps = ["Answer: 125", [{"role": "assistant", "content": "Answer: 25"}], "Answer: 125", "x"]
    out = fn(prompts=["q"] * 4, completions=comps, a=[58, 58, 1, 1], b=[67, 67, 2, 2],
             prompt_id=["p1", "p1", "p2", "p2"], trainer_state=State(), extra="ignored")
    assert len(out) == 4 and set(out) <= {0.0, 1.0}
    assert [r["sample_idx"] for r in log] == [0, 1, 0, 1]
    assert [r["step"] for r in log] == [4] * 4
    assert [r["mode"] for r in log] == ["C", "CARRY", "U", "I"]
    assert all("correct" in r and "reward" in r for r in log)
