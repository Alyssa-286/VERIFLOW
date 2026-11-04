"""Phase 5 CPU-side pieces: Amendment 1 merges, the v3 verifier, config round-trip,
and the trainer's pinned settings (no torch needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from verifiers.matched_verifiers import design_grid, homogeneous_config, load_config, save_configs
from verifiers.noisy_verifier import NoisyVerifier
from verifiers.qref import load_q_ref, merge_small_modes

Q5 = {"I": 0.0301, "LEN": 0.1623, "MULTI": 0.4201, "NEG": 0.0, "ONE_HI": 0.181, "ONE_LO": 0.2065}


def test_merge_rule_amendment_1():
    q, merged = merge_small_modes(Q5)
    assert merged == {"NEG": "LEN"} and "NEG" not in q
    assert q["LEN"] == pytest.approx(0.1623) and sum(q.values()) == pytest.approx(1.0)
    q2, m2 = merge_small_modes({**Q5, "ONE_HI": 0.015, "MULTI": 0.5861})
    assert m2 == {"NEG": "LEN", "ONE_HI": "MULTI"} and q2["MULTI"] == pytest.approx(0.6011)


def test_load_q_ref_refuses_unfrozen_and_reads_provisional(tmp_path, monkeypatch):
    import verifiers.qref as qr
    monkeypatch.setattr(qr, "LOCK", tmp_path / "q_ref.lock")
    with pytest.raises(FileNotFoundError):
        load_q_ref()
    run = tmp_path / "census"
    run.mkdir()
    (run / "q_ref.json").write_text(json.dumps({"scheme": "add_v3", "q_ref": Q5}))
    d = load_q_ref(provisional=run)
    assert d["provisional"] and d["merged"] == {"NEG": "LEN"}


def _q():
    return merge_small_modes(Q5)[0]


def test_v3_verifier_rates_and_merge():
    cfg = homogeneous_config(_q(), Fbar=0.3, tau=0.9)
    assert cfg.Fbar == pytest.approx(0.3, abs=1e-12)
    v = NoisyVerifier(cfg, verifier_seed=5, run_seed=1, scheme="add_v3")
    a, b = 12345, 54321
    assert v.decide(a, b, "66666", 0, "p", 0).mode == "C"
    r = v.decide(a, b, "-5", 0, "p", 1)
    assert r.mode == "NEG" and r.rho == cfg.f["LEN"]  # merged class uses its target's rate
    assert v.decide(a, b, "nonsense", 0, "p", 2).reward == 0.0
    assert r.taxonomy_version == "v3"
    # realised rates over many draws match the configured rates
    acc = np.mean([v.decide(a, b, "66677", s, "p", j).reward for s in range(200) for j in range(50)])
    assert abs(acc - cfg.f["MULTI"]) < 4.5 * np.sqrt(cfg.f["MULTI"] * (1 - cfg.f["MULTI"]) / 10000)


def test_config_roundtrip_and_grid_on_v3_q_ref(tmp_path):
    cfgs = design_grid(_q(), Fbar=0.3, tau=0.9, V_fracs=[0.25, 0.75], fmax_levels=[None], n_per_cell=2, seed=3)
    assert len(cfgs) >= 3
    save_configs(cfgs, tmp_path)
    for c in cfgs:
        back = load_config(tmp_path / f"{c.config_id}.json")
        assert back.sha256 == c.sha256
        assert abs(back.Fbar - 0.3) < 1e-11 and abs(back.V - c.target_V) < 1e-11


def test_base_config_pins_theory_settings():
    base = json.loads((Path(__file__).resolve().parents[1] / "configs" / "base.json").read_text())
    g = base["grpo"]
    assert (g["loss_type"], g["beta"], g["scale_rewards"], g["num_iterations"]) == ("dapo", 0.0, "group", 1)
    assert (g["temperature"], g["top_p"], g["top_k"], g["repetition_penalty"]) == (1.0, 1.0, 0, 1.0)
    assert g["use_vllm"] is False and base["scheme"] == "add_v3" and base["split_seed"] == 20261104
    pps, per_dev, G = base["batch"]["prompts_per_step"], base["batch"]["per_device_train_batch_size"], g["num_generations"]
    assert (pps * G) % per_dev == 0 and per_dev % G == 0 and (pps * 16) % per_dev == 0 and per_dev % 16 == 0
