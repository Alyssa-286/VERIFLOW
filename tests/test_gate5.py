import json
from pathlib import Path

import pandas as pd

from experiments import gate5

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["pilot_het_v0", "pilot_het_vmid", "pilot_het_vhigh", "pilot_het_fmaxhigh_vlow"]


def _fake_run(d: Path, T: int, q0: dict, qT: dict, acc: float, fa: float):
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"status": "completed", "final_step": T}))
    rows = [{"step": s, "greedy_accuracy": acc, "q_t": q0 if s == 0 else qT} for s in range(0, T + 1, 10)]
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "verifier_log").mkdir()
    n = 100
    pd.DataFrame({"step": [T] * n, "correct": [False] * n,
                  "reward": [1.0] * int(fa * n) + [0.0] * (n - int(fa * n))}).to_parquet(
        d / "verifier_log" / "upto.parquet", index=False)


def _table(tmp: Path, amplify: bool, T: int = 100):
    rows = []
    q0 = {"LEN": 0.16, "MULTI": 0.42, "ONE_HI": 0.18, "ONE_LO": 0.21}
    for cid in CONFIGS:
        m = gate5.argmax_mode(json.loads((ROOT / f"configs/pilot/het/{cid}.json").read_text()))
        for s in (1, 2, 3):
            qT = dict(q0)
            if amplify and cid != "pilot_het_v0":
                qT[m] = q0[m] * (2.0 + 0.05 * s)
            else:
                qT[m] = q0[m] * (1.0 + 0.01 * s)
            fa = (0.45 + 0.01 * s) if (amplify and cid == "pilot_het_vhigh") else (0.30 + 0.01 * s)
            name = f"het_{cid}_s{s}"
            _fake_run(tmp / name, T, q0, qT, 0.85, fa)
            rows.append({"run_name": name, "config": f"configs/pilot/het/{cid}.json", "lr": 5e-7,
                         "seed": s, "max_steps": T})
    return pd.DataFrame(rows)


def test_gate5_passes_when_modes_amplified(tmp_path):
    res = gate5.evaluate(_table(tmp_path, amplify=True), tmp_path, 100)
    assert res["gate5_pass"]
    assert "pilot_het_vhigh" in res["engaged_D"] and res["high_V_engaged_F"]


def test_gate5_fails_when_flat(tmp_path):
    res = gate5.evaluate(_table(tmp_path, amplify=False), tmp_path, 100)
    assert not res["gate5_pass"]
    assert res["engaged_D"] == [] and not res["high_V_engaged_F"]


def test_argmax_excludes_U_and_I():
    cfg = {"f": {"U": "0.99", "I": "0.98", "LEN": "0.2", "ONE_HI": "0.9"}}
    assert gate5.argmax_mode(cfg) == "ONE_HI"
