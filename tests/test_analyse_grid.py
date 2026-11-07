"""End-to-end test of the registered grid analysis on synthetic run directories (no real grid data)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "configs/grid/runs_grid.csv").exists(), reason="needs configs/grid")


def _fake_runs(tmp: Path, bV: float, drop: str | None = None):
    from experiments.analyse_grid import config_table
    table = pd.read_csv(ROOT / "configs/grid/runs_grid.csv")
    cfg = config_table(table).set_index("config_id")
    rng = np.random.default_rng(0)
    for r in table.itertuples():
        if r.run_name == drop:
            continue
        cid = Path(r.config).stem
        y = 1.6 + bV * cfg.loc[cid, "V"] + rng.normal(0, 0.1)
        acc = 1 / (1 + np.exp(-y))
        d = tmp / r.run_name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"status": "completed", "verifier_config": {"config_id": cid},
                                                     "args": {"seed": int(r.seed)}}))
        q0 = {"LEN": 0.16, "MULTI": 0.42, "ONE_HI": 0.18, "ONE_LO": 0.21, "I": 0.03}
        rows = [{"step": s, "greedy_accuracy": 0.408 if s == 0 else float(acc), "n_eval": 1000, "q_t": q0}
                for s in range(0, 101, 10)]
        (d / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")


def test_analysis_recovers_negative_v_effect(tmp_path):
    from experiments.analyse_grid import analyse
    _fake_runs(tmp_path, bV=-5.0)
    res, runs = analyse(tmp_path, ROOT / "configs/grid", 100, n_boot=500)
    assert res["n_valid"] == 96
    h1 = res["primary"]["H1"]
    assert h1["ci95"][1] < 0 and h1["ci95"][0] < h1["coef_V"] < h1["ci95"][1]
    assert set(res["primary"]["holm"]) == {"H1", "H2", "H3"}
    assert res["primary"]["n_heldout"] == 13


def test_missing_run_is_reported_not_dropped_silently(tmp_path):
    from experiments.analyse_grid import analyse
    table = pd.read_csv(ROOT / "configs/grid/runs_grid.csv")
    gone = table.run_name.iloc[0]
    _fake_runs(tmp_path, bV=0.0, drop=gone)
    res, runs = analyse(tmp_path, ROOT / "configs/grid", 100, n_boot=200)
    assert res["status_counts"].get("missing") == 1 and res["n_valid"] == 95
    assert Path(table.config.iloc[0]).stem in res["configs_with_one_valid_seed"]
