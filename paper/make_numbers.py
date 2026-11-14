#!/usr/bin/env python
"""Generate paper/generated/numbers.tex: every number the manuscript reports, read from result files.

Sources: configs/pilot/{selection,gate5,power}.json, results/pilot/validate_het.json, docs/gate3_tv_diagnosis.json,
results/grid/analysis.json (if present), results/grid/validate_grid.json (if present).
No number in paper/main.tex is typed by hand; the text uses the macros defined here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated" / "numbers.tex"


def load(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}".replace("-", "\\ensuremath{-}") if isinstance(x, float) else str(x)


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> int:
    m: dict[str, str] = {}
    sel = load("configs/pilot/selection.json")
    m["LR"] = "$5\\times10^{-7}$" if abs(sel["lr"] - 5e-7) < 1e-12 else str(sel["lr"])
    m["HorizonT"] = str(sel["T"])
    for k, v in sel["window_scores"].items():
        m["LRscore" + {"5e-07": "A", "1e-06": "B", "3e-06": "C"}[k]] = fmt(v, 3)
    g5 = load("configs/pilot/gate5.json")
    c = g5["configs"]
    m["GfiveD"] = fmt(c["pilot_het_vhigh"]["D"], 2)
    m["GfiveDse"] = fmt(c["pilot_het_vhigh"]["SE_D"], 2)
    m["GfiveF"] = fmt(c["pilot_het_vhigh"]["dF"], 2)
    m["GfiveFse"] = fmt(c["pilot_het_vhigh"]["SE_dF"], 2)
    m["GfiveY"] = fmt(c["pilot_het_vhigh"]["dY"], 2)
    m["GfiveYse"] = fmt(c["pilot_het_vhigh"]["SE_dY"], 2)
    m["GfiveVmidF"] = fmt(c["pilot_het_vmid"]["dF"], 2)
    m["PilotYvhighMin"] = fmt(min(c["pilot_het_vhigh"]["Y_seeds"]), 2)
    m["PilotYvhighMax"] = fmt(max(c["pilot_het_vhigh"]["Y_seeds"]), 2)
    pw = load("configs/pilot/power.json")
    m["SigmaS"] = fmt(pw["sigma"]["sigma_s"], 3)
    m["SigmaRraw"] = fmt(pw["sigma"]["sigma_r_raw"], 3)
    m["RhoMed"] = fmt(pw["rho_median"], 2)
    m["RhoPess"] = fmt(pw["rho_pessimistic"], 2)
    m["PlanBV"] = fmt(pw["powers"]["planning"]["bV"], 2)
    for R in (12, 24, 36):
        m[f"Pow{ {12: 'A', 24: 'B', 36: 'C'}[R] }"] = fmt(pw["powers"]["planning"][str(R)], 2)
        m[f"PowPess{ {12: 'A', 24: 'B', 36: 'C'}[R] }"] = fmt(pw["powers"]["pessimistic"][str(R)], 2)
    m["GainPilot"] = fmt(pw["gain_pilot"], 2)
    m["VRange"] = fmt(pw["V_range_real"], 3)
    vh = load("results/pilot/validate_het.json")
    m["PilotValid"] = f"{vh['n_ok']}/{vh['n_runs']}"
    tv = load("docs/gate3_tv_diagnosis.json")
    sel_run = [r for r in tv["runs"] if r["run_id"].endswith("j3640")][0]
    m["TVdeclared"] = fmt(sel_run["tv_declared"], 4)
    m["TVpfail"] = fmt(sel_run["tv_random_prompt_halves"]["p_gt_0.03"], 2)
    m["TVreplicate"] = fmt(tv["independent_replication"]["tv"], 3)
    qu = tv["q_ref_uncertainty"]
    m["FbarDev"] = fmt(qu["max_abs_Fbar_deviation_p95_over_draws"], 3)
    m["VrankMin"] = fmt(qu["matched_V_spearman_vs_lock"]["min"], 3)

    a = load("results/grid/analysis.json")
    if a:
        p = a["primary"]
        h1 = p["H1"]
        m["GridValid"] = str(a["n_valid"])
        m["GridRuns"] = str(a["n_runs"])
        m["HoneBV"] = fmt(h1["coef_V"], 2)
        m["HoneLo"] = fmt(h1["ci95"][0], 2)
        m["HoneHi"] = fmt(h1["ci95"][1], 2)
        m["HoneSE"] = fmt(h1["se_hc3"], 2)
        m["HoneP"] = pval(h1["p"])
        for k in ("H2", "H3"):
            m[k.replace("2", "two").replace("3", "three").replace("H", "H") + "Diff"] = fmt(p[k]["mean_diff"], 4)
            m[k.replace("2", "two").replace("3", "three") + "Lo"] = fmt(p[k]["ci95"][0], 4)
            m[k.replace("2", "two").replace("3", "three") + "Hi"] = fmt(p[k]["ci95"][1], 4)
            m[k.replace("2", "two").replace("3", "three") + "P"] = pval(p[k]["p"])
        for k, v in p["holm"].items():
            m["Holm" + k.replace("1", "one").replace("2", "two").replace("3", "three")] = pval(v)
        m["HthreeS"] = fmt(p["H3"]["s"], 2)
        m["LocoFV"] = fmt(p["loco_all_matched"]["fmax+V"], 4)
        m["LocoF"] = fmt(p["loco_all_matched"]["fmax"], 4)
        m["ICCmatched"] = fmt(a["icc_matched"]["icc"], 2)
        m["SpearmanYV"] = fmt(a["spearman_Y_V_matched"], 2)
        m["NheldOut"] = str(p["n_heldout"])
        m["Ntrain"] = str(p["n_train"])
        m["HtwoMSEfv"] = fmt(p["H2"]["mse_fmax_V"], 4)
        m["HtwoMSEf"] = fmt(p["H2"]["mse_fmax"], 4)
        m["HthreeMSEz"] = fmt(p["H3"]["mse_cgf"], 4)
        m["ICCall"] = fmt(a["icc_all"]["icc"], 2)
        rb = a["robustness_exploratory"]
        m["RobNoHomBV"], m["RobNoHomLo"], m["RobNoHomHi"] = (fmt(rb["H1_without_homogeneous_control"]["coef_V"], 2),
                                                           *(fmt(x, 2) for x in rb["H1_without_homogeneous_control"]["ci95"]))
        m["RobFinalBV"], m["RobFinalLo"], m["RobFinalHi"] = (fmt(rb["H1_on_final_accuracy"]["coef_V"], 2),
                                                           *(fmt(x, 2) for x in rb["H1_on_final_accuracy"]["ci95"]))
        so = a["secondary_outcomes_by_arm"]
        m["AccTmatched"] = fmt(so["matched"]["acc_T"], 3)
        m["AccTaux"] = fmt(so["auxiliary"]["acc_T"], 3)
        m["HackMatched"] = fmt(so["matched"]["hacking"], 3)
        m["Divergences"] = str(sum(a["divergences_by_arm"].values()))
        pr = {r["predictor"]: r for r in a["secondary_predictors_all_configs_exploratory"]}
        for k, name in (("V", "V"), ("fmax", "Fmax"), ("trigger", "Trig"), ("V_q2", "Vqtwo"), ("kappa3", "Kappa"),
                        ("fmax+V", "FmaxV"), ("max_qf", "Maxqf"), ("CGF", "CGF"), ("null(intercept)", "Null")):
            m["Rel" + name] = fmt(pr[k]["rel_mse"], 2)

    def tabget(rows, term):
        return [r for r in rows if r["term"] == term][0]

    def coef3(prefix, rows, term, d=2):
        r = tabget(rows, term)
        m[prefix], m[prefix + "Lo"], m[prefix + "Hi"] = fmt(r["coef"], d), fmt(r["ci95"][0], d), fmt(r["ci95"][1], d)

    s = load("results/grid/supplement.json")
    if s:
        coef3("HoneBF", s["A_H1_full_table_registered"], "fmax")
        m["HoneRsq"] = fmt(s["A_H1_r2"], 2)
        C = s["C_realised_rates_registered_commitment"]
        m["FrealMatchedMean"] = fmt(C["matched"]["F_real_window_mean"], 2)
        m["FrealMatchedMin"] = fmt(C["matched"]["F_real_window_min"], 2)
        m["FrealMatchedMax"] = fmt(C["matched"]["F_real_window_max"], 2)
        m["FrealAllMatched"] = fmt(C["matched"]["F_real_all_mean"], 3)
        m["TaurealMin"] = fmt(min(C["matched"]["tau_real_all_min"], C["auxiliary"]["tau_real_all_min"]), 3)
        m["TaurealMax"] = fmt(max(C["matched"]["tau_real_all_max"], C["auxiliary"]["tau_real_all_max"]), 3)
        m["SpearFrealV"] = fmt(C["matched_corr_F_real_window_vs_V"], 2)
        B = s["B_secondary_model_registered"]
        m["SecN"] = str(B["n_configs"])
        m["SecRel"] = fmt(B["loco_mse"] / B["var_Y"], 2)
        coef3("SecV", B["table"], "V")
        coef3("SecFPR", B["table"], "FPR")
        coef3("SecMaxqf", B["table"], "max_qf")
        m["SecVifV"] = fmt(B["vif"]["V"], 0)
        D = s["D_aggregate_vs_heterogeneity_exploratory"]
        for k, name in (("FPR", "AggF"), ("FPR+fmax", "AggFM"), ("FPR+V", "AggFV"), ("FPR+fmax+V", "AggFMV"),
                        ("FPRreal", "AggR"), ("FPRreal+fmax+V", "AggRMV")):
            m["Rel" + name] = fmt(D[k]["rel_mse"], 2)
        coef3("AggFMVcoefV", D["FPR+fmax+V"]["table"], "V")
        coef3("AggFMVcoefF", D["FPR+fmax+V"]["table"], "FPR")
        coef3("AggRMVcoefV", D["FPRreal+fmax+V"]["table"], "V")
        coef3("AggRMVcoefR", D["FPRreal+fmax+V"]["table"], "F_real_window")
        E = s["E_robustness_exploratory"]
        m["LooBVmin"], m["LooBVmax"] = fmt(E["loo_config_bV_min"], 2), fmt(E["loo_config_bV_max"], 2)
        rc = E["run_level_clustered"]
        m["ClustBV"], m["ClustLo"], m["ClustHi"] = fmt(rc["coef_V"], 2), fmt(rc["ci95"][0], 2), fmt(rc["ci95"][1], 2)
        m["PartialSpear"] = fmt(E["spearman_partial_Y_V_given_fmax"], 2)
        coef3("NoLowBV", E["H1_without_runs_Y_below_0.3_sensitivity_only"], "V")
        coef3("MedBV", E["H1_with_realised_FPR_covariate_mediator_not_confounder"], "V")
        coef3("MedBR", E["H1_with_realised_FPR_covariate_mediator_not_confounder"], "F_real_window")
        F = s["F_unusual_runs_exploratory"]
        m["SeedGapMed"] = fmt(F["median_seed_gap"], 2)
        m["SeedGapRho"] = fmt(F["spearman_seed_gap_vs_V_matched"]["rho"], 2)
        m["NhackRuns"] = str(len(F["hacking_flagged_runs"]))
        lowc = F["runs_Y_below_0.3"]
        m["NlowHighV"], m["NlowLowV"], m["NlowAux"] = str(lowc["matched_highV"]), str(lowc["matched_lowV"]), str(lowc["auxiliary"])
        m["NlowDistinct"] = str(len({tuple(t["acc"]) for t in F["low_Y_trajectories"].values()}))
        m["LowAccT"] = fmt(min(t["acc"][-1] for t in F["low_Y_trajectories"].values()), 2) + "--" + fmt(
            max(t["acc"][-1] for t in F["low_Y_trajectories"].values()), 2)
        m["LowLenT"] = fmt(min(t["q_top"][-1] for t in F["low_Y_trajectories"].values()), 2) + "--" + fmt(
            max(t["q_top"][-1] for t in F["low_Y_trajectories"].values()), 2)
        H = s["H_duplicate_configs_disclosure"]
        m["NuniqueMatched"] = str(H["n_unique_matched"])
        m["NuniqueAll"] = str(H["n_unique_verifiers_all"])
        m["NheldTwin"] = str(len(H["heldout_with_identical_training_twin"]))
        m["NdupGroups"] = str(len(H["duplicate_groups"]))
        coef3("UniqBV", H["H1_unique_matched_exploratory"], "V")
        coef3("UniqBF", H["H1_unique_matched_exploratory"], "fmax")
        lo = H["loo_unique_matched_exploratory"]
        m["UniqLooFV"], m["UniqLooF"], m["UniqLooZ"] = fmt(lo["mse_fmax_V"], 4), fmt(lo["mse_fmax"], 4), fmt(lo["mse_cgf"], 4)
        d = lo["fmaxV_minus_fmax"]
        m["UniqDiff"], m["UniqDiffLo"], m["UniqDiffHi"], m["UniqDiffP"] = (fmt(d["mean_diff"], 4), fmt(d["ci95"][0], 4),
                                                                          fmt(d["ci95"][1], 4), pval(d["p"]))
        d = lo["cgf_minus_fmaxV"]
        m["UniqZDiff"], m["UniqZDiffLo"], m["UniqZDiffHi"] = fmt(d["mean_diff"], 4), fmt(d["ci95"][0], 4), fmt(d["ci95"][1], 4)
        ph = H["heldout_without_twins_exploratory"]
        m["PureN"] = str(ph["n_heldout"])
        m["PureDiff"], m["PureLo"], m["PureHi"] = (fmt(ph["H2"]["mean_diff"], 4), fmt(ph["H2"]["ci95"][0], 4),
                                                   fmt(ph["H2"]["ci95"][1], 4))
        m["PureZDiff"], m["PureZLo"], m["PureZHi"] = (fmt(ph["H3"]["mean_diff"], 4), fmt(ph["H3"]["ci95"][0], 4),
                                                      fmt(ph["H3"]["ci95"][1], 4))
        ag = H["aggregate_vs_V_unique_all_exploratory"]
        m["UniqRelFM"], m["UniqRelFMV"] = fmt(ag["FPR+fmax"]["rel_mse"], 2), fmt(ag["FPR+fmax+V"]["rel_mse"], 2)
        coef3("UniqAggV", ag["FPR+fmax+V"]["table"], "V")
        m["UniqSpearYV"] = fmt(H["spearman_Y_V_unique_matched"], 2)
        G = s["G_compute"]
        m["StepSecMed"], m["GPUhours"] = fmt(G["median_step_seconds"], 2), fmt(G["train_gpu_hours_sum"], 1)
    if a and s:
        h1, u = a["primary"]["H1"], s["H_duplicate_configs_disclosure"]["H1_unique_matched_exploratory"]
        uv = tabget(u, "V")
        (ROOT / "paper" / "generated" / "abstract_result.tex").write_text(
            "% Generated by paper/make_numbers.py; do not edit by hand.\n"
            f"All {a['n_valid']} of {a['n_runs']} registered runs completed and passed validation. "
            f"Across the matched configurations, higher dispersion was associated with lower accuracy: "
            f"$b_V={fmt(h1['coef_V'], 2)}$ logit per unit $V$ (95\\% CI {fmt(h1['ci95'][0], 2)} to {fmt(h1['ci95'][1], 2)}), "
            f"and the estimate is stable when the {s['H_duplicate_configs_disclosure']['n_configs_matched'] - s['H_duplicate_configs_disclosure']['n_unique_matched']} "
            f"duplicated verifier configurations that the registered grid contains are removed "
            f"($b_V={fmt(uv['coef'], 2)}$, 95\\% CI {fmt(uv['ci95'][0], 2)} to {fmt(uv['ci95'][1], 2)}). "
            f"Adding $V$ to $\\fmax$ improved prediction of held-out configurations in the registered split "
            f"(paired MSE difference {fmt(a['primary']['H2']['mean_diff'], 3)}, 95\\% CI {fmt(a['primary']['H2']['ci95'][0], 3)} to "
            f"{fmt(a['primary']['H2']['ci95'][1], 3)}), but not on the {s['H_duplicate_configs_disclosure']['heldout_without_twins_exploratory']['n_heldout']} "
            f"held-out verifiers that have no identical twin in training, and the CGF plug-in did not improve on $\\{{\\fmax,V\\}}$. "
            f"In exploratory analyses, the association is accompanied by a rise in the realised, on-policy false-acceptance rate "
            f"(which the matched design-time rate does not capture) as the policy drifts toward the verifier's most-accepted error mode, "
            f"and the lowest outcomes are single seeds that reward-hack one nearly always-accepted mode. "
            f"With two seeds per configuration and one model and task, these are estimates for a narrow setting, not general claims.\n",
            encoding="utf-8")
    lines = ["% Generated by paper/make_numbers.py from result files; do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(m)} macros to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
