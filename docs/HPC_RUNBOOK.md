# VERIFLOW Phase 4 runbook (SIT HPC, exact commands)

Ordered from login to the frozen `q_ref.lock`. Every command is read-only-safe
or freshly-creating inside `~/VERIFLOW`; nothing is run against ErrorEcho.
Companion documents: [CLUSTER_POLICY.md](CLUSTER_POLICY.md) (verified scheduler
facts, 2026-11-02), [GATE3.md](GATE3.md) (gate criteria).

## 0. From your workstation — don't repeat the read-only recon

The §16.2 policy commands were run on 2026-11-02 and are recorded in
`docs/CLUSTER_POLICY.md`. Re-run only if you suspect the node changed:

```bash
ssh <cluster-user>@<login-host-ip> 'squeue -u <cluster-user>'   # ErrorEcho priority check
```

## 1–2. Deploy (done 2026-11-03; repeat only to update)

The cluster cannot read the private GitHub repo, so code goes over a bundle.
`git pull --ff-only` refuses anything but a fast-forward, which keeps the
cluster checkout from diverging:

```bash
# workstation, from E:\Veriflow
git bundle create veriflow.bundle phase4/hpc-ready
scp veriflow.bundle <cluster-user>@<login-host-ip>:veriflow-sync.bundle
ssh <cluster-user>@<login-host-ip> 'cd ~/VERIFLOW && git pull --ff-only ~/veriflow-sync.bundle phase4/hpc-ready'
```

First-time only: `git clone -b phase4/hpc-ready ~/veriflow-<sha>.bundle ~/VERIFLOW`
(the target must not exist), then
`curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh`.
`slurm/_common.sh` puts `~/.local/bin` on PATH for jobs.

## 3. Environment + CPU preflight (job 1, no GPU)

```bash
cd ~/VERIFLOW && export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$HOME/VERIFLOW/.cache/uv
uv sync --frozen --group gpu            # always name the gpu group
sbatch slurm/preflight.sbatch           # ~20 s; logs/slurm/veriflow-preflight_<id>.out
```

## 4. Model download + GPU check (job 2)

```bash
export HF_HOME=$HOME/VERIFLOW/.cache/hf
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
sbatch slurm/gpucheck.sbatch            # report: logs/gpucheck_<id>.json; ok:true required
```

Verify the weights: the safetensors blob name under
`.cache/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/` must equal both its
`sha256sum` and the HF API's LFS sha256 for the pinned revision (done:
`fdf756fa…fb7fe`). `nvml_ok: false` is expected until the node is rebooted
(CLUSTER_POLICY.md).

## 5. Census (job 4), one job at a time

```bash
for N in 5 6 7 8 9 10; do
  J=$(sbatch --parsable --job-name=veriflow-census-n$N --export=ALL,CENSUS_N=$N slurm/census.sbatch)
  while squeue -h -j $J | grep -q .; do sleep 10; done
  scontrol show job $J | egrep -o "JobState=[A-Z]+|ExitCode=[0-9:]+|RunTime=[0-9:]+"
done
```

Each job writes `runs/census/n<N>_<tag>_s20261102_j<jobid>/` (tag `p2t2` = addition, prompt v2 + taxonomy v2; `sub_p2t2` = subtraction via `--export=ALL,CENSUS_N=<n>,CENSUS_TASK=sub`) (never reused):
generations_raw/generations/classification parquet, q_ref.json, stability,
coverage_ambiguity, per-prompt and per-mode summaries, params.json and
manifest.json. Exit 1 means the census completed but the §9.3 gate failed for
that n. Any other nonzero exit, or an empty run directory, is a crash.
Measured: about 72 s and about 2.1 GB of VRAM per n on an L40.

## 6. Verify, compare, choose n

```bash
uv run python -m experiments.compare_census runs/census/<run_id> ... \
    --json runs/census/gate3_comparison.json --md runs/census/gate3_comparison.md
```

This re-derives every statistic from classification.parquet, checks the
manifest (pinned revision, template hash, q_ref hash) and applies the §9.1
selection rule. For the selected run only, export and judge the blinded audit,
then freeze (GATE3.md).

## 7. Go/No-Go escalation paths (§9.3 ladder)

If the best automated gate fails: try another n → answer-only prompt →
subtraction → accept M=3 modes → **stop** (matched-heterogeneity design fails
on this model; that is the kill rule, not an excuse to continue).

## After the gate passes

Record the D7/D8 entries in `docs/DECISIONS.md` (census artifacts, chosen n,
measured cost), then start Phase 5 (Gate 4: `train_grpo.py`, `veriflow-smoke`,
`veriflow-bench`).

## 8. Stage 3: pilot (after q_ref.lock exists)

```bash
cd ~/VERIFLOW
# 0. the researcher's audit freezes the lock (GATE3.md); then
uv run python -m experiments.make_pilot_configs                 # configs/pilot/{lr,het}, runs_lr.csv
sbatch --array=0-3%1 --export=ALL,RUN_TABLE=configs/pilot/runs_lr.csv,RUN_KIND=pilot,RUNS_ROOT=runs/pilot \
    slurm/grpo_array.sbatch                                      # 4 LR-pilot runs, one at a time (~1 GPU-h)
uv run python -m experiments.pilot_select                        # D19 rules -> selection.json, runs_het.csv
sbatch --array=0-11%1 --export=ALL,RUN_TABLE=configs/pilot/runs_het.csv,RUN_KIND=pilot,RUNS_ROOT=runs/pilot \
    slurm/grpo_array.sbatch                                      # 12 heterogeneity runs
```

Commit `configs/pilot/` (configs, run tables, selection) before the heterogeneity array.
Plumbing dry run: array job 3673 (2 tasks, provisional config, run-kind bench).
