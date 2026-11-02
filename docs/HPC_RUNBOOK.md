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

## 1. Create the isolated project tree on the cluster

```bash
ssh <cluster-user>@<login-host-ip>
mkdir -p ~/VERIFLOW && cd ~/VERIFLOW   # never outside ~/VERIFLOW
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv is not installed yet
export PATH="$HOME/.local/bin:$PATH"                    # adjust to installer output
```

## 2. Sync the local repo to `~/VERIFLOW` (laptop → cluster)

From `E:\Veriflow` (this repo, after `git add`/commit of the Phase 4 files):

```bash
git -C . bundle create /tmp/veriflow.bundle --all
scp /tmp/veriflow.bundle <cluster-user>@<login-host-ip>:~/veriflow.bundle
ssh <cluster-user>@<login-host-ip> 'git clone ~/veriflow.bundle ~/VERIFLOW && cd ~/VERIFLOW && git checkout master'
```

(Alternative: `rsync -a --exclude .venv --exclude .cache --exclude runs . remote:~/VERIFLOW/`.)

## 3. Environment build + CPU preflight (job 1 — no GPU)

```bash
cd ~/VERIFLOW
sbatch slurm/preflight.sbatch          # ~30 min incl. the CPU test suite
squeue -u <cluster-user>                    # wait; then check logs/slurm-free output
less veriflow-preflight_<id>.out       # expect preflight ok:true, tests_exit:0
```

## 4. Model download + GPU check (job 2 — first GPU use)

```bash
# on the login node (internet): download ONLY the pinned revisions
export VERIFLOW_ROOT="$HOME/VERIFLOW"
export HF_HOME="$VERIFLOW_ROOT/.cache/hf"
uv run --directory ~/VERIFLOW huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir "$HF_HOME"
uv run --directory ~/VERIFLOW huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --cache-dir "$HF_HOME"   # replication model, optional now

cd ~/VERIFLOW
sbatch slurm/gpucheck.sbatch           # 15 min: driver, torch, bf16, pinned-revision agreement
less veriflow-gpucheck_<id>.out        # expect gpus:2 rows…, ok:true, models ok:true
```

If the driver is too old for the CUDA wheels torch resolved (gpucheck reports
`compatible: false`), STOP and record it in DECISIONS.md before changing
versions — the fallback is a CPU-compatible torch build decided in one place.

## 5. Census (job 4), one GPU, n = 5..10 sequentially, %1

```bash
sbatch --export=CENSUS_N=7 slurm/census.sbatch
# then 5, 6, 8, 9, 10. One at a time (--array=0-%1 style discipline):
#   for N in 5 6 7 8 9 10; do sbatch --export=CENSUS_N=$N slurm/census.sbatch; done
```

Each run writes `runs/census/n<N>_s20261102/` with every §11 artifact plus
`params.json`/`manifest.json`. Runtime target: ≈0.2–0.5 GPU-h per n (measured).
A nonzero exit means the census itself surfaced failing conditions (also check
`coverage_ambiguity.json`).

## 6. Choose n and pass Gate 3

Compare per-n reports: base accuracy (want 0.25–0.60), coverage (want ≥ 70 %),
TV. For the best n, run the gate checker and the §9.3 sample export:

```bash
uv run python -m experiments.check_construct_gate \
    --census runs/census/n<best>_s20261102 \
    --json runs/census/n<best>_s20261102/gate3.json \
    --audit-sample 200 --freeze-qref
```

- automated criteria must all pass (checker exit 0);
- complete `audit_sample.csv` by hand (§9.3.6, ≤ 5 % disagreement);
- only then does `configs/locked/q_ref.lock` exist. Mock runs can never freeze
  (the checker refuses them).

Also measure and record in DECISIONS.md: GPU-h per census run, disk per run.

## 7. Go/No-Go escalation paths (§9.3 ladder)

If the best automated gate fails: try another n → answer-only prompt →
subtraction → accept M=3 modes → **stop** (matched-heterogeneity design fails
on this model; that is the kill rule, not an excuse to continue).

## After the gate passes

Record the D7/D8 entries in `docs/DECISIONS.md` (census artifacts, chosen n,
measured cost), then start Phase 5 (Gate 4: `train_grpo.py`, `veriflow-smoke`,
`veriflow-bench`).
