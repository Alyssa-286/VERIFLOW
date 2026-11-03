# SIT HPC cluster policy (VERIFLOW)

Recorded from the **read-only** §16.2 commands on the login/master node
(`<login-host>`, user `<cluster-user>`) on **2026-11-02**. ErrorEcho always has
priority; nothing was submitted, modified, or cancelled for this survey.

## Verified facts (2026-11-02)

| Item | Value | Source |
|---|---|---|
| Login/master host | `<login-host>` (<login-host-ip>) | ssh + hostname |
| Partition | `gpu-nodes` (default), TIMELIMIT `infinite`, 1 node, state `up`/`mix` | `sinfo -p gpu-nodes` |
| Node | `sit-gpu` — **MIXED**, 96 CPUs, 2 sockets, 2 threads/core | `scontrol show node sit-gpu` |
| GPUs on node | **2** (`Gres=gpu:2`) | `scontrol show node sit-gpu` |
| Memory | 515,705 MB total; 200 GB allocated at survey time | same |
| MaxWall | INFINITE (partition `MaxTime=UNLIMITED`) | `scontrol show partition` |
| Preemption | `PreemptMode=OFF`; OverSubscribe=NO | same |
| Accounting/QoS | **not available** (no slurmdbd plugin: `sacctmgr` refused). Partition shows no per-user QoS limits | `sacctmgr show qos` |
| Concurrent jobs observed | job 3580 `p12-exploratory` (user's own, gpu:1) running on sit-gpu → **1 of 2 GPUs busy** | `squeue` |
| GRES string | `gpu:1` (generic; no per-model GRES like `gpu:l40:1` observed) | `squeue`/`scontrol` |
| OS | Ubuntu 24.04.3 LTS | `/etc/os-release` |
| System Python | 3.12.3 (`python3`) | login node |
| CUDA toolkit on master | 12.0 (`nvcc`), **driver version unknown from login node** (no NVIDIA driver there — normal) | `nvcc --version`; `nvidia-smi` fails on the login node by design |
| Modelling of GPU/driver | **Must be confirmed by `veriflow-gpucheck` inside a GPU allocation** (torch.cuda works? bf16? driver ≥ 550.54 for CUDA 12.4 wheels) | `scripts/gpucheck.py` |
| Home filesystem | NFS `<nfs-server-ip>:/home`, 3.5 TB total, **1.8 TB free** | `df -h ~` |
| Nodes have internet | login node yes (huggingface.co reachable). **Download models on the login node, train offline** | curl |
| `uv` | **not installed** → install with `curl -LsSf https://astral.sh/uv/install.sh \| sh` | — |
| `~/VERIFLOW` | **does not exist yet** (Phase 4 mkdir step) | `ls` |
| ErrorEcho | lives in `~/ErrorEcho*` directories (incl. `~/ErrorEcho-phase11-20`); job 3580 named `p12-explor*`. VERIFLOW never touches these | `squeue`, `ls` |

## Operational rules derived from this

1. All VERIFLOW work happens under `~/VERIFLOW` only. Job names start
   `veriflow-` so `squeue` filtering is unambiguous. Never use ErrorEcho paths,
   caches, or job names (`tests/test_isolation.py` enforces this in code).
2. Before every GPU submission: `squeue -u <cluster-user>`. If the user's own job is
   running with `gres/gpu:1`, **at most one GPU remains**. VERIFLOW's array
   jobs must run `%1` (one concurrent array task) while an ErrorEcho job holds
   the other GPU, unless the user explicitly says otherwise.
3. Resource placeholders resolved:
   - `--gres=gpu:1` ✔ (plain, not `gpu:l40:1`).
   - `--cpus-per-task=8` / `--mem=64G` ✔ (96 CPUs / 515 GB total on one node;
     requests far below partition limits; accounting gives no stricter cap).
   - `--time`: no wall-time ceiling from Slurm. Census jobs are estimated at
     0.2–0.5 GPU-h (§11, to be measured once); the sbatch sets 02:00:00 so a
     hung census still returns the node.
4. Model download happens on a node with internet (login is fine — HF is
   reachable), into `$VERIFLOW_ROOT/.cache/hf`, then all GPU jobs run with
   `HF_HUB_OFFLINE=1`.
5. Driver/CUDA compatibility is checked, not assumed: `gpucheck` parses
   `nvidia-smi` driver vs `torch.version.cuda` (CUDA-12 minor-compat rule:
   driver ≥ 525.60). If the driver is older, fall back to CPU-correct torch
   build selection before locking anything (never fight the stack, §16.5).

## Unresolved (gated on a GPU allocation; `slurm/gpucheck.sbatch`)

- driver version on the GPU node; torch booted with bf16; model revision
  agreement with the HF snapshot cache. These are outputs, not assumptions.

## Live re-check and first GPU jobs (2026-11-03)

| Item | Value | Source |
|---|---|---|
| Partition / node | `gpu-nodes` up, `sit-gpu` MIXED, 96 CPUs, 515,705 MB, `Gres=gpu:2` | `sinfo`, `scontrol show node` |
| ErrorEcho activity | user jobs `p12audit` (3589) and `p6audit` (3603), CPU-only (no GRES), running alongside; untouched | `squeue` |
| GPU (allocated) | NVIDIA L40, 45,709 MiB, compute capability ≥ 8.0, bf16 ok | gpucheck 3593 |
| Driver | kernel module **580.173.02**; userland libcuda/NVML **580.178.04** (upgraded 2026-08-09, node up 42 days, not rebooted) | `/proc/driver/nvidia/version`, `ls -l libcuda.so.1` |
| `nvidia-smi` on sit-gpu | **fails: "Driver/library version mismatch"** (NVML is strict). CUDA itself works | gpucheck 3591 |
| torch / CUDA runtime | 2.8.0+cu128 / 12.8; compatible with driver 580 | gpucheck 3593 |
| `uv` | 0.13.7 installed at `~/.local/bin/uv` (no shell-profile edit) | installer |
| GitHub from cluster | private repo is not reachable without credentials; deploy by `git bundle` | `git ls-remote` |
| HF from login node | reachable; pinned model downloaded to `~/VERIFLOW/.cache/hf` | `hf download` |

Admin action recommended (not ours to take): reboot `sit-gpu` or reload the
nvidia kernel module so NVML matches 580.178.04. Until then `gpucheck` uses a
CUDA-driver-API inventory (`nvml_ok: false` in its report). The bf16 kernels
and the real model generation are the proof that the GPU works.
