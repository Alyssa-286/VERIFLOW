#!/usr/bin/env python
"""veriflow-gpucheck (master plan §16.3 job 2). GPU job (90 s–15 min).

Reports, machine-readably:
  nvidia-smi GPU inventory + driver version;
  CUDA/PyTorch version compatibility (torch.version.cuda vs driver, minor-compat rule);
  torch.cuda availability, device count, bf16 matmul + generation smoke;
  pinned model/tokenizer revision agreement with configs/model/*.yaml;
  optional tiny HF generation (needs the model already downloaded).

Never allocates more than it needs; never raises an import error for torch
absent — it reports torch: absent and exits 0 unless --must-pass is set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NVIDIA_SMI_QUERY = (
    "index,name,memory.total,memory.used,utilization.gpu,driver_version"
)


def parse_nvidia_smi(text: str) -> list[dict]:
    rows = []
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return rows
    header = [h.split(" [")[0].strip() for h in lines[0].split(",")]  # drop [MiB]-style units
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(header):
            rows.append(dict(zip(header, vals)))
    return rows


def smi() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv"],
            capture_output=True, text=True, timeout=20,
        )
        return parse_nvidia_smi(out.stdout) if out.returncode == 0 else []
    except Exception:
        return []


def torch_report() -> dict:
    rep: dict = {"torch": None}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_runtime"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        rep["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        rep["bf16_supported"] = (torch.cuda.is_available()
                                 and torch.cuda.get_device_capability(0) >= (8, 0))
        rep["tests"] = {}
        if rep["cuda_available"]:
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize()
            rep["tests"]["bf16_matmul"] = bool(torch.isfinite(y.float()).all())
            rep["tests"]["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        rep["error"] = "torch not installed (CPU venv?)"
    except Exception as e:
        rep["error"] = repr(e)
    return rep


def driver_cuda_compatible(smi_rows: list[dict], torch_rep: dict) -> dict:
    """Minor-version compatibility: a CUDA 12.x runtime needs driver >= the
    CUDA 12.0 baseline (mot: ≥ 525.60.13); CUDA 11.x runtime needs >= 450.80.02."""
    def drv_major(v: str):
        return int(v.split(".")[0]) if v else None
    driver = smi_rows[0].get("driver_version") if smi_rows else None
    runtime = torch_rep.get("cuda_runtime")
    ok = None
    if driver and runtime:
        m = drv_major(runtime)
        if m is None:
            ok = None
        else:
            floor = 525 if m == 12 else (450 if m == 11 else 0)
            ok = drv_major(driver) is not None and drv_major(driver) >= floor
    return {"driver": driver, "cuda_runtime": runtime, "compatible": ok,
            "rule": f"driver >= baseline for CUDA {runtime} (matplotlib-free check, informational)"}


def check_models(do_generate: bool, names=("qwen0p5b",)) -> list[dict]:
    from models.loading import load_spec, check_revision
    hf_home = Path(_hf_home())
    out = []
    for name in names:
        spec = load_spec(name)
        d = check_revision(spec, hf_home)
        d["name"] = name
        if do_generate and d["ok"]:
            d["generate_smoke"] = _generate_smoke(spec)
        out.append(d)
    return out


def _hf_home() -> str:
    import os
    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def _generate_smoke(spec) -> dict:
    try:
        import torch  # noqa
        from models.loading import load_model_and_tokenizer
        model, tok = load_model_and_tokenizer(spec)
        model = model.eval().to("cuda")
        torch.cuda.reset_peak_memory_stats()
        device = next(model.parameters()).device
        msgs = [{"role": "user", "content": "Compute 5 + 7. Answer: 12"}]
        enc = tok(tok.apply_chat_template([{"role": "user", "content": msgs[0]["content"]}],
                                          tokenize=False, add_generation_prompt=True),
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"ok": True, "sample": text[:80], "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", action="store_true", help="check pinned revisions")
    ap.add_argument("--generate", action="store_true", help="run a tiny HF generate (models must be cached)")
    ap.add_argument("--model", action="append", default=None,
                    help="model spec(s) to check (default qwen0p5b; repeatable)")
    ap.add_argument("--must-pass", action="store_true",
                    help="exit nonzero unless GPU, bf16 and every requested model check pass")
    args = ap.parse_args(argv)

    report = {"stage": "gpucheck", "gpus": smi(), "torch": torch_report()}
    report["torch"]["driver_compat"] = driver_cuda_compatible(report["gpus"], report["torch"])
    ok = (bool(report["gpus"]) and report["torch"].get("cuda_available", False)
          and bool(report["torch"].get("bf16_supported"))
          and bool(report["torch"].get("tests", {}).get("bf16_matmul")))
    if args.models:
        try:
            report["models"] = check_models(args.generate, tuple(args.model or ("qwen0p5b",)))
        except FileNotFoundError as e:
            report["models"] = [{"error": str(e)}]
        ok = ok and all(m.get("ok") and (not args.generate or m.get("generate_smoke", {}).get("ok"))
                        for m in report["models"])
    report["ok"] = ok
    print(json.dumps(report, indent=2))
    if args.must_pass and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
