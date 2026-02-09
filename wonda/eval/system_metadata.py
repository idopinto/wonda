"""
System metadata collection for experiment reproducibility.

Gathers host, CPU, memory, GPU, and environment information in a
best-effort manner — no hard dependencies, never raises.
"""

import os
import platform
import re
import socket
import subprocess
from pathlib import Path


def _safe_run(cmd: list[str]) -> str | None:
    """Run a command and return stdout (stripped) or None on failure."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        out = (completed.stdout or "").strip()
        return out if out else None
    except Exception:
        return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except Exception:
        return None


def _parse_cpu_model_from_proc_cpuinfo() -> str | None:
    txt = _read_text("/proc/cpuinfo")
    if not txt:
        return None
    # Common fields: "model name" (x86), "Hardware" (ARM)
    m = re.search(r"^model name\s*:\s*(.+)$", txt, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^Hardware\s*:\s*(.+)$", txt, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def _mem_total_gb_from_proc_meminfo() -> float | None:
    txt = _read_text("/proc/meminfo")
    if not txt:
        return None
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB$", txt, flags=re.MULTILINE)
    if not m:
        return None
    kb = int(m.group(1))
    return kb / (1024 * 1024)


def _nvidia_smi_gpus() -> list[dict]:
    """
    Return GPU metadata via nvidia-smi if available.
    Example fields: name, vram_total_mb, driver_version.
    """
    q = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    out = _safe_run(q)
    if not out:
        return []
    gpus: list[dict] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem_mb, driver = parts[0], parts[1], parts[2]
        try:
            mem_mb_f = float(mem_mb)
        except Exception:
            mem_mb_f = None
        gpus.append(
            {
                "name": name,
                "vram_total_mb": mem_mb_f,
                "driver_version": driver,
            }
        )
    return gpus


def extract_override_value(overrides: list[str], key: str) -> str | None:
    """Extract 'key=value' from Hydra override strings (best-effort)."""
    prefix = f"{key}="
    for o in overrides:
        if o.startswith(prefix):
            return o[len(prefix) :]
    return None


def collect_system_metadata(hydra_overrides: list[str]) -> dict:
    """Collect host/system metadata for reproducibility (best-effort, no hard deps)."""
    cpu_model = _parse_cpu_model_from_proc_cpuinfo()
    mem_gb = _mem_total_gb_from_proc_meminfo()
    meta = {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": cpu_model,
            "cores_logical": os.cpu_count(),
        },
        "memory": {
            "ram_total_gb": mem_gb,
        },
        "gpu": {
            "nvidia": _nvidia_smi_gpus(),
        },
        "env": {
            "weave_parallelism": os.environ.get("WEAVE_PARALLELISM"),
        },
    }

    # Optional: physical core count if psutil exists
    try:
        import psutil  # type: ignore

        meta["cpu"]["cores_physical"] = psutil.cpu_count(logical=False)
        meta["memory"]["ram_available_gb"] = psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    return meta
