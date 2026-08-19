"""Validate the production host before downloading target data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any


def _host_memory_bytes() -> int | None:
    if os.name == "nt":
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_memory_limit_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Return the effective Linux container memory limit, if it is finite."""

    candidates = (root / "memory.max", root / "memory" / "memory.limit_in_bytes")
    for path in candidates:
        value = _read_text(path)
        if not value or value == "max":
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if 0 < limit < 2**60:
            return limit
    return None


def _cgroup_cpu_cores(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """Return the cgroup CPU quota as a possibly fractional core count."""

    value = _read_text(root / "cpu.max")
    if value:
        parts = value.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return int(parts[0]) / int(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    quota = _read_text(root / "cpu" / "cpu.cfs_quota_us")
    period = _read_text(root / "cpu" / "cpu.cfs_period_us")
    if quota and period:
        try:
            quota_value = int(quota)
            if quota_value > 0:
                return quota_value / int(period)
        except (ValueError, ZeroDivisionError):
            pass
    return None


def inspect_environment(scratch_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scratch_path": str(scratch_path.resolve()),
    }
    host_memory = _host_memory_bytes()
    cgroup_memory = _cgroup_memory_limit_bytes()
    effective_memory_candidates = [
        value for value in (host_memory, cgroup_memory) if value is not None
    ]
    effective_memory = min(effective_memory_candidates) if effective_memory_candidates else None
    report["host_system_ram_gib"] = None if host_memory is None else host_memory / 2**30
    report["cgroup_ram_limit_gib"] = (
        None if cgroup_memory is None else cgroup_memory / 2**30
    )
    report["system_ram_gib"] = None if effective_memory is None else effective_memory / 2**30
    report["host_cpu_count"] = os.cpu_count()
    report["cgroup_cpu_cores"] = _cgroup_cpu_cores()
    effective_cpu_candidates = [
        float(value)
        for value in (os.cpu_count(), report["cgroup_cpu_cores"])
        if value is not None
    ]
    report["effective_cpu_cores"] = (
        min(effective_cpu_candidates) if effective_cpu_candidates else None
    )
    usage = shutil.disk_usage(scratch_path)
    report["scratch_total_gib"] = usage.total / 2**30
    report["scratch_free_gib"] = usage.free / 2**30

    try:
        import torch
    except ImportError:
        report.update(
            {
                "torch_installed": False,
                "cuda_available": False,
                "gpu_name": None,
                "gpu_vram_gib": None,
                "compute_capability": None,
            }
        )
        return report

    report["torch_installed"] = True
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report["gpu_name"] = properties.name
        report["gpu_vram_gib"] = properties.total_memory / 2**30
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        report["gpu_name"] = None
        report["gpu_vram_gib"] = None
        report["compute_capability"] = None
    return report


def validate_production_environment(
    report: dict[str, Any],
    *,
    minimum_ram_gib: float = 100,
    minimum_scratch_gib: float = 2500,
    minimum_cpu_cores: float = 25,
) -> list[str]:
    failures: list[str] = []
    if not report.get("torch_installed"):
        failures.append("PyTorch is not installed")
    if not report.get("cuda_available"):
        failures.append("CUDA is not available")
    gpu_name = report.get("gpu_name") or ""
    if "5090" not in gpu_name:
        failures.append(f"expected an RTX 5090, found {gpu_name or 'no GPU'}")
    if (report.get("gpu_vram_gib") or 0) < 30:
        failures.append("less than 30 GiB GPU memory is visible")
    capability = report.get("compute_capability")
    if capability is not None and tuple(capability) < (12, 0):
        failures.append(f"compute capability {capability} is below 12.0")
    ram = report.get("system_ram_gib")
    if ram is not None and ram < minimum_ram_gib:
        failures.append(
            f"only {ram:.1f} GiB effective system RAM is visible; "
            f"{minimum_ram_gib:.1f} GiB required"
        )
    cpu = report.get("effective_cpu_cores")
    if cpu is not None and cpu < minimum_cpu_cores:
        failures.append(
            f"only {cpu:.2f} effective CPU cores are visible; "
            f"{minimum_cpu_cores:.2f} required"
        )
    if report.get("scratch_free_gib", 0) < minimum_scratch_gib:
        failures.append(f"less than {minimum_scratch_gib:.1f} GiB scratch space is free")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-development-host", action="store_true")
    parser.add_argument("--minimum-ram-gib", type=float, default=100)
    parser.add_argument("--minimum-scratch-gib", type=float, default=2500)
    parser.add_argument("--minimum-cpu-cores", type=float, default=25)
    args = parser.parse_args()

    report = inspect_environment(args.scratch)
    report["validation_thresholds"] = {
        "minimum_ram_gib": args.minimum_ram_gib,
        "minimum_scratch_gib": args.minimum_scratch_gib,
        "minimum_cpu_cores": args.minimum_cpu_cores,
    }
    report["production_failures"] = validate_production_environment(
        report,
        minimum_ram_gib=args.minimum_ram_gib,
        minimum_scratch_gib=args.minimum_scratch_gib,
        minimum_cpu_cores=args.minimum_cpu_cores,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if report["production_failures"] and not args.allow_development_host:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
