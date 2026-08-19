"""Resumable Phase 0-3 AutoDL supervisor with resource and scientific gates."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "automation" / "state" / "full-study"
LOGS = ROOT / "logs" / "full-study"
ARCHITECTURES = (
    "feedforward", "recurrent", "shared_workspace", "private_modules",
    "unlimited_shared_state",
)
MINIMUM_FREE_GIB = 500.0
MAXIMUM_CGROUP_RAM_GIB = 600.0
HUMAN_PILOT_MAXIMUM_RAM_GIB = 80.0


class QueueFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cgroup_memory_gib() -> float | None:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text(encoding="utf-8").strip()) / 2**30
        except (OSError, ValueError):
            continue
    return None


def cgroup_working_set_gib() -> float | None:
    """Return memory.current minus reclaimable inactive file cache."""

    total = cgroup_memory_gib()
    if total is None:
        return None
    try:
        values = {
            name: int(value)
            for name, value in (
                line.split(maxsplit=1)
                for line in Path("/sys/fs/cgroup/memory.stat")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        inactive_file_gib = values.get("inactive_file", 0) / 2**30
        return max(0.0, total - inactive_file_gib)
    except (OSError, ValueError):
        return total


def gpu_snapshot() -> dict[str, object] | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).stdout.strip()
        utilization, used, total, temperature = [int(value.strip()) for value in output.split(",")]
        return {
            "utilization_percent": utilization,
            "memory_used_mib": used,
            "memory_total_mib": total,
            "temperature_c": temperature,
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def resource_snapshot() -> dict[str, object]:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "at_utc": now(),
        "cgroup_ram_gib": cgroup_memory_gib(),
        "cgroup_working_set_gib": cgroup_working_set_gib(),
        "scratch_free_gib": disk.free / 2**30,
        "scratch_used_gib": disk.used / 2**30,
        "gpu": gpu_snapshot(),
    }


class Supervisor:
    def __init__(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        self.queue_state: dict[str, Any] = {
            "status": "STARTING",
            "pid": os.getpid(),
            "started_at_utc": now(),
            "root": str(ROOT),
        }
        self.current_process: subprocess.Popen[str] | None = None
        (STATE / "queue.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._write_queue()
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _write_queue(self, **updates: Any) -> None:
        self.queue_state.update(updates)
        self.queue_state["updated_at_utc"] = now()
        self.queue_state["resources"] = resource_snapshot()
        atomic_json(STATE / "queue.status.json", self.queue_state)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self._write_queue(status="STOPPING", signal=signum)
        if self.current_process is not None and self.current_process.poll() is None:
            os.killpg(self.current_process.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    def phase_status(self, phase: str) -> str | None:
        value = read_json(STATE / f"{phase}.status.json")
        return None if value is None else str(value.get("status"))

    def set_phase(self, phase: str, status: str, **values: Any) -> None:
        payload = {"phase": phase, "status": status, "updated_at_utc": now(), **values}
        atomic_json(STATE / f"{phase}.status.json", payload)
        self._write_queue(phase=phase, phase_status=status)

    def run_command(
        self,
        phase: str,
        name: str,
        command: list[str],
        *,
        retries: int = 1,
        enforce_guards: bool = True,
    ) -> dict[str, Any]:
        receipt_path = STATE / "commands" / f"{phase}--{name}.json"
        existing = read_json(receipt_path)
        if existing and existing.get("status") == "PASS":
            return existing
        last_exit = 1
        for attempt in range(1, retries + 1):
            started = time.monotonic()
            log_path = LOGS / phase / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "JACACCESS_ROOT": str(ROOT),
                    "OMP_NUM_THREADS": "20",
                    "MKL_NUM_THREADS": "20",
                    "OPENBLAS_NUM_THREADS": "20",
                    "NUMEXPR_NUM_THREADS": "20",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            payload: dict[str, Any] = {
                "phase": phase,
                "name": name,
                "status": "RUNNING",
                "attempt": attempt,
                "command": command,
                "started_at_utc": now(),
                "log": str(log_path.relative_to(ROOT)),
            }
            atomic_json(receipt_path, payload)
            peak_ram = 0.0
            peak_total_ram = 0.0
            minimum_free = float("inf")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now()}] attempt {attempt}: {' '.join(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.current_process = process
                payload["pid"] = process.pid
                atomic_json(receipt_path, payload)
                while process.poll() is None:
                    snapshot = resource_snapshot()
                    total_ram = snapshot.get("cgroup_ram_gib")
                    ram = snapshot.get("cgroup_working_set_gib")
                    free = float(snapshot["scratch_free_gib"])
                    if ram is not None:
                        peak_ram = max(peak_ram, float(ram))
                    if total_ram is not None:
                        peak_total_ram = max(peak_total_ram, float(total_ram))
                    minimum_free = min(minimum_free, free)
                    self._write_queue(
                        status="RUNNING",
                        phase=phase,
                        command=name,
                        command_pid=process.pid,
                        command_attempt=attempt,
                        elapsed_seconds=round(time.monotonic() - started, 1),
                    )
                    violation = None
                    if enforce_guards and free < MINIMUM_FREE_GIB:
                        violation = f"scratch guard: {free:.1f} GiB free"
                    if (
                        enforce_guards
                        and ram is not None
                        and float(ram) > MAXIMUM_CGROUP_RAM_GIB
                    ):
                        violation = (
                            f"RAM guard: {float(ram):.1f} GiB non-reclaimable "
                            "working set in use"
                        )
                    if violation:
                        log.write(f"[{now()}] {violation}; terminating process group\n")
                        log.flush()
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        last_exit = 75
                        break
                    time.sleep(15)
                else:
                    last_exit = int(process.returncode or 0)
                if process.poll() is not None:
                    last_exit = int(process.returncode or 0)
                self.current_process = None
            payload.update(
                {
                    "status": "PASS" if last_exit == 0 else "FAILED",
                    "exit_code": last_exit,
                    "finished_at_utc": now(),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "peak_cgroup_ram_gib": round(peak_ram, 3),
                    "peak_cgroup_total_ram_gib": round(peak_total_ram, 3),
                    "minimum_scratch_free_gib": (
                        None if minimum_free == float("inf") else round(minimum_free, 3)
                    ),
                }
            )
            atomic_json(receipt_path, payload)
            if last_exit == 0:
                return payload
            if attempt < retries:
                time.sleep(min(60, 10 * attempt))
        raise QueueFailure(f"{phase}/{name} failed with exit code {last_exit}")

    @staticmethod
    def snakemake(targets: list[str], cores: int = 25) -> list[str]:
        return [
            "snakemake",
            "--snakefile",
            "workflows/Snakefile",
            "--cores",
            str(cores),
            "--resources",
            "gpu=4",
            "mem_gb=85",
            "download_slots=1",
            "--rerun-incomplete",
            "--printshellcmds",
            "--retries",
            "2",
            *targets,
        ]

    def adapter_ready(self) -> bool:
        result = subprocess.run(
            [sys.executable, "scripts/check_adapter_gate.py"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def phase0(self) -> None:
        phase = "phase0"
        if self.phase_status(phase) == "PASS":
            return
        self.set_phase(phase, "RUNNING", started_at_utc=now())
        self.run_command(
            phase,
            "environment",
            [
                sys.executable,
                "-m",
                "jacaccess.environment_check",
                "--scratch",
                "/root/autodl-tmp",
                "--minimum-ram-gib",
                "80",
                "--minimum-scratch-gib",
                "2500",
                "--minimum-cpu-cores",
                "25",
                "--output",
                "environment/environment-report.json",
            ],
        )
        self.run_command(phase, "compile", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "automation"])
        self.run_command(
            phase,
            "tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        )
        self.run_command(
            phase,
            "stage2-ready",
            self.snakemake(["results/stage2_ready.flag"]),
        )
        self.run_command(phase, "freeze", [sys.executable, "automation/freeze_provenance.py"])
        self.set_phase(phase, "PASS", finished_at_utc=now())

    def phase1(self) -> None:
        phase = "phase1"
        status = self.phase_status(phase)
        if status == "PASS":
            return
        if status == "WAITING_REVIEW" and self.adapter_ready():
            self.set_phase(
                phase,
                "PASS",
                completed_at_utc=now(),
                adapter_gate="PASS",
                resumed_from_review=True,
            )
            return
        if status == "WAITING_REVIEW":
            return
        self.set_phase(phase, "RUNNING", started_at_utc=now())
        self.run_command(phase, "pilot-acquisition", ["bash", "automation/phase1_pilots.sh"], retries=3)
        if self.adapter_ready():
            self.set_phase(phase, "PASS", finished_at_utc=now(), adapter_gate="PASS")
        else:
            self.set_phase(
                phase,
                "WAITING_REVIEW",
                finished_at_utc=now(),
                adapter_gate="WAITING_REVIEW",
                evidence="results/source-inspection/phase1-evidence.json",
                review="results/adapter-gate.json",
            )

    def phase3(self) -> None:
        phase = "phase3"
        if self.phase_status(phase) == "PASS":
            return
        self.set_phase(phase, "RUNNING", started_at_utc=now(), stage="benchmark")
        for architecture in ARCHITECTURES:
            self.run_command(
                phase,
                f"benchmark-{architecture}",
                self.snakemake([f"results/machine/{architecture}/seed-0/summary.json"]),
                retries=3,
            )
        self.run_command(
            phase,
            "benchmark-gate",
            [
                sys.executable,
                "automation/check_machine_gate.py",
                "--mode",
                "benchmark",
                "--output",
                "results/gates/machine-benchmark.json",
            ],
        )
        self.set_phase(phase, "RUNNING", stage="production")
        self.run_command(
            phase,
            "production",
            self.snakemake(
                [
                    "results/aggregate/machine/architecture-summary.csv",
                    "results/aggregate/machine/accuracy-matching/accuracy-matching.json",
                ]
            ),
            retries=3,
        )
        self.run_command(
            phase,
            "production-gate",
            [
                sys.executable,
                "automation/check_machine_gate.py",
                "--mode",
                "production",
                "--output",
                "results/gates/machine-production.json",
            ],
        )
        self.set_phase(phase, "PASS", finished_at_utc=now())

    @staticmethod
    def included_participants() -> list[dict[str, str]]:
        with (ROOT / "configs" / "execution" / "participants.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return [
                row
                for row in csv.DictReader(handle, delimiter="\t")
                if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
            ]

    def phase2(self) -> None:
        phase = "phase2"
        if self.phase_status(phase) == "PASS":
            return
        if not self.adapter_ready():
            self.set_phase(
                phase,
                "WAITING_REVIEW",
                adapter_gate="WAITING_REVIEW",
                review="results/adapter-gate.json",
            )
            return
        ipa_path = ROOT / "results" / "registration" / "STAGE1_IPA.json"
        ipa = read_json(ipa_path)
        if not ipa or ipa.get("in_principle_acceptance") is not True:
            subprocess.run(
                [sys.executable, "automation/seal_human_pilot.py"],
                cwd=ROOT,
                check=True,
            )
            self.set_phase(
                phase,
                "WAITING_STAGE1",
                stage="registered-report-gate",
                prospective_boundary="2026-08-11",
                required_receipt=str(ipa_path.relative_to(ROOT)),
                protocol="docs/registered-report/STAGE1_PROTOCOL.md",
            )
            return
        participants = self.included_participants()
        pilots: dict[str, dict[str, str]] = {}
        for row in participants:
            pilots.setdefault(row["dataset_id"], row)
        missing = sorted({"gabor", "somato", "kronemer"} - pilots.keys())
        if missing:
            raise QueueFailure(f"no included pilot participant for {missing}")
        self.set_phase(phase, "RUNNING", started_at_utc=now(), stage="human-pilots")
        pilot_receipts = []
        for dataset in ("gabor", "somato", "kronemer"):
            participant = pilots[dataset]["participant_id"]
            targets = [
                f"results/human/{dataset}/{participant}/fold-{fold}/summary.json"
                for fold in range(5)
            ]
            receipt = self.run_command(
                phase,
                f"pilot-{dataset}-{participant}",
                self.snakemake(targets, cores=20),
                retries=2,
            )
            pilot_receipts.append(
                {
                    "dataset_id": dataset,
                    "participant_id": participant,
                    "peak_cgroup_ram_gib": receipt["peak_cgroup_ram_gib"],
                    "elapsed_seconds": receipt["elapsed_seconds"],
                }
            )
            if float(receipt["peak_cgroup_ram_gib"]) >= HUMAN_PILOT_MAXIMUM_RAM_GIB:
                raise QueueFailure(
                    f"{dataset}/{participant} human pilot reached "
                    f"{receipt['peak_cgroup_ram_gib']} GiB; 80 GiB gate failed"
                )
        atomic_json(
            ROOT / "results" / "gates" / "human-pilots.json",
            {"ready": True, "maximum_ram_gib": 80, "pilots": pilot_receipts},
        )
        self.set_phase(phase, "RUNNING", stage="full-acquisition")
        self.run_command(
            phase,
            "full-acquisition",
            ["bash", "automation/full_acquisition.sh"],
            retries=3,
        )
        self.set_phase(phase, "RUNNING", stage="human-production")
        targets = [
            "results/aggregate/human.parquet",
            "results/prediction/nested-cv/summary.json",
        ]
        for dataset in ("gabor", "somato", "kronemer"):
            config = yaml.safe_load(
                (ROOT / "configs" / "datasets" / f"{dataset}.yaml").read_text(encoding="utf-8")
            )
            for index, _contrast in enumerate(config.get("primary_contrasts", [])):
                targets.append(f"results/statistics/{dataset}-{index}/bayes-factor.json")
        self.run_command(
            phase,
            "human-production",
            self.snakemake(targets),
            retries=2,
        )
        self.run_command(
            phase,
            "human-gate",
            [
                sys.executable,
                "automation/check_human_gate.py",
                "--output",
                "results/gates/human-production.json",
            ],
        )
        self.set_phase(phase, "PASS", finished_at_utc=now())

    def phase4(self) -> None:
        phase = "phase4"
        if self.phase_status(phase) == "PASS":
            return
        if (
            self.phase_status("phase2") != "PASS"
            or self.phase_status("phase3") != "PASS"
        ):
            return
        self.set_phase(phase, "RUNNING", started_at_utc=now(), stage="final-products")
        self.run_command(
            phase,
            "study-complete",
            self.snakemake(["results/study_complete.flag"]),
            retries=2,
        )
        self.run_command(
            phase,
            "final-inventory",
            [
                sys.executable,
                "automation/finalize_results.py",
                "--output",
                "results/provenance/final-results-inventory.json",
            ],
        )
        self.run_command(
            phase,
            "final-freeze",
            [sys.executable, "automation/freeze_provenance.py"],
        )
        self.set_phase(phase, "PASS", finished_at_utc=now())

    def run(self) -> None:
        self._write_queue(status="RUNNING")
        self.phase0()
        try:
            self.phase1()
        except QueueFailure as exc:
            self.set_phase("phase1", "FAILED", error=str(exc), finished_at_utc=now())
        try:
            self.phase2()
        except QueueFailure as exc:
            self.set_phase("phase2", "FAILED", error=str(exc), finished_at_utc=now())
        try:
            self.phase3()
        except QueueFailure as exc:
            self.set_phase("phase3", "FAILED", error=str(exc), finished_at_utc=now())
        if self.phase_status("phase2") == "WAITING_STAGE1":
            self._write_queue(
                status="WAITING_STAGE1",
                phase="phase2",
                required_receipt="results/registration/STAGE1_IPA.json",
                machine_work_continues_independently=True,
            )
            return
        while self.phase_status("phase2") != "PASS":
            if self.adapter_ready():
                try:
                    self.phase2()
                except QueueFailure as exc:
                    self.set_phase("phase2", "FAILED", error=str(exc), finished_at_utc=now())
                    self._write_queue(status="BLOCKED", error=str(exc))
                    return
            else:
                self._write_queue(
                    status="WAITING_REVIEW",
                    phase="phase2",
                    review="results/adapter-gate.json",
                )
                time.sleep(60)
        if self.phase_status("phase3") == "PASS":
            try:
                self.phase4()
            except QueueFailure as exc:
                self.set_phase("phase4", "FAILED", error=str(exc), finished_at_utc=now())
                self._write_queue(status="BLOCKED", error=str(exc))
                return
        if self.phase_status("phase4") == "PASS":
            self._write_queue(status="PASS", finished_at_utc=now())
        else:
            self._write_queue(
                status="BLOCKED",
                error="Phase 3 or Phase 4 failed; inspect its command receipt and log.",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args
    os.chdir(ROOT)
    lock_path = STATE / "queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another full-study queue owns the lock") from exc
        supervisor = Supervisor()
        try:
            supervisor.run()
        except Exception as exc:
            supervisor._write_queue(
                status="BLOCKED",
                error=f"{type(exc).__name__}: {exc}",
                finished_at_utc=now(),
            )
            raise


if __name__ == "__main__":
    main()
