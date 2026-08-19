"""Continuous resumable Phase 1-5 queue for the ordinary research article."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from autodl_queue import QueueFailure, Supervisor, atomic_json, now, read_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "automation" / "state" / "ordinary-study"
PILOTS = (
    ("gabor", "sub-10"),
    ("somato", "sub1"),
    ("kronemer", "223_RP_EEG"),
    ("kronemer", "238_NRP"),
)


class OrdinaryStudySupervisor(Supervisor):
    def __init__(self) -> None:
        super().__init__()
        self.queue_state["study_mode"] = "ordinary_confirmatory_research_article"
        self._write_queue(status="STARTING", ordinary_state=str(STATE.relative_to(ROOT)))

    @staticmethod
    def snakemake(targets: list[str], cores: int = 25) -> list[str]:
        return [
            "snakemake",
            "--snakefile", "workflows/Snakefile",
            "--cores", str(cores),
            # The AutoDL container has a 90-GiB memory cgroup.  Three 24-GiB
            # human-fold jobs can run concurrently while preprocessing remains
            # serial because its rule reserves 80 GiB.
            "--resources", "gpu=3", "mem_gb=85", "download_slots=1",
            "--rerun-incomplete",
            "--printshellcmds",
            "--retries", "2",
            *targets,
        ]

    @staticmethod
    def included_participants(path: Path | None = None) -> list[dict[str, str]]:
        source = path or ROOT / "configs/execution/participants.tsv"
        with source.open(encoding="utf-8", newline="") as handle:
            return [
                row
                for row in csv.DictReader(handle, delimiter="\t")
                if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
            ]

    def ordinary_status(self, phase: int) -> str | None:
        payload = read_json(STATE / f"phase{phase}.status.json")
        return None if payload is None else str(payload.get("status"))

    def set_ordinary(self, phase: int, status: str, **values: object) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": phase,
            "status": status,
            "study_mode": "ordinary_confirmatory_research_article",
            "updated_at_utc": now(),
            **values,
        }
        atomic_json(STATE / f"phase{phase}.status.json", payload)
        self._write_queue(ordinary_phase=phase, ordinary_phase_status=status)

    def phase1_acquisition(self) -> None:
        if self.ordinary_status(1) == "PASS":
            return
        self.set_ordinary(1, "RUNNING", stage="full-human-acquisition")
        self.run_command(
            "ordinary-phase1",
            "full-acquisition",
            ["bash", "automation/full_acquisition.sh"],
            retries=3,
        )
        participants = self.included_participants()
        counts = {
            dataset: sum(row["dataset_id"] == dataset for row in participants)
            for dataset in ("gabor", "kronemer", "somato")
        }
        if any(value < 1 for value in counts.values()):
            raise QueueFailure(f"full acquisition produced an empty dataset: {counts}")
        self.set_ordinary(1, "PASS", finished_at_utc=now(), participant_counts=counts)

    @staticmethod
    def machine_ready() -> bool:
        gate = read_json(ROOT / "results/gates/machine-production-five-architectures.json")
        return bool(gate and gate.get("execution_complete"))

    @staticmethod
    def machine_queue_running() -> bool:
        result = subprocess.run(
            ["pgrep", "-f", "bash automation/run_unlimited_shared_state.sh"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def phase2_machines(self) -> None:
        if self.ordinary_status(2) == "PASS":
            return
        while not self.machine_ready() and self.machine_queue_running():
            summaries = len(list((ROOT / "results/machine/unlimited_shared_state").glob("seed-*/summary.json")))
            interventions = len(list((ROOT / "results/machine").glob("*/seed-*/intervention.json")))
            self.set_ordinary(
                2,
                "WAITING_ACTIVE_MACHINE_QUEUE",
                fifth_architecture_summaries=summaries,
                total_interventions=interventions,
            )
            time.sleep(60)
        if not self.machine_ready():
            self.set_ordinary(2, "RUNNING", stage="five-architecture-completion")
            self.run_command(
                "ordinary-phase2",
                "machine-completion",
                ["bash", "automation/run_unlimited_shared_state.sh"],
                retries=3,
            )
        self.run_command(
            "ordinary-phase2",
            "machine-execution-audit",
            [
                sys.executable,
                "automation/check_machine_gate.py",
                "--mode", "production",
                "--allow-scientific-failure",
                "--output", "results/gates/machine-production-five-architectures.json",
            ],
        )
        self.set_ordinary(2, "PASS", finished_at_utc=now(), machine_runs=100)

    def phase3_pipeline_validation(self) -> None:
        if self.ordinary_status(3) == "PASS":
            return
        self.set_ordinary(3, "RUNNING", stage="excluded-participant-pipeline-validation")
        targets = [
            f"results/human/{dataset}/{participant}/fold-{fold}/summary.json"
            for dataset, participant in PILOTS
            for fold in range(5)
        ]
        self.run_command(
            "ordinary-phase3",
            "human-pipeline-validation",
            self.snakemake(targets, cores=24),
            retries=3,
        )
        self.set_ordinary(3, "PASS", finished_at_utc=now(), excluded_pilot_folds=20)

    def phase4_human_and_theory(self) -> None:
        if self.ordinary_status(4) == "PASS":
            return
        acquired = ROOT / "configs/execution/participants-acquired.tsv"
        roster = acquired if acquired.exists() else ROOT / "configs/execution/participants.tsv"
        participants = self.included_participants(roster)
        self.set_ordinary(4, "RUNNING", stage="outcome-blind-preprocessing", participants=len(participants))
        preprocessing = [
            f"data/derivatives/preprocessed/{row['dataset_id']}/{row['participant_id']}/qc.json"
            for row in participants
        ]
        self.run_command(
            "ordinary-phase4",
            "preprocess-all-participants",
            self.snakemake(preprocessing, cores=25),
            retries=3,
        )
        self.run_command(
            "ordinary-phase4",
            "freeze-qc-roster",
            [sys.executable, "automation/apply_preprocessing_qc.py"],
        )
        targets = [
            "results/aggregate/human.parquet",
            "results/prediction/nested-cv/summary.json",
            "results/theory-comparison/theory-comparison.json",
        ]
        for dataset in ("gabor", "somato", "kronemer"):
            config = yaml.safe_load(
                (ROOT / f"configs/datasets/{dataset}.yaml").read_text(encoding="utf-8")
            )
            targets.extend(
                f"results/statistics/{dataset}-{index}/bayes-factor.json"
                for index, _ in enumerate(config.get("primary_contrasts", []))
            )
        self.set_ordinary(4, "RUNNING", stage="cross-fitted-human-and-theory-analysis")
        self.run_command(
            "ordinary-phase4",
            "human-theory-production",
            self.snakemake(targets, cores=25),
            retries=3,
        )
        self.run_command(
            "ordinary-phase4",
            "human-quality-audit",
            [
                sys.executable,
                "automation/check_human_gate.py",
                "--allow-quality-failure",
                "--output", "results/gates/human-production.json",
            ],
        )
        self.set_ordinary(4, "PASS", finished_at_utc=now())

    def phase5_article(self) -> None:
        if self.ordinary_status(5) == "PASS":
            return
        self.set_ordinary(5, "RUNNING", stage="figures-results-and-final-audit")
        self.run_command(
            "ordinary-phase5",
            "final-products",
            self.snakemake(
                ["results/study_complete.flag", "results/article/results-digest.md"],
                cores=25,
            ),
            retries=3,
        )
        self.run_command(
            "ordinary-phase5",
            "final-inventory",
            [
                sys.executable,
                "automation/finalize_results.py",
                "--output", "results/provenance/final-results-inventory.json",
            ],
        )
        self.run_command(
            "ordinary-phase5", "freeze-provenance", [sys.executable, "automation/freeze_provenance.py"]
        )
        self.set_ordinary(5, "PASS", finished_at_utc=now())

    def run_ordinary(self) -> None:
        self._write_queue(status="RUNNING", study_mode="ordinary_confirmatory_research_article")
        phases = (
            (1, self.phase1_acquisition),
            (2, self.phase2_machines),
            (3, self.phase3_pipeline_validation),
            (4, self.phase4_human_and_theory),
            (5, self.phase5_article),
        )
        for number, function in phases:
            consecutive_failures = 0
            while self.ordinary_status(number) != "PASS":
                try:
                    function()
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    delay = min(1800, 60 * (2 ** min(consecutive_failures - 1, 5)))
                    self.set_ordinary(
                        number,
                        "RETRYING",
                        error=f"{type(exc).__name__}: {exc}",
                        consecutive_failures=consecutive_failures,
                        retry_delay_seconds=delay,
                    )
                    time.sleep(delay)
        self._write_queue(status="PASS", finished_at_utc=now(), ordinary_phase=5)


def main() -> None:
    os.chdir(ROOT)
    STATE.mkdir(parents=True, exist_ok=True)
    lock_path = STATE / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another ordinary-study queue owns the lock") from exc
        supervisor = OrdinaryStudySupervisor()
        supervisor.run_ordinary()


if __name__ == "__main__":
    main()
