"""Resumable end-to-end AutoDL queue for Experiment 2."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/analysis/human_adaptation.yaml"
MACHINE_CONFIG = ROOT / "configs/models/machine.yaml"
EXPERIMENT = ROOT / "results-extension/human-adaptation"
STATE = EXPERIMENT / "queue-state.json"
ARCHITECTURES = [
    "feedforward",
    "recurrent",
    "shared_workspace",
    "private_modules",
    "unlimited_shared_state",
]
CONDITIONS = ["human_adapted", "sham_adapted"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_state(phase: str, status: str, **details: Any) -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    value = {"updated_at": now(), "phase": phase, "status": status, **details}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_result_manifest() -> dict[str, Any]:
    files = [path for path in (ROOT / "results").rglob("*") if path.is_file()]
    return {
        "created_at": now(),
        "root": "results",
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files)
        },
    }


def verify_frozen_results(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing: {relative}")
        elif path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            failures.append(f"changed: {relative}")
    if failures:
        raise RuntimeError("Experiment 1 immutability failure: " + "; ".join(failures[:20]))


def run(command: list[str], log: Path | None = None) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "OPENBLAS_NUM_THREADS": "6",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if log is None:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def python_command(*arguments: str) -> list[str]:
    return [sys.executable, *arguments]


def adaptation_job(
    architecture: str,
    seed: int,
    fold: int,
    condition: str,
    *,
    output_root: Path,
    max_steps: int | None = None,
) -> tuple[str, float]:
    output = output_root / architecture / f"seed-{seed}" / f"fold-{fold}" / condition
    summary = output / "summary.json"
    if summary.exists() and (output / "model.pt").exists():
        return f"{architecture}/{seed}/{fold}/{condition}", 0.0
    command = python_command(
        "-m",
        "jacaccess.machine.human_adaptation",
        "adapt",
        "--architecture",
        architecture,
        "--seed",
        str(seed),
        "--outer-fold",
        str(fold),
        "--condition",
        condition,
        "--config",
        str(CONFIG),
        "--machine-config",
        str(MACHINE_CONFIG),
        "--target",
        str(EXPERIMENT / "targets" / f"fold-{fold}" / "neural-targets.npz"),
        "--parent-checkpoint",
        str(ROOT / "results/machine" / architecture / f"seed-{seed}" / "model.pt"),
        "--stimulus-cache",
        str(ROOT / "data/derivatives/machine-stimuli" / f"seed-{seed}"),
        "--output",
        str(output),
    )
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    started = time.monotonic()
    run(
        command,
        EXPERIMENT
        / "logs"
        / "adaptation"
        / architecture
        / f"seed-{seed}-fold-{fold}-{condition}.log",
    )
    return f"{architecture}/{seed}/{fold}/{condition}", time.monotonic() - started


def run_parallel(jobs: list[tuple[Any, ...]], worker: Any, workers: int, phase: str) -> list[Any]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, *job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            write_state(phase, "RUNNING", completed=index, total=len(jobs), latest=results[-1][0])
    return results


def analyze_job(
    architecture: str,
    seed: int,
    fold: int,
    stage: str,
) -> tuple[str, float]:
    if stage == "random_init":
        model = EXPERIMENT / "checkpoints-random" / architecture / f"seed-{seed}" / "model.pt"
        output = EXPERIMENT / "stage-analysis" / stage / architecture / f"seed-{seed}"
    else:
        model = (
            EXPERIMENT
            / "checkpoints"
            / architecture
            / f"seed-{seed}"
            / f"fold-{fold}"
            / stage
            / "model.pt"
        )
        output = (
            EXPERIMENT
            / "stage-analysis"
            / stage
            / architecture
            / f"seed-{seed}"
            / f"fold-{fold}"
        )
    if (output / "intervention.json").exists() and (
        output / "jacobian-signatures.parquet"
    ).exists():
        return f"{stage}/{architecture}/{seed}/{fold}", 0.0
    started = time.monotonic()
    run(
        python_command(
            "-m",
            "jacaccess.machine.analyze",
            "--architecture",
            architecture,
            "--seed",
            str(seed),
            "--config",
            str(MACHINE_CONFIG),
            "--model",
            str(model),
            "--output",
            str(output),
        ),
        EXPERIMENT
        / "logs"
        / "analysis"
        / stage
        / architecture
        / f"seed-{seed}-fold-{fold}.log",
    )
    return f"{stage}/{architecture}/{seed}/{fold}", time.monotonic() - started


def prepare() -> None:
    write_state("phase-0-audit", "RUNNING")
    provenance = EXPERIMENT / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    manifest = provenance / "frozen-input-hashes.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(frozen_result_manifest(), indent=2) + "\n", encoding="utf-8"
        )
    verify_frozen_results(manifest)
    run(
        python_command(
            "-m",
            "jacaccess.machine.human_adaptation",
            "prepare-targets",
            "--config",
            str(CONFIG),
            "--roster",
            str(ROOT / "configs/execution/participants.tsv"),
            "--preprocessed-root",
            str(ROOT / "data/derivatives/preprocessed/gabor"),
            "--output",
            str(EXPERIMENT / "targets"),
        ),
        EXPERIMENT / "logs/prepare-targets.log",
    )
    redacted = EXPERIMENT / "splits/gabor-outer-fold-manifest-redacted.json"
    redacted.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPERIMENT / "targets/gabor-outer-fold-manifest-redacted.json", redacted)
    write_state("phase-0-audit", "COMPLETE")


def smoke() -> dict[str, Any]:
    write_state("phase-2-smoke", "RUNNING")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    steps = int(config["execution"]["smoke_steps"])
    durations = []
    for condition in CONDITIONS:
        _, duration = adaptation_job(
            "feedforward",
            0,
            0,
            condition,
            output_root=EXPERIMENT / "smoke/checkpoints",
            max_steps=steps,
        )
        durations.append(duration)
    run(
        python_command(
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_human_adaptation*.py",
        ),
        EXPERIMENT / "logs/smoke-tests.log",
    )
    summaries = [
        json.loads(
            (
                EXPERIMENT
                / "smoke/checkpoints/feedforward/seed-0/fold-0"
                / condition
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for condition in CONDITIONS
    ]
    estimate = {
        "smoke_steps": steps,
        "smoke_wall_time_seconds": sum(durations),
        "estimated_adaptation_seconds_per_run": (
            sum(durations) / max(len(durations), 1) * int(config["adaptation"]["max_steps"]) / steps
        ),
        "planned_adaptation_runs": len(ARCHITECTURES) * int(config["seeds"]) * int(
            config["outer_subject_folds"]
        ) * len(CONDITIONS),
        "performance_gates_passed": all(value["performance_gate_passed"] for value in summaries),
        "note": "Analysis/intervention time is estimated separately after the first production analysis.",
    }
    (EXPERIMENT / "gates").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "gates/human-adaptation-smoke.json").write_text(
        json.dumps({"passed": estimate["performance_gates_passed"], **estimate}, indent=2) + "\n",
        encoding="utf-8",
    )
    (EXPERIMENT / "compute-estimate.json").write_text(
        json.dumps(estimate, indent=2) + "\n", encoding="utf-8"
    )
    if not estimate["performance_gates_passed"]:
        raise RuntimeError("smoke test failed the task-retention gate")
    write_state("phase-2-smoke", "COMPLETE", **estimate)
    return estimate


def freeze() -> None:
    write_state("phase-3-freeze", "RUNNING")
    config_dir = EXPERIMENT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    frozen = config_dir / "human-adaptation-frozen.yaml"
    shutil.copy2(CONFIG, frozen)
    (config_dir / "config-sha256.txt").write_text(sha256(frozen) + "\n", encoding="utf-8")
    governed = [
        CONFIG,
        ROOT / "src/jacaccess/machine/human_adaptation.py",
        ROOT / "src/jacaccess/machine/stage_comparison.py",
        ROOT / "src/jacaccess/stats/adaptation_comparison.py",
        Path(__file__),
    ]
    source = {
        "frozen_at": now(),
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in governed
        },
    }
    (EXPERIMENT / "provenance/source-freeze.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    write_state("phase-3-freeze", "COMPLETE", config_sha256=sha256(frozen))


def production() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    jobs = [
        (architecture, seed, fold, condition)
        for architecture in ARCHITECTURES
        for seed in range(int(config["seeds"]))
        for fold in range(int(config["outer_subject_folds"]))
        for condition in CONDITIONS
    ]

    def worker(architecture: str, seed: int, fold: int, condition: str) -> tuple[str, float]:
        return adaptation_job(
            architecture,
            seed,
            fold,
            condition,
            output_root=EXPERIMENT / "checkpoints",
        )

    write_state("phase-4-production-adaptation", "RUNNING", completed=0, total=len(jobs))
    run_parallel(jobs, worker, int(config["execution"]["gpu_workers"]), "phase-4-production-adaptation")
    write_state("phase-4-production-adaptation", "COMPLETE", completed=len(jobs), total=len(jobs))


def reconstruct_random() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    write_state("phase-4-random-reconstruction", "RUNNING")
    for architecture in ARCHITECTURES:
        for seed in range(int(config["seeds"])):
            output = EXPERIMENT / "checkpoints-random" / architecture / f"seed-{seed}" / "model.pt"
            if output.exists():
                continue
            run(
                python_command(
                    "-m",
                    "jacaccess.machine.human_adaptation",
                    "reconstruct-random",
                    "--architecture",
                    architecture,
                    "--seed",
                    str(seed),
                    "--machine-config",
                    str(MACHINE_CONFIG),
                    "--output",
                    str(output),
                )
            )
    write_state("phase-4-random-reconstruction", "COMPLETE")


def analyses() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    jobs = [
        (architecture, seed, 0, "random_init")
        for architecture in ARCHITECTURES
        for seed in range(int(config["seeds"]))
    ] + [
        (architecture, seed, fold, condition)
        for architecture in ARCHITECTURES
        for seed in range(int(config["seeds"]))
        for fold in range(int(config["outer_subject_folds"]))
        for condition in CONDITIONS
    ]
    write_state("phase-5-stage-analysis", "RUNNING", completed=0, total=len(jobs))
    run_parallel(
        jobs,
        analyze_job,
        int(config["execution"]["analysis_gpu_workers"]),
        "phase-5-stage-analysis",
    )
    write_state("phase-5-stage-analysis", "COMPLETE", completed=len(jobs), total=len(jobs))


def finalize() -> None:
    write_state("phase-7-finalization", "RUNNING")
    aggregate = EXPERIMENT / "aggregate"
    run(
        python_command(
            "-m",
            "jacaccess.machine.stage_comparison",
            "--experiment-root",
            str(EXPERIMENT),
            "--human-table",
            str(ROOT / "results/aggregate/human.parquet"),
            "--split-manifest",
            str(EXPERIMENT / "targets/split-manifest-private.json"),
            "--original-machine-root",
            str(ROOT / "results/machine"),
            "--accuracy-matching",
            str(ROOT / "results/aggregate/machine/accuracy-matching/accuracy-matching.json"),
            "--fallback-weights",
            str(ROOT / "results/aggregate/machine/accuracy-matching/fallback-ipw.parquet"),
            "--output",
            str(aggregate),
        ),
        EXPERIMENT / "logs/stage-comparison.log",
    )
    run(
        python_command(
            "-m",
            "jacaccess.stats.adaptation_comparison",
            "--experiment-root",
            str(EXPERIMENT),
            "--output",
            str(EXPERIMENT / "statistics"),
        ),
        EXPERIMENT / "logs/statistics.log",
    )
    run(
        python_command(
            "scripts/retention_gate_sensitivity.py",
            "--experiment-root",
            str(EXPERIMENT),
        ),
        EXPERIMENT / "logs/retention-sensitivity.log",
    )
    run(
        python_command(
            "scripts/render_human_adaptation_figures.py",
            "--experiment-root",
            str(EXPERIMENT),
        ),
        EXPERIMENT / "logs/figures.log",
    )
    figure_root = EXPERIMENT / "figures/science-advances-r"
    figure_data = figure_root / "data"
    figure_output = figure_root / "final"
    run(
        python_command(
            "scripts/prepare_science_advances_figure_data.py",
            "--project-root",
            str(ROOT),
            "--extension-root",
            str(EXPERIMENT),
            "--output",
            str(figure_data),
        ),
        EXPERIMENT / "logs/science-advances-figure-data.log",
    )
    run(
        [
            "Rscript",
            "--vanilla",
            str(ROOT / "scripts/render_science_advances_figures.R"),
            "--data",
            str(figure_data),
            "--output",
            str(figure_output),
            "--manifest",
            str(figure_output / "manifest.json"),
        ],
        EXPERIMENT / "logs/science-advances-r-figures.log",
    )
    run(
        python_command(
            "automation/finalize_human_adaptation.py",
            "--experiment-root",
            str(EXPERIMENT),
        ),
        EXPERIMENT / "logs/finalize.log",
    )
    verify_frozen_results(EXPERIMENT / "provenance/frozen-input-hashes.json")
    (EXPERIMENT / "study_complete.flag").touch()
    write_state("complete", "COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-phase",
        choices=["prepare", "smoke", "freeze", "production", "analysis", "finalize"],
        default="prepare",
    )
    args = parser.parse_args()
    phases = ["prepare", "smoke", "freeze", "production", "analysis", "finalize"]
    start = phases.index(args.from_phase)
    try:
        if start <= 0:
            prepare()
        if start <= 1:
            smoke()
        if start <= 2:
            freeze()
        if start <= 3:
            production()
            reconstruct_random()
        if start <= 4:
            analyses()
        if start <= 5:
            finalize()
    except Exception as exc:
        write_state("failed", "FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
