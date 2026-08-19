"""Resume adaptation analysis with a two-worker attempt and safe fallback.

This wrapper changes scheduling only. It does not alter scientific configuration,
seeds, checkpoints, target construction, geometry, or statistical endpoints.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import run_human_adaptation_queue as queue


ORIGINAL_SAFE_LOAD = queue.yaml.safe_load
ANALYSIS_WORKERS = 2


def safe_load_runtime_workers(stream: object) -> object:
    value = ORIGINAL_SAFE_LOAD(stream)
    if isinstance(value, dict) and isinstance(value.get("execution"), dict):
        value["execution"]["gpu_workers"] = 1
        value["execution"]["analysis_gpu_workers"] = ANALYSIS_WORKERS
    return value


def main() -> None:
    global ANALYSIS_WORKERS
    queue.yaml.safe_load = safe_load_runtime_workers
    provenance = queue.EXPERIMENT / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    amendment = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "reason": "recover three incomplete work units after subprocess exit -9",
        "change": (
            "validate production serially; run stage analysis with two GPU workers; "
            "automatically retry unfinished analysis with one worker after a failure"
        ),
        "scientific_configuration_changed": False,
        "adaptation_grid_expected": 1000,
        "adaptation_grid_repaired_before_resume": 1000,
        "cgroup_oom_events_at_restart": 0,
        "cgroup_oom_kills_at_restart": 0,
    }
    (provenance / "operational-resume-worker-fallback.json").write_text(
        json.dumps(amendment, indent=2) + "\n", encoding="utf-8"
    )
    try:
        queue.production()
        queue.reconstruct_random()
        try:
            queue.analyses()
        except Exception as first_error:
            ANALYSIS_WORKERS = 1
            fallback = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "two_worker_error": repr(first_error),
                "action": "retry only unfinished analysis outputs with one worker",
                "scientific_configuration_changed": False,
            }
            (provenance / "operational-analysis-worker-fallback.json").write_text(
                json.dumps(fallback, indent=2) + "\n", encoding="utf-8"
            )
            queue.analyses()
        queue.finalize()
    except Exception as exc:
        queue.write_state("failed", "FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
