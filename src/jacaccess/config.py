"""Configuration loading, validation and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("configuration loading requires PyYAML") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def configuration_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_analysis_config(config: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    study = config.get("study", {})
    crossfit = config.get("crossfit", {})
    jacobian = config.get("jacobian", {})
    compute = config.get("compute", {})

    if study.get("registered_report_claim") is not False:
        failures.append("ordinary-study configuration must set registered_report_claim=false")
    if int(crossfit.get("folds", 0)) < 2:
        failures.append("crossfit.folds must be at least two")
    latent_dimensions = int(crossfit.get("latent_dimensions", 0))
    if latent_dimensions < 4:
        failures.append("crossfit.latent_dimensions must be at least four")
    horizons = jacobian.get("horizons_samples", [])
    if not isinstance(horizons, list) or not horizons:
        failures.append("jacobian.horizons_samples must be a non-empty list")
    elif any(int(h) < 1 for h in horizons) or sorted(set(horizons)) != horizons:
        failures.append("jacobian horizons must be unique, sorted positive integers")
    rank = int(jacobian.get("rank", 0))
    if rank < 1 or rank > latent_dimensions:
        failures.append("jacobian.rank must be between one and latent dimension")
    weights = jacobian.get("component_weights", [])
    components = jacobian.get("access_index_components", [])
    expected_count = len(components) if components else len(weights)
    if (
        not weights
        or len(weights) != expected_count
        or abs(sum(float(value) for value in weights) - 1.0) > 1e-6
    ):
        failures.append("Access Index weights must match its components and sum to one")
    if bool(jacobian.get("persist_full_tensors", True)):
        failures.append("persist_full_tensors must remain false for bounded storage")
    if int(compute.get("local_scratch_gb_minimum", 0)) < 2500:
        failures.append("compute.local_scratch_gb_minimum must be at least 2500")
    return failures


def validate_machine_config(config: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_heads = {
        "presence",
        "orientation",
        "location",
        "contrast_bin",
        "delayed_action",
    }
    if set(config.get("output_heads", [])) != expected_heads:
        failures.append(f"machine output_heads must equal {sorted(expected_heads)}")
    if int(config.get("integration_state_dimensions", 0)) != 32:
        failures.append("machine integration state must be exactly 32 dimensions")
    if int(config.get("internal_steps", 0)) != 6:
        failures.append("machine internal_steps must equal six")
    if int(config.get("seeds", 0)) < 2:
        failures.append("machine experiment needs at least two paired seeds")
    target = int(config.get("parameter_target", 0))
    tolerance = float(config.get("parameter_tolerance_fraction", -1))
    if target <= 0 or not 0 <= tolerance <= 0.25:
        failures.append("machine parameter target or tolerance is invalid")
    return failures


def validate_repository_configs(root: Path) -> dict[str, Any]:
    analysis_path = root / "configs" / "analysis" / "primary.yaml"
    machine_path = root / "configs" / "models" / "machine.yaml"
    human_path = root / "configs" / "models" / "human.yaml"
    analysis = load_yaml(analysis_path)
    machine = load_yaml(machine_path)
    human = load_yaml(human_path)
    failures = validate_analysis_config(analysis) + validate_machine_config(machine)
    return {
        "valid": not failures,
        "failures": failures,
        "hashes": {
            "analysis": configuration_hash(analysis),
            "human": configuration_hash(human),
            "machine": configuration_hash(machine),
        },
    }
