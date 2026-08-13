"""End-to-end raw telemetry to model-ready sequence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .features import aggregate_mission_features, to_phase_feature_schema
from .preprocessing import canonicalize_telemetry, summarize_cycle_instances
from .sequences import (
    SequenceBatch,
    build_loco_manifest,
    build_rul_sequence_batch,
    build_soc_sequence_batch,
    build_soh_sequence_batch,
)
from .targets import construct_health_targets, threshold_token, validate_threshold_order
from .telemetry_features import build_enriched_phase_instances


@dataclass(frozen=True)
class CellPipelineResult:
    cell_id: str
    telemetry: pd.DataFrame
    cycle_index: pd.DataFrame
    capacity_tests: pd.DataFrame
    phase_instances: pd.DataFrame
    phase_features: pd.DataFrame
    phase_targets: pd.DataFrame
    mission_features: pd.DataFrame
    mission_targets: pd.DataFrame


def _legacy_rul_threshold(config: PipelineConfig) -> float:
    thresholds = tuple(float(value) for value in config.targets.rul_thresholds_pct)
    return 85.0 if 85.0 in thresholds else thresholds[0]


def process_cell_frame(
    raw: pd.DataFrame,
    *,
    cell_id: str,
    config: PipelineConfig,
) -> CellPipelineResult:
    """Run every deterministic preprocessing stage for one in-memory cell."""

    config.validate()
    telemetry = canonicalize_telemetry(raw, cell_id=cell_id, config=config.preprocessing)
    cycle_index = summarize_cycle_instances(
        telemetry,
        preprocessing=config.preprocessing,
        capacity_test=config.capacity_test,
    )
    health = construct_health_targets(cycle_index, config=config.targets)
    threshold_errors = validate_threshold_order(
        health.mission_targets, config.targets.rul_thresholds_pct
    )
    if threshold_errors:
        raise ValueError(f"{cell_id}: invalid RUL threshold ordering: {threshold_errors}")
    phase_instances = build_enriched_phase_instances(
        telemetry,
        cycle_index,
        health.mission_targets,
        config=config.preprocessing,
        legacy_rul_threshold_pct=_legacy_rul_threshold(config),
    )
    phase_schema = to_phase_feature_schema(phase_instances)

    # Preserve the reconstructed identifier even though the legacy published
    # schema predates it. mission_id + seg_id is unique within a cell.
    trace = phase_instances[
        ["cell_id", "mission_id", "seg_id", "cycle_instance_id"]
    ].drop_duplicates(["cell_id", "mission_id", "seg_id"])
    phase_schema = phase_schema.merge(
        trace,
        on=["cell_id", "mission_id", "seg_id"],
        how="left",
        validate="one_to_one",
    )
    columns = phase_schema.columns.tolist()
    columns.insert(columns.index("cycleNumber") + 1, columns.pop(columns.index("cycle_instance_id")))
    phase_schema = phase_schema[columns]

    phase_targets = phase_instances[
        [
            "cell_id",
            "mission_id",
            "cycle_instance_id",
            "seg_id",
            "phase_name",
            "start_SOC",
            "end_SOC",
            "delta_SOC",
            "SOH_mission_end_pct",
            "RUL_missions_after_phase",
            "RUL_missions_censored",
        ]
    ].rename(
        columns={
            "start_SOC": "SOC_start_pct",
            "end_SOC": "SOC_end_pct",
            "delta_SOC": "dis_dSOC",
        }
    )
    phase_features = phase_schema.drop(
        columns=[
            "dis_dSOC",
            "SOH_mission_end_pct",
            "RUL_missions_after_phase",
            "RUL_missions_censored",
        ]
    )

    mission_legacy = aggregate_mission_features(
        phase_schema,
        cv_setpoints_v=config.cv_setpoints_v,
        default_cv_setpoint_v=config.default_cv_setpoint_v,
    )
    mission_features = mission_legacy.drop(
        columns=["dis_dSOC_sum", "SOH_end_pct", "RUL_med"], errors="raise"
    )
    mission_targets = health.mission_targets.merge(
        mission_legacy[["cell_id", "mission_id", "dis_dSOC_sum"]],
        on=["cell_id", "mission_id"],
        how="left",
        validate="one_to_one",
    )
    return CellPipelineResult(
        cell_id=str(cell_id),
        telemetry=telemetry,
        cycle_index=cycle_index,
        capacity_tests=health.capacity_tests,
        phase_instances=phase_instances,
        phase_features=phase_features,
        phase_targets=phase_targets,
        mission_features=mission_features,
        mission_targets=mission_targets,
    )


def feature_names_from_dictionary(
    feature_dictionary: pd.DataFrame,
    *,
    level: str,
    available_columns: Iterable[str],
) -> tuple[str, ...]:
    required = {"feature", "level"}
    missing = sorted(required - set(feature_dictionary.columns))
    if missing:
        raise ValueError(f"feature dictionary is missing columns: {missing}")
    available = set(str(name) for name in available_columns)
    selected = feature_dictionary.loc[
        feature_dictionary["level"].astype(str).str.lower().eq(level.lower()), "feature"
    ].astype(str)
    names = tuple(name for name in selected if name in available)
    absent = [name for name in selected if name not in available]
    if absent:
        raise ValueError(f"{level} dictionary features are absent from the table: {absent}")
    if not names:
        raise ValueError(f"no {level} predictors were selected")
    return names


def build_sequence_artifacts(
    phase_features: pd.DataFrame,
    phase_targets: pd.DataFrame,
    mission_features: pd.DataFrame,
    mission_targets: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    *,
    config: PipelineConfig,
) -> dict[str, SequenceBatch]:
    phase_keys = ["cell_id", "mission_id", "seg_id"]
    mission_keys = ["cell_id", "mission_id"]
    phase_table = phase_features.merge(
        phase_targets[phase_keys + ["dis_dSOC"]],
        on=phase_keys,
        how="left",
        validate="one_to_one",
    )
    mission_table = mission_features.merge(
        mission_targets,
        on=mission_keys,
        how="left",
        validate="one_to_one",
    )
    phase_names = feature_names_from_dictionary(
        feature_dictionary, level="phase", available_columns=phase_features.columns
    )
    mission_names = feature_names_from_dictionary(
        feature_dictionary, level="mission", available_columns=mission_features.columns
    )
    windows = config.windows
    batches: dict[str, SequenceBatch] = {
        "soc": build_soc_sequence_batch(
            phase_table,
            phase_names,
            max_steps=windows.soc_max_steps,
            minimum_steps=windows.minimum_steps,
            stride=windows.stride,
        )
    }
    health_cells = set(config.targets.excluded_health_cells)
    health_table = mission_table.loc[
        ~mission_table["cell_id"].astype(str).isin(health_cells)
    ].copy()
    batches["soh"] = build_soh_sequence_batch(
        health_table,
        mission_names,
        horizon_missions=config.targets.soh_horizon_missions,
        max_steps=windows.mission_max_steps,
        minimum_steps=windows.minimum_steps,
        stride=windows.stride,
    )
    for threshold in config.targets.rul_thresholds_pct:
        token = threshold_token(threshold)
        batches[f"rul_{token}"] = build_rul_sequence_batch(
            health_table,
            mission_names,
            threshold_pct=threshold,
            max_steps=windows.mission_max_steps,
            minimum_steps=windows.minimum_steps,
            stride=windows.stride,
        )
    return batches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_pipeline(
    raw_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    feature_dictionary_path: str | Path,
    config: PipelineConfig,
    write_cleaned_telemetry: bool = False,
    continue_on_error: bool = False,
) -> dict[str, object]:
    """Process raw cell files and write auditable combined artifacts."""

    from .preprocessing import cell_id_from_path

    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"output directory must be empty to prevent stale artifact mixing: {output}"
        )
    paths = sorted((Path(path).expanduser().resolve() for path in raw_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("no raw cell files were supplied")
    cell_ids = [cell_id_from_path(path) for path in paths]
    duplicates = sorted({cell for cell in cell_ids if cell_ids.count(cell) > 1})
    if duplicates:
        raise ValueError(f"multiple raw files resolve to the same cell ID: {duplicates}")
    dictionary_path = Path(feature_dictionary_path).expanduser().resolve()
    dictionary = pd.read_csv(dictionary_path)

    table_attributes = {
        "cycle_index.csv": "cycle_index",
        "capacity_tests.csv": "capacity_tests",
        "phase_features.csv": "phase_features",
        "phase_targets.csv": "phase_targets",
        "mission_features.csv": "mission_features",
        "mission_targets.csv": "mission_targets",
    }
    table_frames: dict[str, list[pd.DataFrame]] = {
        filename: [] for filename in table_attributes
    }
    processed_cells: list[str] = []
    failures: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for path, cell_id in zip(paths, cell_ids):
        input_hashes[path.name] = _sha256(path)
        try:
            raw = pd.read_csv(path)
            result = process_cell_frame(raw, cell_id=cell_id, config=config)
            if write_cleaned_telemetry:
                _write_csv(result.telemetry, output / "cleaned" / f"{cell_id}_telemetry.csv")
            for filename, attribute in table_attributes.items():
                table_frames[filename].append(getattr(result, attribute))
            processed_cells.append(result.cell_id)
            del result, raw
        except Exception as error:  # recorded with cell/path in the manifest
            failures[path.name] = f"{type(error).__name__}: {error}"
            if not continue_on_error:
                raise
    if not processed_cells:
        raise RuntimeError("the pipeline produced no cell artifacts")

    combined = {
        filename: pd.concat(frames, ignore_index=True)
        for filename, frames in table_frames.items()
    }
    for filename, frame in combined.items():
        _write_csv(frame, output / filename)
    dictionary_output = output / "feature_dictionary.csv"
    if dictionary_path != dictionary_output:
        shutil.copyfile(dictionary_path, dictionary_output)

    batches = build_sequence_artifacts(
        combined["phase_features.csv"],
        combined["phase_targets.csv"],
        combined["mission_features.csv"],
        combined["mission_targets.csv"],
        dictionary,
        config=config,
    )
    fold_frames: list[pd.DataFrame] = []
    for name, batch in batches.items():
        batch.save_npz(output / "samples" / f"{name}.npz")
        if len(np.unique(batch.cell_ids)) >= 3:
            fold = build_loco_manifest(batch.cell_ids)
            fold.insert(0, "task", name)
            fold_frames.append(fold)
    if fold_frames:
        _write_csv(pd.concat(fold_frames, ignore_index=True), output / "loco_folds.csv")

    config_payload = config.to_dict()
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest: dict[str, object] = {
        "pipeline_schema_version": config.schema_version,
        "configuration_sha256": config_sha,
        "configuration": config_payload,
        "inputs_sha256": input_hashes,
        "feature_dictionary": {
            "filename": dictionary_path.name,
            "sha256": _sha256(dictionary_path),
        },
        "processed_cells": processed_cells,
        "failures": failures,
        "counts": {
            "cycle_instances": int(len(combined["cycle_index.csv"])),
            "capacity_tests": int(len(combined["capacity_tests.csv"])),
            "missions": int(len(combined["mission_features.csv"])),
            "phase_segments": int(len(combined["phase_features.csv"])),
            "sequence_samples": {name: int(len(batch.targets)) for name, batch in batches.items()},
        },
        "outputs_sha256": {
            str(path.relative_to(output)): _sha256(path) for path in output_files
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
