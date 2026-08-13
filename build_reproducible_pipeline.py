#!/usr/bin/env python3
"""CLI for the complete reproducible preprocessing/target pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evt_battery.config import PipelineConfig  # noqa: E402
from evt_battery.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cycle/RPT audits, phase and mission tables, explicit SOH/RUL "
            "targets, sliding-window samples, LOCO manifests, and checksums."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing raw VAH*.csv files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty destination directory for derived artifacts",
    )
    parser.add_argument("--pattern", default="VAH*.csv", help="Raw-cell filename glob")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "pipeline.json",
        help="Versioned JSON pipeline configuration",
    )
    parser.add_argument(
        "--feature-dictionary",
        type=Path,
        default=REPOSITORY_ROOT / "feature_dictionary.csv",
        help="Checked-in predictor allow-list",
    )
    parser.add_argument(
        "--write-cleaned-telemetry",
        action="store_true",
        help="Also write large canonical row-level telemetry tables",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a bad cell and continue; default is fail-fast",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"raw-data directory does not exist: {data_dir}")
    paths = sorted(data_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {args.pattern!r} in {data_dir}")
    config = PipelineConfig.from_json(args.config)
    manifest = run_pipeline(
        paths,
        output_dir=args.output_dir,
        feature_dictionary_path=args.feature_dictionary,
        config=config,
        write_cleaned_telemetry=args.write_cleaned_telemetry,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Manifest: {(args.output_dir.expanduser().resolve() / 'manifest.json')}")


if __name__ == "__main__":
    main()
