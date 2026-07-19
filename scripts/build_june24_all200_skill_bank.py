"""Build the audited June 24 all-200 cohort, 160/40 split, and train-only bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed.june24_skill_summary import (
    DEFAULT_SPLIT_SEED,
    build_all200_cohort_manifest,
    build_stratified_split_manifest,
    materialize_all200_training_skill_bank,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-tasks-csv", type=Path, required=True)
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-repaired-task-id",
        action="append",
        default=[],
        help="Explicit historical task-level repair exception; repeat once per authorized task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairwise_csv = args.pairwise_tasks_csv.expanduser().resolve()
    cohort_path = args.cohort_manifest.expanduser().resolve()
    split_path = args.split_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    cohort = build_all200_cohort_manifest(pairwise_csv)
    _write_json(cohort_path, cohort)
    split = build_stratified_split_manifest(
        cohort_path,
        seed=args.seed,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )
    _write_json(split_path, split)
    bank = materialize_all200_training_skill_bank(
        source_dirs=[path.expanduser().resolve() for path in args.source_dir],
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        output_path=output_path,
        allowed_repaired_task_ids=args.allow_repaired_task_id,
    )
    print(
        json.dumps(
            {
                "cohort_task_count": cohort["task_count"],
                "train_task_count": split["train"]["task_count"],
                "validation_task_count": split["validation"]["task_count"],
                "total_updates": split["training_schedule"]["total_updates"],
                "total_rollouts": split["training_schedule"]["total_rollouts"],
                "source_task_count": bank["source_task_count"],
                "repaired_source_overrides": bank["repaired_source_overrides"],
                "cohort_manifest": str(cohort_path),
                "split_manifest": str(split_path),
                "skill_bank": str(output_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
