"""Build the corrected fixed-40 manifest and frozen June 24 SEED skill bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed.june24_skill_summary import build_fixed_manifest, materialize_skill_bank


def parse_task_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    task_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not task_ids:
        raise argparse.ArgumentTypeError("--task-ids must select at least one task")
    return task_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument("--rejected-75-manifest", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_manifest = build_fixed_manifest(
        args.fixed_csv.expanduser().resolve(),
        args.rejected_75_manifest.expanduser().resolve(),
    )
    args.fixed_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.fixed_manifest.write_text(
        json.dumps(fixed_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    bank = materialize_skill_bank(
        source_dirs=[path.expanduser().resolve() for path in args.source_dir],
        fixed_manifest=args.fixed_manifest.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        selected_task_ids=parse_task_ids(args.task_ids),
    )
    print(
        json.dumps(
            {
                "fixed_task_count": fixed_manifest["task_count"],
                "bank_task_count": bank["task_count"],
                "task_ids": bank["task_ids"],
                "fixed_manifest": str(args.fixed_manifest.expanduser().resolve()),
                "skill_bank": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
