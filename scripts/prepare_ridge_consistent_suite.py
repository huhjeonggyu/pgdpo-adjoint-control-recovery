#!/usr/bin/env python3
"""Create an isolated paper-suite manifest for covariance/loading-affected jobs.

The generated suite reuses the same external policy checkpoints but reruns the
reference rollout, adjoint harvesting, and recovery under the corrected model.
It does not overwrite runs_paper_full_shift.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml


def is_affected(mapping: dict) -> bool:
    model = dict(mapping.get("model", {}))
    name = str(model.get("name", "")).lower()
    if name == "merton":
        mode = str(model.get("market_mode", "")).lower()
        return mode in {"legacy_cap", "cap", "paper_cap"}
    if name == "affine_factor":
        return not bool(model.get("exact_one_factor", False))
    return False


def normalize_path(value, base: Path):
    if not value:
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def _dependencies(job) -> list[str]:
    values: list[str] = []
    if job.reuse_from:
        values.append(job.reuse_from)
    compare_to = (job.tags or {}).get("compare_to")
    if compare_to:
        values.append(str(compare_to))
    return values


def affected_job_names(suite) -> set[str]:
    """Return the direct correction set plus all changed dependencies/derivatives."""
    jobs_by_name = {job.name: job for job in suite.jobs}
    selected = {job.name for job in suite.jobs if is_affected(job.mapping)}
    changed = True
    while changed:
        changed = False
        for job in suite.jobs:
            deps = _dependencies(job)
            for dep in deps:
                if dep not in jobs_by_name:
                    raise KeyError(f"{job.name} depends on unknown job {dep}")
            if job.name in selected:
                for dep in deps:
                    if dep not in selected:
                        selected.add(dep)
                        changed = True
            elif any(dep in selected for dep in deps):
                selected.add(job.name)
                changed = True
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="paper/paper_suite.yaml")
    parser.add_argument("--output", default="paper/ridge_fix/paper_suite.yaml")
    parser.add_argument("--runs-root", default="runs_ridge_consistent")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    sys.path.insert(0, str(project_root / "src"))
    from mf_revision.experiments.suite import PaperSuite

    source_manifest = Path(args.manifest).expanduser().resolve()
    suite = PaperSuite(source_manifest)
    selected = affected_job_names(suite)

    missing_assets = sorted(
        {
            alias
            for job in suite.jobs
            if job.name in selected
            for alias in suite.missing_assets(job)
        }
    )
    if missing_assets:
        aliases = ", ".join(missing_assets)
        raise SystemExit(
            "The corrected paper rerun requires the original immutable assets. "
            f"Missing or unavailable aliases: {aliases}. "
            "Create paper/legacy_catalog.json with `mf-revision discover-legacy` "
            "before preparing the rerun suite."
        )

    output_manifest = Path(args.output).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    generated_source_base = source_manifest.parent / "generated_configs"
    runs_root = Path(args.runs_root).expanduser()
    if not runs_root.is_absolute():
        runs_root = (project_root / runs_root).resolve()

    rows = []
    for job in suite.jobs:
        if job.name not in selected:
            continue
        mapping = copy.deepcopy(job.mapping)
        mapping.pop("name", None)
        mapping.pop("output_root", None)

        policy = mapping.get("policy")
        if isinstance(policy, dict) and policy.get("checkpoint"):
            policy["checkpoint"] = normalize_path(
                policy["checkpoint"], generated_source_base
            )
        model = mapping.get("model")
        if isinstance(model, dict) and model.get("market_snapshot"):
            model["market_snapshot"] = normalize_path(
                model["market_snapshot"], generated_source_base
            )

        row = {
            "name": job.name,
            "group": ["ridge_fix"],
            "stage": job.stage,
            "overrides": mapping,
        }
        if job.reuse_from:
            row["reuse_from"] = job.reuse_from
        if job.tags:
            row["tags"] = copy.deepcopy(job.tags)
        rows.append(row)

    payload = {
        "output_root": str(runs_root),
        "jobs": rows,
    }
    output_manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print(f"source manifest : {source_manifest}")
    print(f"output manifest : {output_manifest}")
    print(f"output root     : {runs_root}")
    print(f"selected jobs   : {len(rows)}")
    for row in rows:
        print(f"  - {row['name']} ({row['stage']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
