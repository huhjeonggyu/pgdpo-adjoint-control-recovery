"""Manifest-driven execution of the Section 5 and Appendix revalidation suite."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import copy
import json
import os
import time
import yaml

from mf_revision.config import ExperimentConfig
from mf_revision.runtime import write_json
from .legacy_catalog import load_legacy_catalog
from .runner import ExperimentRunner
from .audits import compare_graph_estimates, run_initialization_audit


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class SuiteJob:
    name: str
    groups: tuple[str, ...]
    stage: str
    mapping: dict[str, Any]
    reuse_from: str | None = None
    tags: dict[str, Any] | None = None
    checkpoint_alias: str | None = None
    market_snapshot_alias: str | None = None


class PaperSuite:
    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Suite manifest root must be a mapping")
        self.payload = _expand(payload)
        self.output_root = Path(
            str(self.payload.get("output_root", "runs_full_shift"))
        )
        if not self.output_root.is_absolute():
            self.output_root = (self.path.parent.parent / self.output_root).resolve()
        catalog_value = self.payload.get("catalog", "paper/legacy_catalog.json")
        catalog_path = Path(str(catalog_value))
        if not catalog_path.is_absolute():
            catalog_path = (self.path.parent.parent / catalog_path).resolve()
        self.catalog_path = catalog_path
        self.catalog = load_legacy_catalog(catalog_path) if catalog_path.exists() else {}
        self.defaults = dict(self.payload.get("defaults", {}))
        self.templates = dict(self.payload.get("templates", {}))
        self.jobs = self._parse_jobs(self.payload.get("jobs", []))

    def _parse_jobs(self, rows: Any) -> list[SuiteJob]:
        if not isinstance(rows, list):
            raise ValueError("manifest.jobs must be a list")
        jobs: list[SuiteJob] = []
        names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Every suite job must be a mapping")
            name = str(row.get("name", ""))
            if not name or name in names:
                raise ValueError(f"Invalid or duplicate suite job name: {name!r}")
            names.add(name)
            template_name = row.get("template")
            mapping = copy.deepcopy(self.defaults)
            if template_name:
                if template_name not in self.templates:
                    raise KeyError(f"Unknown suite template: {template_name}")
                mapping = _deep_merge(mapping, dict(self.templates[template_name]))

            config_value = row.get("config")
            if config_value:
                config_path = Path(str(config_value))
                if not config_path.is_absolute():
                    config_path = (self.path.parent / config_path).resolve()
                config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if not isinstance(config_payload, dict):
                    raise ValueError(f"Suite config must be a mapping: {config_path}")
                mapping = _deep_merge(mapping, config_payload)

            mapping = _deep_merge(mapping, dict(row.get("overrides", {})))
            mapping["name"] = name
            mapping["output_root"] = str(self.output_root)

            checkpoint_alias = (
                None if row.get("checkpoint_alias") is None else str(row["checkpoint_alias"])
            )
            if checkpoint_alias and checkpoint_alias in self.catalog:
                mapping.setdefault("policy", {})["checkpoint"] = self.catalog[checkpoint_alias]

            market_alias = (
                None
                if row.get("market_snapshot_alias") is None
                else str(row["market_snapshot_alias"])
            )
            if market_alias and market_alias in self.catalog:
                mapping.setdefault("model", {})["market_snapshot"] = self.catalog[market_alias]
            groups_raw = row.get("groups", row.get("group", []))
            if isinstance(groups_raw, str):
                groups = (groups_raw,)
            else:
                groups = tuple(str(value) for value in groups_raw)
            jobs.append(
                SuiteJob(
                    name=name,
                    groups=groups,
                    stage=str(row.get("stage", "pipeline")).lower(),
                    mapping=mapping,
                    reuse_from=(
                        None if row.get("reuse_from") is None else str(row["reuse_from"])
                    ),
                    tags=dict(row.get("tags", {})),
                    checkpoint_alias=checkpoint_alias,
                    market_snapshot_alias=market_alias,
                )
            )
        return jobs

    def selected(self, groups: Iterable[str] | None = None) -> list[SuiteJob]:
        """Select groups together with manifest-declared dependencies.

        Barrier sweeps, initialization audits, and graph comparisons reuse
        outputs from earlier jobs.  Including dependencies automatically makes
        commands such as ``--group table13`` and ``--group table15`` safe on a
        fresh output tree while still preserving manifest order.
        """
        requested = {str(group) for group in (groups or []) if str(group)}
        if not requested or "all" in requested:
            return list(self.jobs)
        selected_names = {
            job.name for job in self.jobs if requested.intersection(job.groups)
        }
        jobs_by_name = {job.name: job for job in self.jobs}
        pending = list(selected_names)
        while pending:
            name = pending.pop()
            job = jobs_by_name[name]
            dependencies: list[str] = []
            if job.reuse_from:
                dependencies.append(job.reuse_from)
            compare_to = (job.tags or {}).get("compare_to")
            if compare_to:
                dependencies.append(str(compare_to))
            for dependency in dependencies:
                if dependency not in jobs_by_name:
                    raise KeyError(
                        f"Suite job {name!r} depends on unknown job {dependency!r}"
                    )
                if dependency not in selected_names:
                    selected_names.add(dependency)
                    pending.append(dependency)
        return [job for job in self.jobs if job.name in selected_names]

    def missing_assets(self, job: SuiteJob) -> list[str]:
        """Return aliases that are absent or point to unavailable local files."""
        missing: list[str] = []
        for alias in (job.checkpoint_alias, job.market_snapshot_alias):
            if not alias:
                continue
            value = self.catalog.get(alias)
            if value is None or not Path(value).expanduser().exists():
                missing.append(alias)
        return missing

    def _ensure_assets(self, job: SuiteJob) -> None:
        missing = self.missing_assets(job)
        if not missing:
            return
        aliases = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Missing legacy assets for suite job {job.name!r}: {aliases}. "
            f"Create {self.catalog_path} with `mf-revision discover-legacy`."
        )

    def config_for(self, job: SuiteJob) -> ExperimentConfig:
        self._ensure_assets(job)
        source = self.path.parent / "generated_configs" / f"{job.name}.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            yaml.safe_dump(job.mapping, sort_keys=False), encoding="utf-8"
        )
        return ExperimentConfig.from_mapping(job.mapping, source_path=str(source))

    def plan(self, groups: Iterable[str] | None = None) -> list[dict[str, Any]]:
        rows = []
        for job in self.selected(groups):
            rows.append(
                {
                    "name": job.name,
                    "groups": list(job.groups),
                    "stage": job.stage,
                    "reuse_from": job.reuse_from,
                    "output": str(self.output_root / job.name),
                    "missing_assets": self.missing_assets(job),
                }
            )
        return rows

    def materialize(self, groups: Iterable[str] | None = None) -> list[Path]:
        paths: list[Path] = []
        for job in self.selected(groups):
            config = self.config_for(job)
            assert config.source_path is not None
            paths.append(Path(config.source_path))
        return paths

    def run(
        self,
        *,
        groups: Iterable[str] | None = None,
        force: bool = False,
        stop_on_error: bool = True,
    ) -> dict[str, Any]:
        selected = self.selected(groups)
        status: dict[str, Any] = {
            "manifest": str(self.path),
            "output_root": str(self.output_root),
            "started_at": time.time(),
            "jobs": [],
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        final_names = {
            "pipeline": "recovery_summary.json",
            "recover": "recovery_summary.json",
            "harvest": "adjoint_summary.json",
            "init_audit": "initialization_audit.json",
            "graph_compare": "graph_ablation.json",
        }
        for index, job in enumerate(selected, start=1):
            output = self.output_root / job.name
            if job.stage not in final_names:
                raise ValueError(f"Unknown suite stage: {job.stage}")
            final_file = output / final_names[job.stage]
            row: dict[str, Any] = {
                "name": job.name,
                "stage": job.stage,
                "groups": list(job.groups),
                "output": str(output),
                "started_at": time.time(),
            }
            print(f"[suite {index}/{len(selected)}] {job.name} ({job.stage})")
            if final_file.exists() and not force:
                row["status"] = "skipped_complete"
                row["finished_at"] = time.time()
                status["jobs"].append(row)
                continue
            try:
                if job.stage == "graph_compare":
                    if not job.reuse_from:
                        raise ValueError("graph_compare requires reuse_from for the OL job")
                    compare_to = str((job.tags or {}).get("compare_to", ""))
                    if not compare_to:
                        raise ValueError("graph_compare requires tags.compare_to")
                    output.mkdir(parents=True, exist_ok=True)
                    compare_graph_estimates(
                        self.output_root / job.reuse_from / "adjoints.pt",
                        self.output_root / compare_to / "adjoints.pt",
                        output / "graph_ablation.json",
                    )
                else:
                    config = self.config_for(job)
                    runner = ExperimentRunner(config)
                    if job.stage == "pipeline":
                        runner.pipeline()
                    elif job.stage == "harvest":
                        runner.harvest()
                        runner.harvest_holdout()
                    elif job.stage == "recover":
                        if not job.reuse_from:
                            runner.recover()
                        else:
                            parent = self.output_root / job.reuse_from
                            holdout_path = parent / "adjoints_holdout.pt"
                            runner.recover(
                                parent / "adjoints.pt",
                                holdout_adjoint_path=(
                                    holdout_path if holdout_path.exists() else None
                                ),
                            )
                    elif job.stage == "init_audit":
                        if not job.reuse_from:
                            raise ValueError("init_audit requires reuse_from")
                        parent = self.output_root / job.reuse_from
                        run_initialization_audit(
                            runner.model,
                            parent / "adjoints.pt",
                            parent / "recovery.pt",
                            output,
                            states=(job.tags or {}).get("audit_states"),
                            iterations=int(
                                (job.tags or {}).get("audit_iterations", 2000)
                            ),
                            seed=int((job.tags or {}).get("audit_seed", 12345)),
                        )
                if job.tags:
                    write_json(output / "paper_tags.json", job.tags)
                row["status"] = "complete"
            except Exception as error:  # recorded before optional re-raise
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {error}"
                status["jobs"].append(row)
                status["finished_at"] = time.time()
                write_json(self.output_root / "suite_status.json", status)
                if stop_on_error:
                    raise
                continue
            row["finished_at"] = time.time()
            row["elapsed_seconds"] = row["finished_at"] - row["started_at"]
            status["jobs"].append(row)
            write_json(self.output_root / "suite_status.json", status)
        status["finished_at"] = time.time()
        status["elapsed_seconds"] = status["finished_at"] - status["started_at"]
        write_json(self.output_root / "suite_status.json", status)
        return status
