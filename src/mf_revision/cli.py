"""Command-line entry point."""
from __future__ import annotations

import argparse
import json

from mf_revision.config import load_config
from mf_revision.experiments import ExperimentRunner
from mf_revision.experiments.collect import collect_paper_results
from mf_revision.experiments.legacy_catalog import write_legacy_catalog
from mf_revision.experiments.suite import PaperSuite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mf-revision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "harvest", "recover", "pipeline", "inspect"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--output-root", default=None)
        if command == "harvest":
            subparser.add_argument("--checkpoint", default=None)
        if command == "recover":
            subparser.add_argument("--adjoints", default=None)
            subparser.add_argument("--holdout-adjoints", default=None)

    discover = subparsers.add_parser("discover-legacy")
    discover.add_argument("--root", required=True)
    discover.add_argument("--output", default="paper/legacy_catalog.json")
    discover.add_argument("--strict", action="store_true")

    suite = subparsers.add_parser("suite")
    suite.add_argument("--manifest", default="paper/paper_suite.yaml")
    suite.add_argument("--group", action="append", default=[])
    suite.add_argument("--force", action="store_true")
    suite.add_argument("--keep-going", action="store_true")
    suite.add_argument("--plan", action="store_true")
    suite.add_argument("--materialize", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", default="paper/paper_suite.yaml")
    collect.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "discover-legacy":
        values = write_legacy_catalog(
            arguments.root, arguments.output, strict=arguments.strict
        )
        print(json.dumps(values, indent=2, sort_keys=True))
        return
    if arguments.command == "suite":
        suite = PaperSuite(arguments.manifest)
        if arguments.plan:
            print(json.dumps(suite.plan(arguments.group), indent=2))
            return
        if arguments.materialize:
            for path in suite.materialize(arguments.group):
                print(path)
            return
        suite.run(
            groups=arguments.group,
            force=arguments.force,
            stop_on_error=not arguments.keep_going,
        )
        return
    if arguments.command == "collect":
        values = collect_paper_results(
            arguments.manifest, output_dir=arguments.output
        )
        print(json.dumps(values, indent=2, sort_keys=True))
        return

    runner = ExperimentRunner(
        load_config(arguments.config), output_override=arguments.output_root
    )
    if arguments.command == "train":
        runner.train()
    elif arguments.command == "harvest":
        runner.harvest(arguments.checkpoint)
        runner.harvest_holdout(arguments.checkpoint)
    elif arguments.command == "recover":
        runner.recover(
            arguments.adjoints, holdout_adjoint_path=arguments.holdout_adjoints
        )
    elif arguments.command == "pipeline":
        runner.pipeline()
    elif arguments.command == "inspect":
        runner.inspect()


if __name__ == "__main__":
    main()
