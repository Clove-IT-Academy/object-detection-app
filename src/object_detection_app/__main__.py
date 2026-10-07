"""CLI entry point. Help is side-effect free; all artifact workflows are explicit."""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from .contracts import AppError
from .paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local webcam object detection and model workflows")
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("run", "benchmark", "verify-export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=ROOT / "configs/app.yaml")
        cmd.add_argument("--bundle", type=lambda s: Path(s).resolve())
        cmd.add_argument("--device")
        cmd.add_argument("--camera", type=int)
        cmd.add_argument("--confidence", type=float)
        cmd.add_argument("--input-size", type=int)
        if name == "run":
            cmd.add_argument("--max-frames", type=int, help="Optional bounded webcam smoke run")
        else:
            cmd.add_argument("--images", type=Path, nargs="+", required=True)
            cmd.add_argument("--output", type=Path, required=True)
        if name == "benchmark":
            cmd.add_argument("--warmup", type=int, default=3)
            cmd.add_argument("--iterations", type=int, default=20)
        if name == "verify-export":
            cmd.add_argument("--export-bundle", type=Path, required=True)
            cmd.add_argument("--score-tolerance", type=float, default=0.02)
            cmd.add_argument("--box-tolerance", type=float, default=2.0)
    setup = sub.add_parser("setup-model", help="Explicitly download official YOLO26n baseline")
    setup.add_argument("--output", type=Path, default=ROOT / "models/yolo26n")
    package = sub.add_parser("package-model", help="Bundle a trusted local model and provenance")
    package.add_argument("--artifact", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--source", required=True, help="Documented acquisition/training source")
    package.add_argument("--input-size", type=int, default=640)
    inspect = sub.add_parser("inspect-model")
    inspect.add_argument("--bundle", type=Path, default=ROOT / "models/yolo26n/bundle.json")
    for name in ("validate-data", "train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", type=Path, required=True)
        cmd.add_argument("--groups", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name != "validate-data":
            cmd.add_argument("--bundle", type=Path, required=True)
        if name == "train":
            cmd.add_argument("--train-config", type=Path, default=ROOT / "configs/train.yaml")
        if name == "evaluate":
            cmd.add_argument("--split", choices=("val", "test"), default="val")
            cmd.add_argument("--device", default="cpu")
            cmd.add_argument(
                "--final-test",
                action="store_true",
                help="Confirm frozen candidate before exposing the test split",
            )
    export = sub.add_parser("export")
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return cli


def execute(args):
    # Heavy libraries are imported after parsing, so --help needs no camera/model work.
    from . import workflows
    from .config import load_config
    from .detector import read_bundle

    if args.command in ("run", "benchmark", "verify-export"):
        config = load_config(
            args.config,
            {
                k: getattr(args, k)
                for k in (
                    "bundle",
                    "device",
                    "camera",
                    "confidence",
                    "input_size",
                )
            },
        )
        if args.command == "run":
            from .controller import run

            if args.max_frames is not None and args.max_frames < 1:
                raise AppError("max-frames must be positive.")
            return run(config, max_frames=args.max_frames)
        if args.command == "benchmark":
            report = workflows.benchmark(config, args.images, args.warmup, args.iterations)
        else:
            _, _, metadata = read_bundle(args.export_bundle)
            exported = replace(
                config,
                bundle=args.export_bundle.resolve(),
                input_size=metadata["input_size"],
                device="cpu",
            )
            if config.input_size != exported.input_size or config.device != "cpu":
                raise AppError("Compare checkpoint/export with matching input size on CPU.")
            report = workflows.verify_export(
                config, exported, args.images, args.score_tolerance, args.box_tolerance
            )
        workflows.write_json(args.output, report)
        if report.get("passed") is False:
            raise AppError("Export parity failed; inspect the written comparison report.")
        return report
    if args.command == "setup-model":
        return {"bundle": str(workflows.setup_model(args.output))}
    if args.command == "package-model":
        # Validate input_size with the same runtime rules before writing metadata.
        from .config import Config

        Config(bundle=Path("unused"), input_size=args.input_size)
        return {
            "bundle": str(
                workflows.package_model(
                    args.artifact,
                    args.output,
                    {"source": args.source},
                    args.input_size,
                )
            )
        }
    if args.command == "inspect-model":
        _, _, metadata = read_bundle(args.bundle)
        return metadata
    if args.command == "validate-data":
        from .data import validate_dataset

        report = validate_dataset(args.data, args.groups)
        workflows.write_json(args.output, report)
        return report
    if args.command == "train":
        return workflows.train(args.bundle, args.data, args.train_config, args.groups, args.output)
    if args.command == "evaluate":
        if args.split == "test" and not args.final_test:
            raise AppError(
                "Freeze model/settings first, then pass --final-test for held-out evaluation."
            )
        report = workflows.evaluate(
            args.bundle, args.data, args.groups, args.split, args.device, args.output
        )
        workflows.write_json(args.output / "metrics.json", report)
        return report
    if args.command == "export":
        return {"bundle": str(workflows.export_model(args.bundle, args.output))}
    raise AppError("Unsupported command.")


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        print(json.dumps(execute(args), indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except AppError as exc:
        logging.error("%s", exc)
        return 1
    except (OSError, ValueError) as exc:
        logging.error(
            "Cannot complete operation (%s). Check local files and settings.", type(exc).__name__
        )
        return 1
    except Exception as exc:
        # Do not print library exceptions that might include frame data or credentials.
        logging.error(
            "Operation failed (%s). Check compatible dependencies and artifact metadata.",
            type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
