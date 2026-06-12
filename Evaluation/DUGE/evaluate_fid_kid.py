from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute DUGE FID and KID with cleanfid.")
    parser.add_argument("--input_dir", required=True, help="Directory with generated images.")
    parser.add_argument("--reference_dir", required=True, help="Directory with reference images.")
    parser.add_argument("--output_dir", required=True, help="Directory where metrics are written.")
    parser.add_argument("--device", default=None, help="Optional cleanfid device override, such as cuda or cpu.")
    return parser


def parse_cli_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _check_image_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {path}")

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not any(child.is_file() and child.suffix.lower() in valid_ext for child in path.rglob("*")):
        raise ValueError(f"{name} directory contains no supported images: {path}")


def evaluate_fid_kid(args: argparse.Namespace) -> dict[str, float | str]:
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    _check_image_dir(input_dir, "Input")
    _check_image_dir(reference_dir, "Reference")

    try:
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError("cleanfid is required for FID/KID. Install it with `pip install clean-fid`.") from exc

    cleanfid_kwargs = {}
    if args.device is not None:
        cleanfid_kwargs["device"] = args.device

    kid_score = fid.compute_kid(str(input_dir), str(reference_dir), **cleanfid_kwargs)
    fid_score = fid.compute_fid(str(input_dir), str(reference_dir), **cleanfid_kwargs)

    result = {
        "input_dir": str(input_dir),
        "reference_dir": str(reference_dir),
        "kid": float(kid_score),
        "fid": float(fid_score),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "fid_kid.json", "w") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    evaluate_fid_kid(parse_cli_args())


if __name__ == "__main__":
    main()
