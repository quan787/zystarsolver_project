from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


MODELS = ("10-35", "30-100")
DEFAULT_MODEL_DIR = Path.cwd() / "models"
GITHUB_REPO = "quan787/zystarsolver_project"
ASSET_NAMES = {
    "10-35": "zystarsolver-model-10-35.zip",
    "30-100": "zystarsolver-model-30-100.zip",
}
OVERLAP_ASSET_NAME = "overlap_10.json"


def _download_file(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url) as response, dst.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _latest_release_assets(repo: str = GITHUB_REPO) -> dict[str, str]:
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    with urllib.request.urlopen(api_url) as response:
        data = json.loads(response.read().decode("utf-8"))
    assets = {}
    for asset in data.get("assets", []):
        name = asset.get("name")
        download_url = asset.get("browser_download_url")
        if name and download_url:
            assets[name] = download_url
    return assets


def _validate_args(args: argparse.Namespace) -> None:
    if args.model is None and args.url:
        raise SystemExit("--url requires --model")
    if args.model is None and args.archive:
        raise SystemExit("--archive requires --model")
    if args.url and args.archive:
        raise SystemExit("--url and --archive cannot be used together")


def _extract_archive(
    archive_path: Path,
    model: str,
    model_root: Path,
    *,
    require_overlap: bool = True,
) -> Path:
    target_dir = model_root / model
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_path)

        source_dir = tmp_path / model
        if not source_dir.exists():
            source_dir = tmp_path

        for path in source_dir.rglob("*"):
            if path.is_file():
                relative = path.relative_to(source_dir)
                destination = target_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

        overlap_candidates = [
            tmp_path / "overlap_10.json",
            source_dir / "overlap_10.json",
            target_dir / "overlap_10.json",
        ]
        for candidate in overlap_candidates:
            if candidate.exists():
                shutil.copy2(candidate, model_root / "overlap_10.json")
                break

    _validate_extracted_model(model, model_root, require_overlap=require_overlap)
    return target_dir


def _validate_extracted_model(model: str, model_root: Path, *, require_overlap: bool = True) -> None:
    model_dir = model_root / model
    missing = []
    if not list(model_dir.glob("*.onnx")):
        missing.append("one .onnx file")
    if not (model_dir / "star_id_map.json").exists():
        missing.append("star_id_map.json")
    if require_overlap and not (model_root / "overlap_10.json").exists() and not (model_dir / "overlap_10.json").exists():
        missing.append("overlap_10.json")
    if missing:
        raise SystemExit(
            f"{model} archive layout is invalid; missing {', '.join(missing)} after extraction"
        )


def _download_or_use_archive(model: str, args: argparse.Namespace, model_root: Path) -> Path:
    model_root.mkdir(parents=True, exist_ok=True)
    if args.archive:
        archive_path = Path(args.archive)
        if not archive_path.exists():
            raise SystemExit(f"archive does not exist: {archive_path}")
        return _extract_archive(archive_path, model, model_root)

    if args.url:
        url = args.url
        require_overlap = True
    else:
        assets = _latest_release_assets()
        asset_name = ASSET_NAMES[model]
        try:
            url = assets[asset_name]
        except KeyError as exc:
            raise SystemExit(f"latest release does not contain asset {asset_name}") from exc
        require_overlap = False

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / ASSET_NAMES[model]
        print(f"Downloading {model} from {url}")
        _download_file(url, archive_path)
        target = _extract_archive(archive_path, model, model_root, require_overlap=require_overlap)

    if not require_overlap and not (model_root / "overlap_10.json").exists():
        assets = _latest_release_assets()
        try:
            overlap_url = assets[OVERLAP_ASSET_NAME]
        except KeyError as exc:
            raise SystemExit(f"latest release does not contain asset {OVERLAP_ASSET_NAME}") from exc
        print(f"Downloading {OVERLAP_ASSET_NAME} from {overlap_url}")
        _download_file(overlap_url, model_root / "overlap_10.json")

    _validate_extracted_model(model, model_root)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ZYstarsolver ONNX model artifacts.")
    parser.add_argument("--model", choices=MODELS, help="Model to download. Omit to download both official models.")
    parser.add_argument("--url", help="Custom zip URL. Requires --model.")
    parser.add_argument("--archive", help="Local model zip path. Requires --model.")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory that will contain 10-35/, 30-100/, and overlap_10.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    models = [args.model] if args.model else list(MODELS)
    model_root = Path(args.model_dir)
    for model in models:
        target = _download_or_use_archive(model, args, model_root)
        print(f"{model} model ready: {target}")
        print(f"Try: python examples/local_infer.py --model {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
