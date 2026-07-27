#!/usr/bin/env python3
"""Fetch pretrained weights for the ML detector backends into models/.

Stdlib only -- does not require turret or any ML package to download files;
`--verify` additionally loads whatever's on disk through the real detector
classes (best-effort: prints a clear message if that runtime isn't
installed yet, rather than failing the whole run).

    python scripts/download_models.py                 # tflite set (default, recommended)
    python scripts/download_models.py --set opencv_dnn
    python scripts/download_models.py --set all
    python scripts/download_models.py --list
    python scripts/download_models.py --verify
    python scripts/download_models.py --force          # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class Asset:
    url: str
    filename: str
    sha256: str
    size: int


ASSETS: dict[str, list[Asset]] = {
    "tflite": [
        Asset(
            url="https://raw.githubusercontent.com/google-coral/test_data/master/"
            "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
            filename="ssd_mobilenet_v2_coco_quant_postprocess.tflite",
            sha256="42fb3d70ffb7bb37dd518f730f7be784b831c2078f30c497d0019cc2e987fa26",
            size=6_220_797,
        ),
        Asset(
            url="https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt",
            filename="coco_labels.txt",
            sha256="dc183f003fc753c4c43fae6fdf7f387559449573f13fa32e517fb7453fd380f1",
            size=661,
        ),
    ],
    "opencv_dnn": [
        Asset(
            url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
            "object_detection_nanodet/object_detection_nanodet_2022nov_int8bq.onnx",
            filename="object_detection_nanodet_2022nov_int8bq.onnx",
            sha256="8a2c877cc6f09e7dfac7a9066e33ee5ae68de530b3b994f6ee9125cff6e34d3f",
            size=1_123_958,
        ),
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(asset: Asset, *, force: bool) -> None:
    dest = MODELS_DIR / asset.filename
    if dest.exists() and not force:
        if _sha256(dest) == asset.sha256:
            print(f"  ok      {asset.filename} (already present, checksum matches)")
            return
        print(f"  stale   {asset.filename} (checksum mismatch, re-downloading)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  fetch   {asset.filename} <- {asset.url}")
    with urllib.request.urlopen(asset.url) as resp, part.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

    actual = _sha256(part)
    if actual != asset.sha256:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch for {asset.filename}:\n"
            f"  expected {asset.sha256}\n"
            f"  got      {actual}\n"
            "The upstream file may have changed; not installing it. "
            "Please check the URL / open an issue before retrying."
        )
    part.replace(dest)
    print(f"  done    {asset.filename} ({dest.stat().st_size} bytes)")


def _sets_for(name: str) -> list[str]:
    if name == "all":
        return list(ASSETS)
    return [name]


def cmd_list() -> None:
    for set_name, assets in ASSETS.items():
        print(f"{set_name}:")
        for a in assets:
            print(f"  {a.filename:45s} {a.size:>10,} B  {a.url}")


def cmd_download(set_name: str, *, force: bool) -> None:
    for name in _sets_for(set_name):
        print(f"[{name}]")
        for asset in ASSETS[name]:
            _download_one(asset, force=force)


def _print_describe(name: str, info: dict[str, object]) -> None:
    print(f"{name}:")
    for k, v in info.items():
        print(f"  {k}: {v}")


def cmd_verify() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

    tflite_model = MODELS_DIR / ASSETS["tflite"][0].filename
    tflite_labels = MODELS_DIR / ASSETS["tflite"][1].filename
    if not tflite_model.exists():
        print("tflite: not downloaded -- run: python scripts/download_models.py --set tflite")
    else:
        try:
            from turret.config import MLDetectionConfig
            from turret.vision.tflite_detector import TFLiteDetector

            det = TFLiteDetector(
                MLDetectionConfig(model_path=str(tflite_model), labels_path=str(tflite_labels))
            )
            _print_describe("tflite", det.describe())
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostic, report and continue
            print(f"tflite: FAILED to load -- {exc}")

    dnn_model = MODELS_DIR / ASSETS["opencv_dnn"][0].filename
    if not dnn_model.exists():
        print("opencv_dnn: not downloaded -- run: python scripts/download_models.py --set opencv_dnn")
    else:
        try:
            from turret.config import MLDetectionConfig
            from turret.vision.dnn_detector import DnnDetector

            det = DnnDetector(MLDetectionConfig(model_path=str(dnn_model)))
            _print_describe("opencv_dnn", det.describe())
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostic, report and continue
            print(f"opencv_dnn: FAILED to load -- {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--set", default="tflite", choices=["tflite", "opencv_dnn", "all"], help="asset set to fetch"
    )
    parser.add_argument("--list", action="store_true", help="list assets and exit, no download")
    parser.add_argument("--verify", action="store_true", help="load what's on disk and report; no download")
    parser.add_argument("--force", action="store_true", help="re-download even if checksum already matches")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return
    if args.verify:
        cmd_verify()
        return
    cmd_download(args.set, force=args.force)


if __name__ == "__main__":
    main()
