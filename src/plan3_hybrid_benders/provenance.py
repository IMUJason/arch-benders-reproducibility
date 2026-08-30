from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_literature_manifest(reference_dir: Path) -> list[dict[str, object]]:
    manifest = []
    for pdf_path in sorted(reference_dir.glob("*.PDF")):
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        manifest.append(
            {
                "filename": pdf_path.name,
                "relative_path": str(pdf_path),
                "sha256": sha256_file(pdf_path),
                "page_count": len(reader.pages),
                "title": metadata.get("/Title"),
                "author": metadata.get("/Author"),
            }
        )
    return manifest


def build_file_manifest(root: Path, pattern: str) -> list[dict[str, object]]:
    manifest = []
    for file_path in sorted(root.glob(pattern)):
        if file_path.is_file():
            manifest.append(
                {
                    "filename": file_path.name,
                    "relative_path": str(file_path),
                    "sha256": sha256_file(file_path),
                    "size_bytes": file_path.stat().st_size,
                }
            )
    return manifest


def pip_freeze(python_executable: str) -> list[str]:
    try:
        result = subprocess.run(
            [python_executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        packages = []
        for distribution in sorted(
            importlib.metadata.distributions(),
            key=lambda item: (item.metadata.get("Name") or "").lower(),
        ):
            name = distribution.metadata.get("Name")
            if not name:
                continue
            packages.append(f"{name}=={distribution.version}")
        return packages


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_provenance_bundle(
    *,
    package_root: Path,
    plan_root: Path,
    result_root: Path,
    manifests_root: Path,
) -> dict[str, object]:
    literature_manifest = build_literature_manifest(plan_root / "references")
    raw_result_manifest = build_file_manifest(result_root / "raw", "*.json")
    figure_manifest = build_file_manifest(result_root / "figures", "*")
    data_manifest = build_file_manifest(package_root / "data" / "generated", "*.json")
    environment_manifest = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pip_freeze": pip_freeze(sys.executable),
    }
    bundle = {
        "literature": literature_manifest,
        "data": data_manifest,
        "raw_results": raw_result_manifest,
        "figures": figure_manifest,
        "environment": environment_manifest,
    }
    write_json(manifests_root / "literature_manifest.json", literature_manifest)
    write_json(manifests_root / "environment_manifest.json", environment_manifest)
    write_json(manifests_root / "provenance_manifest.json", bundle)
    return bundle
