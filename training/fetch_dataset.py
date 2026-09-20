"""Download the mask dataset at a pinned revision and verify every file against training/dataset_manifest.json."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = "hmnshudhmn24/face-mask-detection"
REVISION = "681b307920cad305bdbc042254d933dbb18d1d28"
MANIFEST = Path(__file__).with_name("dataset_manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] in {"images", "annotations"}
    }


def verify(root: Path) -> None:
    expected = json.loads(MANIFEST.read_text())["files"]
    bad = [name for name, digest in expected.items() if not (root / name).is_file() or sha256(root / name) != digest]
    if bad:
        raise SystemExit(f"{len(bad)} dataset files missing or modified, e.g. {bad[:3]}")
    print(f"dataset verified: {len(expected)} files match the manifest")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/maskdata")
    if not (root / "images").is_dir():
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION, local_dir=str(root), max_workers=8)
    if "--write-manifest" in sys.argv:
        MANIFEST.write_text(
            json.dumps(
                {
                    "repo": REPO,
                    "revision": REVISION,
                    "license": "CC0-1.0 (Kaggle andrewmvd/face-mask-detection)",
                    "files": build_manifest(root),
                },
                indent=0,
            )
        )
        print(f"wrote {MANIFEST}")
    else:
        verify(root)


if __name__ == "__main__":
    main()
