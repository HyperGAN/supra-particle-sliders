#!/usr/bin/env python3
"""Build the separate HNDL candidate wheel without changing the stable metadata."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]



def restore_pre_lr_runtime(root, destination):
    """Keep the closed HNDL cohort's borrowed runtime bytes unchanged."""
    import hashlib
    import json
    archive = root / 'cohorts/pre-lr-hndl'
    manifest = json.loads((archive / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        data = (archive / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('archived HNDL runtime checksum differs: ' + relative)
        (destination / relative).write_bytes(data)
    (destination / 'supra/learning_rates.py').unlink(missing_ok=True)
    # Archive ships in the sdist so its profile builder remains reproducible.
    target = destination / 'cohorts/pre-lr-hndl'
    if not target.exists():
        shutil.copytree(archive, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/hndl-wheels")
    parser.add_argument("--sdist", action="store_true", help="also build the standalone source distribution (requires build)")
    options = parser.parse_args(argv)
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supra-hndl-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(ROOT / "supra", stage / "supra",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "profiles/hndl/pyproject.toml", stage / "pyproject.toml")
        shutil.copy2(ROOT / "profiles/hndl/README.md", stage / "README.md")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        restore_pre_lr_runtime(ROOT, stage)
        (stage / 'MANIFEST.in').write_text('recursive-include cohorts/pre-lr-hndl *.py *.json\n')
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(output), str(stage)], check=True)
        if options.sdist:
            subprocess.run([sys.executable, "-m", "build", "--sdist", "--no-isolation",
                            "--outdir", str(output), str(stage)], check=True)


if __name__ == "__main__":
    main()
