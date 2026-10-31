#!/usr/bin/env python3
"""Build the separate HNDL candidate wheel without changing the stable metadata."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/hndl-wheels")
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
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(output), str(stage)], check=True)


if __name__ == "__main__":
    main()
