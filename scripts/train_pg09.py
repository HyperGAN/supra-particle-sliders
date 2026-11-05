#!/usr/bin/env python3
"""Train the canonical PG0.9.5 Supra session through the shared dispatcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.train import main

if __name__ == "__main__":
    raise SystemExit(main())
