#!/usr/bin/env python3
"""Compatibility alias of train_pg09.py; executes the same native GAN worker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.train import main

if __name__ == "__main__":
    main()
