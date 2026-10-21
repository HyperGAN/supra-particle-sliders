"""Compatibility entry point for the native shared PG0.9 GAN worker."""
from __future__ import annotations


def main(argv=None):
    from scripts.train_pg09 import main as train_shared
    return train_shared(argv)


if __name__ == "__main__":
    main()
