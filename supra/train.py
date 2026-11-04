"""Canonical installed training entry using the shared isolated dispatcher."""
from __future__ import annotations


def main(argv=None):
    import sys
    from particle_sliders.session_cli import main as dispatch
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--profile" not in argv:
        from importlib.metadata import entry_points
        selected = "supra" if entry_points(group="particle_sliders.training_profiles", name="supra") else "supra_hndl"
        argv = ["--profile", selected, *argv]
    return dispatch(["train", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
