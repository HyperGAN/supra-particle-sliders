"""Resolve the reviewed installed shared package; never use vendor code."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

LOCK = Path(__file__).resolve().parents[1] / "shared.lock.json"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_root(root=None):
    value = root or os.environ.get("PARTICLE_SLIDERS_ROOT")
    if value:
        root = Path(value).resolve()
        candidates = (root / "packages/particle-sliders-core/src/particle_sliders",
                      root / "particle_sliders", root)
        root = next((p for p in candidates if (p / "pg09.py").is_file()), root)
    else:
        spec = importlib.util.find_spec("particle_sliders")
        if spec is None or not spec.origin:
            raise RuntimeError("Install the pinned particle-sliders-core package; see docs/pg09.md")
        root = Path(spec.origin).resolve().parent
    lock = json.loads(LOCK.read_text())
    for relative, expected in lock["sources"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha(path) != expected:
            raise RuntimeError("Shared particle-sliders source differs: " + relative)
    return root


def shared_module(name, root=None):
    """Use one pinned package, refusing an already-imported shadow package."""
    root = shared_root(root)
    if not name.startswith("particle_sliders."):
        raise ValueError("Only the public shared package namespace is supported")
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "particle_sliders" or module_name.startswith("particle_sliders."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise RuntimeError("Shared import is shadowed: " + module_name)
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Shared module resolved outside reviewed checkout: " + name)
    return module
