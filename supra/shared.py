"""Resolve the reviewed external particle-sliders tree; never use vendor code."""
from __future__ import annotations

import hashlib
import importlib
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
    if not value:
        raise RuntimeError("Set PARTICLE_SLIDERS_ROOT to the reviewed external particle-sliders checkout; see docs/pg09.md")
    root = Path(value).resolve()
    lock = json.loads(LOCK.read_text())
    for relative, expected in lock["sources"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha(path) != expected:
            raise RuntimeError("Shared particle-sliders source differs: " + relative)
    return root


def shared_module(name, root=None):
    """Use one pinned package, refusing an already-imported shadow package."""
    root = shared_root(root)
    package_root = root / "packages/particle-sliders-core/src"
    for prefix, expected_root in (("particle_sliders", package_root), ("conceptmod", root)):
        for module_name, module in tuple(sys.modules.items()):
            if module_name == prefix or module_name.startswith(prefix + "."):
                filename = getattr(module, "__file__", None)
                if filename and not Path(filename).resolve().is_relative_to(expected_root):
                    raise RuntimeError("Shared import is shadowed: " + module_name)
    for path in (str(root), str(package_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Shared module resolved outside reviewed checkout: " + name)
    return module
