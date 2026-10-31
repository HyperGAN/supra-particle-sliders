"""Cleanup restores every owner even when a callback fails."""
from types import SimpleNamespace

import pytest
import torch

from supra import native_io


@pytest.mark.parametrize("primary", [False, True])
def test_cleanup_attempts_every_owner_and_preserves_primary_error(monkeypatch, primary):
    roots = [torch.nn.Linear(1, 1) for _ in range(3)]
    runtime = SimpleNamespace(model=roots[0], text_encoder=roots[1], vae=roots[2],
        cache={"original": 1}, token_counts={"original": 2}, cfg_adapter_branch="both",
        loras=[SimpleNamespace(multiplier=.25)])
    original_cache, tokens = runtime.cache, runtime.token_counts
    calls = []
    native_cleanup = native_io.native_cleanup

    def failing_cleanup(root):
        restore = native_cleanup(root)

        def cleanup():
            calls.append(root)
            restore()
            if root is runtime.vae:
                raise RuntimeError("decoder cleanup probe")
        return cleanup

    monkeypatch.setattr(native_io, "native_cleanup", failing_cleanup)
    restore = native_io.snapshot_native(runtime)
    for root in roots:
        root.eval().requires_grad_(False)
    runtime.cache, runtime.cfg_adapter_branch = {}, "conditional_only"
    runtime.token_counts["new"] = 4
    runtime.encode = lambda prompt: None
    runtime.loras[0].multiplier = 1.
    if primary:
        with pytest.raises(LookupError, match="primary native error") as error:
            try:
                raise LookupError("primary native error")
            finally:
                restore()
        assert any("decoder cleanup probe" in note for note in error.value.__notes__)
    else:
        with pytest.raises(RuntimeError, match="decoder cleanup probe"):
            restore()
    assert calls == list(reversed(roots))
    assert runtime.cache is original_cache and runtime.token_counts is tokens
    assert tokens == {"original": 2} and runtime.cfg_adapter_branch == "both"
    assert "encode" not in vars(runtime) and runtime.loras[0].multiplier == .25
    assert all(root.training and all(parameter.requires_grad for parameter in root.parameters()) for root in roots)


@pytest.mark.parametrize("contents", [None, "{broken", "[]"])
def test_context_receipt_has_explicit_value_error_contract(tmp_path, contents):
    contexts = tmp_path / "contexts.safetensors"
    if contents is not None:
        contexts.with_suffix(".json").write_text(contents)
    with pytest.raises(ValueError, match="context receipt"):
        native_io.read_context_receipt(contexts)


def test_checkout_override_cannot_mask_the_imported_core(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import particle_sliders
    import concept_slider_core
    root = Path(particle_sliders.__file__).parent
    shutil.copytree(root, tmp_path / "particle_sliders", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(concept_slider_core.__file__).parent, tmp_path / "concept_slider_core",
                    ignore=shutil.ignore_patterns("__pycache__"))
    path = tmp_path / "particle_sliders/pg09.py"
    path.write_text(path.read_text() + "\n# deliberate imported source tamper\n")
    program = "import sys; sys.path.insert(0, sys.argv[1]); from supra import native_io; native_io.verify_sources()"
    result = subprocess.run([sys.executable, "-I", "-c", program, str(tmp_path)],
        env=dict(os.environ, PARTICLE_SLIDERS_ROOT=str(root), CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "Imported shared particle-sliders source differs" in result.stderr
