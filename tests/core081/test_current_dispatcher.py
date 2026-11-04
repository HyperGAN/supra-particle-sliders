"""Current installed package + dispatcher admission, with synthetic CPU assets."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from particle_sliders.execution.request import resolve_request
from supra import native_io
from supra.training_profile import profile
from supra.shared import file_sha
from .test_lifecycle_closure import configured  # noqa: F401
from .test_thin_relocation import assets  # noqa: F401


def test_current_training_profile_preserves_native_qualification(configured):
    _, config = configured
    request = resolve_request(config.as_dict(), profile=profile(), no_reporting=True)
    assert request["profile"] == "supra"
    assert request["resolved"]["profile"]["status"] == "source_only"
    assert request["training"]["backend"] == "supra"
    assert request["resolved"]["target_steps"] == 4


def test_installed_dispatcher_plan_and_check_without_native_allocation(configured, tmp_path):
    _, config = configured
    path = tmp_path / "session.json"
    path.write_text(json.dumps(config.as_dict()))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONPATH="")
    planned = subprocess.run([sys.executable, "-I", "-m", "particle_sliders.session_cli", "plan",
        "--profile", "supra", "--config", str(path), "--no-reporting"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert planned.returncode == 0, planned.stderr
    request = json.loads(planned.stdout)
    assert request["resolved"]["profile"]["status"] == "source_only"
    assert request["resolved"]["profile"]["backend"] == "supra"
    checked = subprocess.run([sys.executable, "-I", "-m", "supra.train",
        "--config", str(path), "--check", "--no-reporting"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["status"] == "checked" and result["profile"] == "supra"
    assert result["qualification"] == "source_only" and result["target_steps"] == 4
    assert json.loads(result["worker_output"]) == {"status": "checked", "backend": "supra"}


def test_dispatcher_rejects_changed_input_before_output(configured, tmp_path):
    _, config = configured
    value = config.as_dict()
    value["options"]["cache_sha256"] = "0" * 64
    path = tmp_path / "bad-session.json"
    path.write_text(json.dumps(value))
    output = tmp_path / "must-remain-absent"
    checked = subprocess.run([sys.executable, "-I", "-m", "supra.train", "--config", str(path),
        "--out", str(output), "--check", "--no-reporting"], cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONPATH=""),
        capture_output=True, text=True, timeout=60)
    assert checked.returncode == 2 and "checksum" in checked.stderr
    assert not output.exists()


@pytest.mark.parametrize("field", ["version", "pg_version", "schema", "source_commit", "namespaces"])
def test_current_version_and_identity_receipt_are_enforced(tmp_path, monkeypatch, field):
    value = json.loads(native_io.LOCK.read_text())
    value[field] = "unqualified"
    path = tmp_path / "wrong-current-receipt.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(native_io, "LOCK", path)
    with pytest.raises(ValueError, match="receipt"):
        native_io.verify_sources()
