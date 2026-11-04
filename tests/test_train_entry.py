"""The historical public entry must execute the actual native GAN worker."""
from pathlib import Path
import os
import subprocess
import sys
import pytest
from supra.train import main

ROOT = Path(__file__).resolve().parents[1]


def test_public_entry_dispatches_the_installed_shared_worker(monkeypatch):
    import particle_sliders.session_cli as worker
    calls = []
    monkeypatch.setattr(worker, "main", lambda argv: calls.append(argv) or "worker-result")
    argv = ["--config", "/reviewed/session.json", "--check", "--no-reporting"]
    assert main(argv) == "worker-result"
    assert calls == [["train", "--profile", "supra", *argv]]


def test_public_help_has_isolated_session_and_full_resume_contract(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text and "--runtime-python" in help_text
    assert "--checkpoint-every" in help_text and "--resume" in help_text
    assert "--stop-after" in help_text and "--check" in help_text


def test_product_uses_the_installed_dispatcher_instead_of_historical_campaign():
    source = (ROOT / "supra/train.py").read_text()
    assert "particle_sliders.session_cli" in source
    assert "scripts.train_pg09" not in source


@pytest.mark.parametrize("script", ["infer_supra.py", "distill_final_boss.py", "train_pg09.py",
    "prepare_pg09_contexts.py", "train_lora_supra.py", "train_final_boss.py", "converge_final_boss.py"])
def test_entrypoint_help_works_before_package_resolution(script):
    environment = dict(os.environ)
    environment.pop("PARTICLE_SLIDERS_ROOT", None)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["train_final_boss.py", "converge_final_boss.py"])
def test_archival_mse_entry_rejects_before_source_or_output(script, tmp_path):
    environment = dict(os.environ)
    environment.pop("PARTICLE_SLIDERS_ROOT", None)
    destination = tmp_path / "absent"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--out", str(destination)],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "--allow-legacy-mse" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("script", ["train_pg09.py", "train_lora_supra.py"])
def test_canonical_alias_propagates_dispatcher_failure(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
        "--config", str(tmp_path / "missing.json"), "--check", "--no-reporting"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "FileNotFoundError" in result.stderr
