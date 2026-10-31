"""Check actual built wheel membership, including previously stale build output."""
from email.parser import BytesParser
import os
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from packaging.version import Version
import pytest

STABLE = set("""__init__.py cache.py campaign.py card.py convergence.py core081-receipt.json
distillation.py lifecycle.lock.json lifecycle.py native_inference.py native_io.py
native_model.py pg09.py runtime.py shared.lock.json shared.py thin-product-provenance.json
train.py licenses/LICENSE licenses/LICENSE-Supra2-IMG licenses/NOTICE""".split())
CANDIDATE = {"hndl.py", "hndl_lifecycle.py", "hndl_inference.py", "supra_lora.hndl", "core012-receipt.json"}


@pytest.mark.parametrize("profile", ["STABLE", "CANDIDATE"])
def test_exact_wheel_contents_and_metadata(profile):
    value = os.environ.get("SUPRA_" + profile + "_WHEEL")
    if not value:
        pytest.skip("supply the built wheel paths for packaging qualification")
    with zipfile.ZipFile(Path(value)) as wheel:
        names = wheel.namelist()
        contents = {name.removeprefix("supra/") for name in names if name.startswith("supra/") and not name.endswith("/")}
        assert contents == STABLE | (CANDIDATE if profile == "CANDIDATE" else set())
        metadata = BytesParser().parsebytes(wheel.read(next(name for name in names if name.endswith(".dist-info/METADATA"))))
        assert metadata["Name"] == "supra-particle-sliders"
        version = Version(metadata["Version"])
        assert version.is_prerelease == (profile == "CANDIDATE")
        entries = wheel.read(next(name for name in names if name.endswith(".dist-info/entry_points.txt"))).decode()
        if profile == "STABLE":
            assert str(version) == "0.2.0" and "supra_hndl" not in entries
        else:
            assert str(version) == "0.3.0.dev1" and "supra_hndl" in entries
            assert any("4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231" in requirement
                       for requirement in metadata.get_all("Requires-Dist"))


def test_normal_pip_selection_does_not_upgrade_stable_to_candidate(tmp_path):
    value = os.environ.get("SUPRA_STABLE_WHEEL")
    if not value:
        pytest.skip("supply the built wheel directory for resolver qualification")
    report = tmp_path / "selection.json"
    result = subprocess.run([sys.executable, "-m", "pip", "--isolated", "install", "--dry-run",
        "--ignore-installed", "--no-deps", "--no-index", "--find-links", str(Path(value).parent),
        "--report", str(report), "supra-particle-sliders"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert [item["metadata"]["version"] for item in json.loads(report.read_text())["install"]] == ["0.2.0"]
