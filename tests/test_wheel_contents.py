"""Check actual built wheel membership, including previously stale build output."""
from email.parser import BytesParser
import os
import json
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

from packaging.version import Version
import pytest

STABLE = set("""__init__.py cache.py campaign.py card.py convergence.py core081-receipt.json
distillation.py lifecycle.lock.json lifecycle.py native_inference.py native_io.py
native_model.py pg09.py runtime.py shared.lock.json shared.py thin-product-provenance.json
train.py licenses/LICENSE licenses/LICENSE-Supra2-IMG licenses/NOTICE""".split())
CANDIDATE = {"hndl.py", "hndl_lifecycle.py", "hndl_inference.py", "supra_lora.hndl", "core012-receipt.json"}


@pytest.fixture
def built_wheels():
    values = {profile: os.environ.get("SUPRA_" + profile + "_WHEEL") for profile in ("STABLE", "CANDIDATE")}
    if not any(values.values()):
        pytest.skip("supply both built wheel paths for packaging qualification")
    assert all(values.values()), "Both stable and candidate wheel paths are required"
    wheels = {profile: Path(value).resolve() for profile, value in values.items()}
    assert wheels["STABLE"] != wheels["CANDIDATE"]
    assert all(path.is_file() for path in wheels.values()), "Both wheel artifacts must actually exist"
    return wheels


@pytest.mark.parametrize("profile", ["STABLE", "CANDIDATE"])
def test_exact_wheel_contents_and_metadata(profile, built_wheels):
    with zipfile.ZipFile(built_wheels[profile]) as wheel:
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


@pytest.mark.parametrize("available,expected", [("both", "0.2.0"), ("candidate_only", "0.3.0.dev1")])
def test_pip_selection_with_both_and_candidate_only_availability(tmp_path, built_wheels, available, expected):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    profiles = ("STABLE", "CANDIDATE") if available == "both" else ("CANDIDATE",)
    for profile in profiles:
        shutil.copy2(built_wheels[profile], wheelhouse)
    assert {path.name for path in wheelhouse.iterdir()} == {built_wheels[profile].name for profile in profiles}
    report = tmp_path / "selection.json"
    result = subprocess.run([sys.executable, "-m", "pip", "--isolated", "install", "--dry-run",
        "--ignore-installed", "--no-deps", "--no-index", "--find-links", str(wheelhouse),
        "--report", str(report), "supra-particle-sliders"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert [item["metadata"]["version"] for item in json.loads(report.read_text())["install"]] == [expected]
