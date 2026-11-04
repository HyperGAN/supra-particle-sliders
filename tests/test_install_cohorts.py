"""Keep the default installer and historical checkpoint source cohorts distinct."""
import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def requirements(name):
    return [line.strip() for line in (ROOT / name).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def test_default_installer_uses_the_packaged_lifecycle_receipt():
    assert requirements("requirements.txt") == [".[test]"]
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    receipt = json.loads((ROOT / "supra/core012-ddb4142-receipt.json").read_text())
    core = next(Requirement(value) for value in metadata["project"]["dependencies"]
                if Requirement(value).name == "particle-sliders-core")
    assert core.url == (
        "git+https://github.com/HyperGAN/particle-sliders.git@"
        + receipt["source_commit"] + "#subdirectory=packages/particle-sliders-core")
    assert metadata["project"]["entry-points"]["particle_sliders.image_backends"] == {
        "supra": "supra.lifecycle:BACKEND"}
    assert metadata["project"]["entry-points"]["particle_sliders.inference_backends"] == {
        "supra": "supra.native_inference:BACKEND"}


def test_historical_installer_keeps_the_cdf_lock_and_exact_pg():
    lock_bytes = (ROOT / "shared.lock.json").read_bytes()
    assert (ROOT / "supra/shared.lock.json").read_bytes() == lock_bytes
    lock = json.loads(lock_bytes)
    legacy = {item.name: item for item in map(Requirement, requirements("requirements-legacy.txt"))}
    assert legacy["particle-sliders-core"].url == (
        "git+" + lock["repository"] + ".git@" + lock["commit"]
        + "#subdirectory=packages/particle-sliders-core")
    assert str(legacy["particlegan"].specifier) == "==" + lock["pg_version"]
    current = json.loads((ROOT / "supra/core012-ddb4142-receipt.json").read_text())
    assert lock["commit"] != current["source_commit"]
