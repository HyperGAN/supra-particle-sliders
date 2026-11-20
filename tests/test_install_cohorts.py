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
    receipt = json.loads((ROOT / "supra/engine-receipt.json").read_text())
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
    current = json.loads((ROOT / "supra/engine-receipt.json").read_text())
    assert lock["commit"] != current["source_commit"]


def test_current_installed_pg095_guard_and_old_profile_refusal():
    from importlib.metadata import version
    import pytest
    from supra.native_io import verify_sources
    from particle_sliders.engine import ConditionalParticleGAN, PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.engine import validate_engine_domain
    verify_sources()
    assert version('particle-sliders-core') == '0.15.1'
    assert (PG_VERSION, PG_SOURCE_PROFILE) == ('0.11.0', 'release-0.11.0')
    with pytest.raises(ValueError, match='source profile'):
        validate_engine_domain((1, 4, 1, 2, 2), response_domain='continuous_field', pg_source_profile='release-0.9.4')
