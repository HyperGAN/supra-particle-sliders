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
    backend = json.loads((ROOT / "backend.lock.json").read_text())
    assert backend["source_lock"] == "supra/engine-receipt.json"
    assert backend["core_commit"] == receipt["source_commit"]
    assert backend["core_version"] == receipt["version"]
    assert backend["particlegan_version"] == receipt["pg_version"]
    assert backend["entrypoint"] == "particle_sliders.engine.ConditionalParticleGAN"
    core = next(Requirement(value) for value in metadata["project"]["dependencies"]
                if Requirement(value).name == "particle-sliders-core")
    assert core.url == (
        "git+https://github.com/HyperGAN/particle-sliders.git@"
        + receipt["source_commit"] + "#subdirectory=packages/particle-sliders-core")
    assert metadata["project"]["entry-points"]["particle_sliders.image_backends"] == {
        "supra": "supra.lifecycle:BACKEND"}
    assert metadata["project"]["entry-points"]["particle_sliders.inference_backends"] == {
        "supra": "supra.native_inference:BACKEND"}


def test_current_installed_release_guard_and_old_profile_refusal():
    from importlib.metadata import version
    import pytest
    from supra.native_io import verify_sources
    from particle_sliders.engine import ConditionalParticleGAN, PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.engine import validate_engine_domain
    verify_sources()
    assert version('particle-sliders-core') == '0.15.2'
    assert (PG_VERSION, PG_SOURCE_PROFILE) == ('0.11.1', 'release-0.11.1')
    with pytest.raises(ValueError, match='source profile'):
        validate_engine_domain((1, 4, 1, 2, 2), response_domain='continuous_field', pg_source_profile='release-0.11.0')
