"""Product profile for the shared training dispatcher; native qualification is pending."""

def profile():
    return {
        "schema": "particle-slider-profile-v1", "name": "supra", "product": "supra-particle-sliders",
        "config_kind": "session", "backend": "supra", "interface": "image", "status": "source_only",
        "reason": "CPU software conformance only; pretrained/native CUDA/cache parity and quality remain unqualified",
        "runtime": {}, "default_checkpoint_every": 80,
        "capabilities": {"resume": True, "periodic_checkpoint": True, "check": True},
    }


def hndl_profile():
    value = profile()
    value.update(name="supra_hndl", backend="supra_hndl")
    return value
