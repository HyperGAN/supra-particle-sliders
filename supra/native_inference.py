"""Supra native-only v2 factory; imports no campaign/cache/critic/optimizer."""
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Capabilities

from . import native_io

PROTOCOL = "supra-native-conditional-cfg-euler-v1"
CAPABILITIES = Capabilities(generator_arm="adapter_gan", render=True,
    strength_min=None, strength_max=None, reference_images=False,
    base_precisions=("float32",), render_outputs=("pil",), operations=("render",),
    qualification="Source proposal; native-only installed CPU conformance and native GPU parity pending")


def validate(options, spec):
    if set(options) - {"device", "base_precision", "frozen_base_sha256"}:
        raise ValueError("Native Supra accepts only local device/precision/base identity; no training assets")
    device = options.get("device", "")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Native Supra inference requires explicit cuda:N; no CPU native loader")
    if options.get("base_precision", "float32") != "float32":
        raise ValueError("Supra preserves FP32 storage with native BF16 CUDA autocast")
    if spec.configuration.keys() != {"rank"} or type(spec.configuration["rank"]) is not int:
        raise ValueError("Native Supra specification requires only the exact adapter rank")
    if not 1 <= spec.configuration["rank"] <= 256:
        raise ValueError("Native Supra adapter rank is outside bounds")
    if options.get("frozen_base_sha256") != spec.identity["model"]["frozen_parameter_sha256"]:
        raise ValueError("Supply the artifact's exact admitted frozen base SHA")
    native_io.verify_sources()


def preflight(options, request, spec):
    native_io.validate_render(request["native"], partial=True)


def prepare(options, spec):
    from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter
    runtime = native_io.load_native(dict(options, rank=spec.configuration["rank"]))
    close = native_io.snapshot_native(runtime)
    try:
        if not runtime.stored_unconditional:
            raise ValueError("Native Supra requires its reviewed stored unconditional context")
        identity = native_io.model_identity(runtime)
        if identity != spec.identity["model"]:
            raise ValueError("Loaded native Supra base/source identity differs")
        backend = SupraRuntimeResponseAdapter(runtime, model_identity=identity)
        runtime.cfg_adapter_branch = "conditional_only"
        return InferenceBinding(backend.network,
            native_identity(backend, model_identity=identity, protocol=PROTOCOL), close,
            render=lambda **request: native_io.render(backend, runtime, **request),
            observation=lambda: native_io.observation(runtime, runtime.encode))
    except BaseException:
        close()
        raise


BACKEND = InferenceBackend("supra", CAPABILITIES, prepare, validate,
                           response=False, preflight=preflight)
