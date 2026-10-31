"""Native-only HNDL inference factory; no product campaign or cache construction."""
from contextlib import contextmanager

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Capabilities

from . import hndl, native_io
from .shared import source_cohort

PROTOCOL = "supra-hndl-native-conditional-cfg-euler-v1"
CAPABILITIES = Capabilities(generator_arm="adapter_gan", render=True,
    strength_min=None, strength_max=None, reference_images=False,
    base_precisions=("float32",), render_outputs=("pil",), operations=("render",),
    qualification="HNDL native-only CPU software fixture; pretrained/native CUDA qualification pending")


def validate(options, spec):
    if set(options) - {"device", "base_precision", "frozen_base_sha256"}:
        raise ValueError("Native Supra HNDL accepts no training assets")
    device = options.get("device", "")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Native Supra HNDL inference requires explicit cuda:N")
    if options.get("base_precision", "float32") != "float32":
        raise ValueError("Native Supra HNDL preserves FP32 storage")
    if (spec.configuration.keys() != {"rank"} or type(spec.configuration["rank"]) is not int
            or not 1 <= spec.configuration["rank"] <= 256):
        raise ValueError("Native Supra HNDL requires exact bounded adapter rank")
    if options.get("frozen_base_sha256") != spec.identity["model"]["frozen_parameter_sha256"]:
        raise ValueError("Native frozen base identity differs")
    hndl.verify_sources()


def preflight(options, request, spec):
    native_io.validate_render(request["native"], partial=True)


def prepare(options, spec):
    from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter
    runtime = hndl.load_native(dict(options, rank=spec.configuration["rank"]))
    close = native_io.snapshot_native(runtime)
    try:
        if not runtime.stored_unconditional:
            raise ValueError("Native Supra requires stored unconditional context")
        identity = hndl.model_identity(runtime)
        if identity != spec.identity["model"]:
            raise ValueError("Native Supra HNDL model/source identity differs")
        backend = SupraRuntimeResponseAdapter(runtime, model_identity=identity)
        runtime.cfg_adapter_branch = "conditional_only"
        original_encode = runtime.encode

        @contextmanager
        def observation():
            restore = native_io.snapshot_native(runtime)
            runtime.cache = dict(runtime.cache)
            runtime.encode = original_encode
            try:
                with source_cohort(hndl.LOCK):
                    yield
            finally:
                restore()

        return InferenceBinding(backend.network,
            native_identity(backend, model_identity=identity, protocol=PROTOCOL), close,
            render=lambda **request: native_io.render(backend, runtime, **request),
            observation=observation)
    except BaseException:
        close()
        raise


BACKEND = InferenceBackend("supra_hndl", CAPABILITIES, prepare, validate,
                          response=False, preflight=preflight)
