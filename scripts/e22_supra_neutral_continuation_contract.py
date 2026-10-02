"""New orchestration contracts; the qualified optimizer/model code stays held.

The nested helpers below execute the original code objects in an isolated
namespace with explicit closure owners. No held module globals are mutated.
This module imports no Torch/native package and is usable by CPU preflight.
"""
from __future__ import annotations

import hashlib
import gc
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
PIN = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
ARMS = ("sampled_control", "sampled_hb_neutral")
MODES = dict(zip(ARMS, ("sampled_v1", "sampled_hb_neutral_v1")))
START, STOP = 6400, 12800
ENDPOINTS = (9600, 12800)
LOCAL_CHECKPOINTS = (6402, *range(6800, STOP + 1, 400))
CARD = ROOT / "docs/e22_supra_neutral_continuation_protocol.json"
HELPERS = ("save_state", "immutable_owners", "all_edit", "fresh", "full_evaluation", "clean_probe")
CLASSES = ("FastReference", "OrdinaryReference")


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_identity(code):
    """Bind all CodeType fields, including nested constants, without marshal flags."""
    def encode(value):
        if isinstance(value, types.CodeType):
            return {name: encode(getattr(value, name)) for name in dir(value)
                    if name.startswith("co_") and name not in ("co_lines", "co_positions")}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [encode(item) for item in value]}
        if isinstance(value, frozenset):
            return {"frozenset": sorted(map(repr, value))}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"type": type(value).__name__, "repr": repr(value)}
    return hashlib.sha256(json.dumps(encode(code), sort_keys=True, allow_nan=False).encode()).hexdigest()


def nested_code(parent, name):
    found = [item for item in parent.co_consts if isinstance(item, types.CodeType) and item.co_name == name]
    if len(found) != 1:
        raise ValueError("exactly one held code object required: " + name)
    return found[0]


def _cell(value):
    return (lambda: value).__closure__[0]


def bind_code(code, namespace, owners):
    missing = set(code.co_freevars) - set(owners)
    if missing:
        raise ValueError("missing explicit closure owners: " + repr(sorted(missing)))
    return types.FunctionType(code, dict(namespace), code.co_name,
                              closure=tuple(_cell(owners[name]) for name in code.co_freevars) or None)


def held_manifest(held):
    main = held.main.__code__
    return dict(source_sha256=sha(held.__file__), main_code_sha256=code_identity(main),
        helpers={name: dict(code_sha256=code_identity(nested_code(main, name)),
                           closure_names=list(nested_code(main, name).co_freevars)) for name in HELPERS},
        classes={name: {item.co_name: dict(code_sha256=code_identity(item), closure_names=list(item.co_freevars))
                        for item in nested_code(main, name).co_consts if isinstance(item, types.CodeType)}
                 for name in CLASSES},
        binding="Exact held code objects; new isolated globals and explicit closure cells; no source copy or module mutation.")


def bind_helpers(held, owners):
    namespace = dict(held.__dict__)
    namespace["__name__"] = __name__ + ".isolated_held_helpers"
    result = {name: bind_code(nested_code(held.main.__code__, name), namespace, owners) for name in HELPERS}
    for name in CLASSES:
        methods = {item.co_name: bind_code(item, namespace, owners)
                   for item in nested_code(held.main.__code__, name).co_consts if isinstance(item, types.CodeType)}
        result[name] = type(name, (), methods)
    # Decorators/defaults live outside the function code. Preserve them explicitly.
    result["FastReference"].__init__.__defaults__ = (False,)
    for name in ("full_evaluation", "clean_probe"):
        result[name] = owners["torch"].no_grad()(result[name])
    result["OrdinaryReference"].routed_forward = owners["torch"].no_grad()(result["OrdinaryReference"].routed_forward)
    return result


def require_checkpoint_contract(state, expected, step):
    """Canonical comparison is only for JSON plans, never native restore mutation."""
    config, policy = state["config"], state["policy"]
    if policy["completed_steps"] != step or canonical(config) != expected:
        raise ValueError("continuation clock/config mismatch")
    if not (config["training_schedule"] == "fresh_editing_only_v1"
            and config["preservation_game_weight"] == 0 and not config["output_error_guard"]
            and config["max_feature_context_harm"] == 0 and config["recipe"] == policy["recipe"]
            and config["recipe"].get("total_steps") is None):
        raise ValueError("continuation must retain the native editing-only game")


def paired_row_contract(left, right, step, expected_indices=None):
    for row in (left, right):
        if row["step"] != step or row["hold"] is not False or row["game_weight"] != 1.:
            raise ValueError("editing-only continuation row mismatch")
        if expected_indices is not None and row["batch_indices"] != expected_indices:
            raise ValueError("CPU7 continuation stream mismatch")
    for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight"):
        if left[key] != right[key]:
            raise ValueError("unmatched continuation streams: " + key)


def ordinary_metadata_contract(metadata, host, step, prompts_sha256):
    """The historical native LoRA exporter JSON-encodes metadata scalar values."""
    expected=dict(model_id=host["model_id"],model_revision=host["model_revision"],
        text_encoder_revision=host["text_encoder_revision"],vae_revision=host["vae_revision"],
        step=step,rank=16,alpha=16,prompts_sha256=prompts_sha256,
        targets=["ctx_proj","cross_attn.q","cross_attn.kv","cross_attn.proj","self_attn.qkv","self_attn.proj"])
    if json.loads(metadata["format"])!="supra-native-lora-v1" or any(json.loads(metadata[key])!=value for key,value in expected.items()):
        raise ValueError("historical ordinary artifact source/geometry/clock pins changed")


def exact_recovery(fresh_restore, update, checkpoint, digest):
    """Two public restores from the same boundary; all native/caller owners count."""
    direct = fresh_restore()
    rows = [update(direct) for _ in range(2)]
    direct_digest = digest(checkpoint(direct))
    del direct
    gc.collect()
    replay = fresh_restore()
    replay_rows = [update(replay) for _ in range(2)]
    replay_digest = digest(checkpoint(replay))
    del replay
    gc.collect()
    if digest(rows) != digest(replay_rows) or direct_digest != replay_digest:
        raise ValueError("6400 to 6402 native/caller recovery is not exact")
    return dict(from_step=START, to_step=START + 2, native_replay_updates=4,
                rows=rows, row_digest=digest(rows), native_digest=direct_digest,
                rows_exact=True, state_exact=True)
