"""Source prototype: TRAIN trajectory support2->8, with native game unchanged.

No experiment is launched here. Root must separately qualify/freeze preparation
and a matched reduced-carrier trial before authorizing full training. Old data,
native reservoirs, optimizer/EMA/controller state and precision are preserved.
"""
from copy import deepcopy
import time

import torch

from .particle_final_precision import checkpoint_precision, restore_precision, precision_rng_scope
from .particle_game import update
from .particle_pilot import state_digest
from .particle_training_data import FrozenSliderContexts

DATA_SCHEMA = "supra_latent_coverage_data_v1"
SOFTWARE_DATA_SCHEMA = "supra_latent_coverage_software_data_v1"
CHECKPOINT_SCHEMA = "supra_latent_coverage_checkpoint_v1"
TAG = "latent_coverage"
SOURCES = (18, 4, 17, 5, 16, 15)
TARGETS = (11, 3, 2, 13, 12, 14)
BASE_SEEDS = (7063, 7064)
TRAIN_SEEDS = tuple(range(7063, 7071))
TEST_SEEDS = (39001, 39002)
STEPS = 50
TIMES = tuple(range(0, STEPS, 5))
SAMPLING = dict(common_addresses=960,draw="CPU7 torch.randint(960,(4,)) with replacement",
    expanded="160*s+20*j+10*p+k",baseline="40*s+20*floor(j/4)+10*p+k",
    marginals="uniform240 versus uniform960; common source/time/path addresses",
    historical_sequence="original torch.randint(240) address sequence is not reproduced")


def _finite(value, shape, dtype, name):
    if (not isinstance(value, torch.Tensor) or value.shape != torch.Size(shape)
            or value.dtype != dtype or not bool(torch.isfinite(value).all())):
        raise ValueError("finite tensor shape/dtype differs: " + name)


def _original(data):
    """Validate existing ordered data, without re-estimating scale or guards."""
    expected_time = torch.tensor([k / STEPS for k in TIMES], dtype=torch.float32)
    for pool in ("fit", "test"):
        x = data[pool]["context"]
        _finite(x, (240, 4100), torch.float32, pool)
        if x.device.type != "cpu" or bool(data[pool]["hold"].any()):
            raise ValueError("original CPU editing-only pool required")
        for s, (source, target) in enumerate(zip(SOURCES, TARGETS)):
            for j in range(2):
                for p in range(2):
                    for k in range(10):
                        row = x[40*s + 20*j + 10*p + k]
                        if (row[4096] != expected_time[k] or float(row[4097]) != source
                                or float(row[4098]) != target or float(row[4099]) != 1.):
                            raise ValueError("original source/target/time/index law differs")
        for name in ("targets", "base"):
            _finite(data[pool][name], (240, 4, 32, 32), torch.float32, pool+"/"+name)
    _finite(data["guard"]["context"], (64, 4100), torch.float32, "fixed guard")
    _finite(data["guard"]["targets"], (64, 4, 32, 32), torch.float32, "fixed guard targets")
    _finite(data["coordinate_scale"], (16,), torch.float32, "fixed coordinate scale")


def _packed(z, time_value, source, target):
    tail = torch.tensor([[time_value, a, b, 1.] for a, b in zip(source, target)],
                        device=z.device, dtype=torch.float32)
    return torch.cat((z.flatten(1), tail), dim=1)


@torch.no_grad()
def build_latent_coverage_data(original, encoder, *, seconds, provenance, progress=None):
    """Prepare960 canonical FIT rows using the original full pure-base teacher.

Every call to teacher_velocity is B4 (internal CFG B8), a declared NEW builder
arithmetic cohort. Original240 cached rows are copied exactly, not regenerated.
At ten retained times the opposite caption is also evaluated so new cached
base/target fields keep their existing physical-velocity meaning. They never
become losses or update guards. Requires an external startup-through-writes cap.
"""
    started = time.monotonic()
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        raise ValueError("explicit root-reviewed preparation budget required")
    def deadline():
        if time.monotonic() - started > seconds:
            raise TimeoutError("latent coverage builder cap")
    _original(original)
    if (type(encoder) is not FrozenSliderContexts
            or getattr(encoder.teacher_velocity,"__func__",None) is not FrozenSliderContexts.teacher_velocity
            or getattr(encoder.unpack,"__func__",None) is not FrozenSliderContexts.unpack):
        raise ValueError("exact live pure-target FrozenSliderContexts contract required")
    teacher = encoder.teacher
    if (encoder.cfg != 3. or len(teacher.blocks) != 14
            or any(p.requires_grad for p in teacher.parameters())
            or not torch.equal(encoder.contexts.cpu(), original["text_contexts"])
            or not torch.equal(encoder.masks.cpu(), original["text_masks"])):
        raise ValueError("original frozen14-block/text/CFG3 teacher required")
    device = encoder.contexts.device
    modes = [(m, m.training) for m in encoder.modules()]
    multipliers = [(m, m.multiplier) for m in teacher.modules() if hasattr(m, "multiplier")]
    encoder_origin = state_digest(encoder.state_dict())
    original_origin = state_digest(original)
    devices = [device.index or 0] if device.type == "cuda" else []
    new_rows = {}
    counts = dict(teacher_B4_forwards=0, Euler_B4_forwards=0,
                  opposite_caption_B4_forwards=0, private_seed_draws=0,
                  native_updates=0, optimizer_steps=0)
    try:
        with torch.random.fork_rng(devices=devices):
            # One named initial Gaussian per added seed, reused globally for
            # every source and both paths. No held-out seed is generated.
            initial = {}
            for seed in TRAIN_SEEDS[2:]:
                stream = torch.Generator(device=device).manual_seed(seed)
                initial[seed] = torch.randn(1, 4, 32, 32, device=device,
                                           dtype=torch.float32, generator=stream)
                counts["private_seed_draws"] += 1
            addresses = [(s, j, p) for s in range(6) for j in range(2, 8) for p in range(2)]
            for panel in range(18):
                group = addresses[4*panel:4*panel+4]
                z = torch.cat([initial[TRAIN_SEEDS[j]].clone() for _, j, _ in group])
                path_ids = [SOURCES[s] if p == 0 else TARGETS[s] for s, _, p in group]
                opposite_ids = [TARGETS[s] if p == 0 else SOURCES[s] for s, _, p in group]
                for step in range(STEPS):
                    velocity = encoder.teacher_velocity(_packed(z, step/STEPS, path_ids, path_ids))
                    _finite(velocity, (4, 4, 32, 32), torch.float32, "pure trajectory velocity")
                    counts["teacher_B4_forwards"] += 1; counts["Euler_B4_forwards"] += 1
                    if step in TIMES:
                        opposite = encoder.teacher_velocity(_packed(z, step/STEPS, opposite_ids, opposite_ids))
                        _finite(opposite, (4, 4, 32, 32), torch.float32, "paired cached velocity")
                        counts["teacher_B4_forwards"] += 1; counts["opposite_caption_B4_forwards"] += 1
                        for b, (s, j, path) in enumerate(group):
                            source, target = SOURCES[s], TARGETS[s]
                            context = _packed(z[b:b+1], step/STEPS, [source], [target])[0].cpu().clone()
                            new_rows[(s, j, path, step//5)] = dict(context=context,
                                base=(velocity[b] if path == 0 else opposite[b]).cpu().clone(),
                                targets=(opposite[b] if path == 0 else velocity[b]).cpu().clone())
                    z = z + velocity / STEPS
                    deadline()
                if progress:
                    progress(dict(stage="pure_teacher_trajectory",panel=panel+1,panels=18,
                                  counts=dict(counts),seconds=time.monotonic()-started))
    finally:
        for module, mode in modes: module.training = mode
        for module, multiplier in multipliers: module.multiplier = multiplier
        if state_digest(encoder.state_dict()) != encoder_origin or state_digest(original) != original_origin:
            raise AssertionError("pure teacher/data changed during builder")
    if counts != dict(teacher_B4_forwards=1080,Euler_B4_forwards=900,
                      opposite_caption_B4_forwards=180,private_seed_draws=6,
                      native_updates=0,optimizer_steps=0):
        raise AssertionError("fixed pure-base preparation counts differ")
    result = deepcopy(original); rows = []; original_indices = []
    for s in range(6):
        for j in range(8):
            for path in range(2):
                for k in range(10):
                    if j < 2:
                        i = 40*s + 20*j + 10*path + k
                        row = {n: original["fit"][n][i].clone() for n in ("context", "base", "targets")}
                        original_indices.append(len(rows))
                    else: row = new_rows[(s, j, path, k)]
                    row.update(prompt=original["fit"]["prompts"][40*s],
                               target_prompt=original["fit"]["target_prompts"][40*s])
                    rows.append(row)
    result["fit"] = dict(context=torch.stack([r["context"] for r in rows]),
        base=torch.stack([r["base"] for r in rows]),targets=torch.stack([r["targets"] for r in rows]),
        source_indices=torch.arange(960),hold=torch.zeros(960,dtype=torch.bool),
        prompts=[r["prompt"] for r in rows],target_prompts=[r["target_prompt"] for r in rows])
    manifest = dict(schema=DATA_SCHEMA,software_only=False,source_order=list(SOURCES),target_order=list(TARGETS),
        original_seeds=list(BASE_SEEDS),train_seeds=list(TRAIN_SEEDS),TEST_seeds=list(TEST_SEEDS),
        index_law="160*s+20*j+10*p+k",paths=["neutral","positive"],Euler_steps=STEPS,
        recorded_steps=list(TIMES),cfg=3.,builder_batch=4,original_indices=original_indices,
        original_dataset_digest=original_origin,expanded_dataset_digest=state_digest(result),
        encoder_tensor_digest=encoder_origin,counts=counts,provenance=deepcopy(provenance),
        builder_arithmetic="original frozen storage/BF16 teacher, B4/internal CFG B8; new added-row cohort",
        original_rows="copied exact240 including cached base/target velocities; not regenerated",
        coordinate_scale="unchanged original",guard="unchanged original mixed64",
        targets="cached absolute velocities for reporting; native paired residual labels remain zero")
    artifact = dict(schema=DATA_SCHEMA,data=result,manifest=manifest)
    validate_latent_coverage_data(original, artifact);deadline()
    return artifact


def _validate_latent_coverage_data(original, artifact, *, software_only):
    _original(original)
    schema=SOFTWARE_DATA_SCHEMA if software_only else DATA_SCHEMA
    if artifact.get("schema") != schema: raise ValueError("coverage data schema differs")
    data, m = artifact["data"], artifact["manifest"]
    expected_counts=dict(teacher_B4_forwards=0,Euler_B4_forwards=0,
        opposite_caption_B4_forwards=0,private_seed_draws=0,native_updates=0,optimizer_steps=0) if software_only else dict(
        teacher_B4_forwards=1080,Euler_B4_forwards=900,opposite_caption_B4_forwards=180,
        private_seed_draws=6,native_updates=0,optimizer_steps=0)
    indices = [160*s+20*j+10*p+k for s in range(6) for j in range(2) for p in range(2) for k in range(10)]
    if (m["train_seeds"] != list(TRAIN_SEEDS) or m["original_seeds"] != list(BASE_SEEDS)
            or m["source_order"] != list(SOURCES) or m["target_order"] != list(TARGETS)
            or m["original_indices"] != indices or m["index_law"] != "160*s+20*j+10*p+k"
            or m["TEST_seeds"] != list(TEST_SEEDS) or m["paths"] != ["neutral","positive"]
            or m["schema"]!=schema or m["software_only"] is not software_only
            or m["Euler_steps"] != (0 if software_only else STEPS) or m["recorded_steps"] != list(TIMES)
            or m["cfg"] != 3. or m["builder_batch"] != (0 if software_only else 4)
            or m["counts"] != expected_counts
            or m["original_dataset_digest"] != state_digest(original)
            or m["expanded_dataset_digest"] != state_digest(data)):
        raise ValueError("coverage seeds/order/subset/data digest differs")
    x = data["fit"]["context"]; _finite(x,(960,4100),torch.float32,"expanded FIT")
    if x.device.type != "cpu" or not torch.equal(data["fit"]["source_indices"],torch.arange(960)):
        raise ValueError("expanded CPU ordered rows required")
    _finite(data["fit"]["base"],(960,4,32,32),torch.float32,"expanded cached base")
    _finite(data["fit"]["targets"],(960,4,32,32),torch.float32,"expanded cached target")
    if data["fit"]["hold"].dtype != torch.bool or data["fit"]["hold"].shape != (960,) or bool(data["fit"]["hold"].any()):
        raise ValueError("expanded FIT must remain editing-only")
    for key in original:
        if key != "fit" and state_digest(data[key]) != state_digest(original[key]):
            raise ValueError("only FIT may change: " + key)
    for name in ("context","base","targets","hold"):
        if not torch.equal(data["fit"][name][indices],original["fit"][name]):
            raise ValueError("original240 subset is not exact: "+name)
    for name in ("prompts","target_prompts"):
        if [data["fit"][name][i] for i in indices] != original["fit"][name]:
            raise ValueError("original cached prompt subset differs: "+name)
    expected_time=torch.tensor([k/STEPS for k in TIMES],dtype=torch.float32)
    for s,(source,target) in enumerate(zip(SOURCES,TARGETS)):
        for j in range(8):
            for p in range(2):
                for k in range(10):
                    row=x[160*s+20*j+10*p+k]
                    if row[4096]!=expected_time[k] or row[4097]!=source or row[4098]!=target or row[4099]!=1.:
                        raise ValueError("expanded source/time/strength row differs")
    return True


def validate_latent_coverage_data(original, artifact):
    return _validate_latent_coverage_data(original,artifact,software_only=False)


def validate_software_latent_coverage_data(original, artifact):
    """Explicit synthetic CPU fixture only; production validator rejects it."""
    return _validate_latent_coverage_data(original,artifact,software_only=True)


def _attach_latent_coverage(loop, original, artifact, *, seeds, bootstrap_proof, software_only):
    """Migrate caller FIT after the explicitly declared public origin restore.

Production requires genuine trained12800 and a full-teacher artifact. The
separate synthetic software schema requires genuine fresh0 and cannot enter
the production validator or attach function.

Do not reset native FIT FIFO, guards, evidence, clocks or RNG. Old observed FIT
rows remain valid members of both pools. New observed samples enter the native
FIFO through unchanged begin_step. The old dataset_digest remains historical;
the explicit tag identifies effective caller data for every new checkpoint.
"""
    _validate_latent_coverage_data(original,artifact,software_only=software_only)
    seeds=tuple(seeds)
    origin=0 if software_only else 12800
    proof_ok=(bootstrap_proof.get("software_only") is True
        and bootstrap_proof.get("public_restore_exact") is True
        and bootstrap_proof.get("completed_steps")==0) if software_only else (
        bootstrap_proof.get("legacy_roundtrip_exact") is True and bootstrap_proof.get("completed_steps")==12800)
    if (seeds not in (BASE_SEEDS,TRAIN_SEEDS) or loop.policy.completed_steps!=origin
            or TAG in loop.config or not proof_ok
            or bootstrap_proof["initialized_after_restore"] is not False
            or loop.config["dataset_digest"] != state_digest(original)
            or not torch.equal(loop.fit_context.cpu(),original["fit"]["context"])):
        raise ValueError("declared public origin restore must precede explicit migration")
    before=checkpoint_precision(loop)
    fit=original["fit"] if seeds==BASE_SEEDS else artifact["data"]["fit"]
    tag=dict(schema=SOFTWARE_DATA_SCHEMA if software_only else DATA_SCHEMA,software_only=software_only,
        origin_step=origin,seeds=list(seeds),fit_rows=len(fit["context"]),
        fit_context_digest=state_digest(fit["context"]),
        original_dataset_digest=state_digest(original),artifact_dataset_digest=artifact["manifest"]["expanded_dataset_digest"],
        manifest_digest=state_digest(artifact["manifest"]),
        coordinate_scale_digest=state_digest(original["coordinate_scale"]),guard_digest=state_digest(original["guard"]),
        historical_dataset_digest_retained=True,native_FIFO="restored retained; no clearing/reset",
        origin_public_restore=deepcopy(bootstrap_proof),sampling=deepcopy(SAMPLING))
    old_fit,old_targets,old_config=loop.fit_context,loop.fit_targets,loop.config
    try:
        loop.fit_context=fit["context"].to(loop.policy.device)
        loop.fit_targets=torch.zeros(len(fit["context"]),4,32,32,device=loop.policy.device,dtype=torch.float32)
        loop.config={**deepcopy(loop.config),TAG:tag}
        after=checkpoint_precision(loop)
        if (state_digest(before["pilot"]["policy"])!=state_digest(after["pilot"]["policy"])
                or state_digest(before["caller"])!=state_digest(after["caller"])
                or not torch.equal(before["pilot"]["data_rng"],after["pilot"]["data_rng"])
                or not torch.equal(before["pilot"]["paired_noise_rng"],after["pilot"]["paired_noise_rng"])
                or not torch.equal(loop.guard_context.cpu(),original["guard"]["context"])
                or bool(loop.guard_targets.count_nonzero())):
            raise AssertionError("migration changed native/caller/guard state")
    except BaseException:
        loop.fit_context,loop.fit_targets,loop.config=old_fit,old_targets,old_config
        restore_precision(loop,before)
        raise
    return deepcopy(tag)


def attach_latent_coverage(loop, original, artifact, *, seeds, bootstrap_proof):
    """Production path: strict genuine12800 origin and full-teacher artifact."""
    return _attach_latent_coverage(loop,original,artifact,seeds=seeds,
        bootstrap_proof=bootstrap_proof,software_only=False)


def attach_software_latent_coverage(loop, original, artifact, *, seeds, bootstrap_proof):
    """Explicit CPU synthetic clock0 path; no trained12800 claim or label."""
    return _attach_latent_coverage(loop,original,artifact,seeds=seeds,
        bootstrap_proof=bootstrap_proof,software_only=True)


def checkpoint_latent_coverage(loop):
    if TAG not in loop.config: raise ValueError("explicit migrated data tag required")
    tag=loop.config[TAG]
    _finite(loop.fit_context,(tag["fit_rows"],4100),torch.float32,"effective FIT contexts")
    _finite(loop.fit_targets,(tag["fit_rows"],4,32,32),torch.float32,"effective zero FIT labels")
    if state_digest(loop.fit_context.cpu())!=tag["fit_context_digest"] or bool(loop.fit_targets.count_nonzero()):
        raise ValueError("effective FIT/zero labels differ from tag")
    return dict(schema=CHECKPOINT_SCHEMA,tag=deepcopy(tag),
        fit_context=loop.fit_context.detach().cpu().clone(),
        precision_checkpoint=checkpoint_precision(loop))


def restore_latent_coverage(loop, state):
    """Own-state public restore only; fresh matching owners/data already exist.

No initializer or data/config rewrite is called while restoring learned state.
Wrong seeds, pool/order/dtype/shape/precision tags reject BEFORE public load.
"""
    if (state.get("schema")!=CHECKPOINT_SCHEMA or state.get("tag")!=loop.config.get(TAG)
            or state["precision_checkpoint"]["pilot"]["config"]!=loop.config
            or not torch.equal(state["fit_context"],loop.fit_context.detach().cpu())
            or state["fit_context"].dtype!=torch.float32
            or state_digest(state["fit_context"])!=state["tag"]["fit_context_digest"]):
        raise ValueError("coverage checkpoint/pool/precision mismatch")
    _finite(loop.fit_targets,(state["tag"]["fit_rows"],4,32,32),torch.float32,"effective zero labels")
    if bool(loop.fit_targets.count_nonzero()):raise ValueError("physical residual labels must remain zero")
    restore_precision(loop,state["precision_checkpoint"])


def coverage_update(loop):
    """Paired caller addresses; the native D/G update is imported unchanged.

Both arms draw the same common960-address law from their exactly restored
CPU7 stream. Baseline collapses seed_index j to floor(j/4). This preserves
uniform original240 sampling as a marginal, not its old literal RNG sequence.
Accuracy never participates in sampling, gradients, guards or stopping.
"""
    tag=loop.config.get(TAG)
    if tag is None or tag["sampling"] != SAMPLING:
        raise ValueError("frozen coverage caller sampler required")
    seeds=tuple(tag["seeds"])
    if seeds not in (BASE_SEEDS,TRAIN_SEEDS): raise ValueError("unknown coverage support")
    with precision_rng_scope(loop),torch.autograd.set_multithreading_enabled(False):
        common=torch.randint(960,(4,),generator=loop.data_rng)
        source=common//160; seed_index=(common%160)//20; offset=common%20
        indices=common if seeds==TRAIN_SEEDS else 40*source+20*(seed_index//4)+offset
        context=loop.fit_context[indices.to(loop.fit_context.device)]
        row=update(loop,context=context,
            target=torch.zeros(4,4,32,32,device=loop.policy.device,dtype=torch.float32),
            batch_indices=indices,game_weight=1.)
    row.update(hold=False,game_weight=1.,coverage_addresses=common.tolist(),
        coverage_seeds=[seeds[int(j)] if seeds==TRAIN_SEEDS else seeds[int(j)//4] for j in seed_index],
        source_positions=source.tolist(),coverage_rows=tag["fit_rows"])
    return row
