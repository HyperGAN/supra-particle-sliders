#!/usr/bin/env python3
"""Exact late V3 replay and isolated native event-action counterfactuals.

Diagnostic laws are explicit manifests, never migrated production recipes.
No output metric, seed sweep, schedule change, or checkpoint selection occurs.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_e22_supra_particle_gated import PIN, diagnostic, emit, sha, write
from e22_supra_editing_only import make_from_saved, training_update

LATE_CLOCK = 6387                 # begin_step's completed_steps; training row6388
DRIFT_ROW = 6222                 # observe after update6222; applied LR changes6223
END = 6400
ARMS = {
    "cancel_generator_drift_release": dict(fork=6221, law="cancel_generator_drift_release6222_v1",
        description="Run the native6222 DRIFT decision, then restore only generator tester.s from1 to.5 once; later native decisions/events remain active."),
    "omit_moment_rescale": dict(fork=6387, law="omit_event6387_moment_rescale_v1",
        description="Keep the accepted detector/guard event; omit only _reopen_moments at begin6388."),
    "omit_anchor_latch": dict(fork=6387, law="omit_event6387_anchor_latch_v1",
        description="Forward _anchor_release(False) at begin6388 only; the verified absent latch stays absent, with later native calls unchanged."),
    "omit_ladder_reopens": dict(fork=6387, law="omit_event6387_ladder_reopens_v1",
        description="Keep detection/moments/anchor action; replace all five tester.restart(reopen=True) calls at begin6388 with tester.begin."),
    "omit_all_event_actions": dict(fork=6387, law="omit_event6387_all_three_actions_v1",
        description="Keep the accepted detector/guard bookkeeping, but omit moment rescaling, latch creation and all five ladder reopen calls at begin6388."),
}


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(child) for child in value)
    return deepcopy(value)


def fast_values(policy):
    """Actual trainable FAST values, safe at a transient native phase."""
    owned = {id(parameter) for optimizer in policy.optimizers
             for group in optimizer.param_groups for parameter in group["params"]}
    owners = {role: {name: parameter.detach().cpu().clone()
                    for name, parameter in module.named_parameters() if id(parameter) in owned}
              for role, module in policy._training_modules().items()}
    owners["table"] = {"table": policy.table.detach().cpu().clone()}
    if policy.log_output_sigma is not None:
        owners["noise"] = {"log_output_sigma": policy.log_output_sigma.detach().cpu().clone()}
    return owners


def action_values(policy):
    """Event owners without calling checkpoint while begin_step is active."""
    groups = {}
    for player, (optimizer, roles) in enumerate(zip(policy.optimizers, policy.roles)):
        for group_index, (group, role) in enumerate(zip(optimizer.param_groups, roles)):
            tester = policy.lr_settle.testers[player][group_index]
            groups[f"{player}.{group_index}"] = dict(role=role, lr=group["lr"],
                base_lr=policy.initial_lrs[player][group_index], raw_scale=tester.s,
                tester=cpu_copy(tester.__dict__), moments=[cpu_copy(optimizer.state.get(parameter, {}))
                    for parameter in group["params"]])
    return dict(fast=fast_values(policy), groups=groups,
        surprise=cpu_copy(policy.surprise.state_dict()), guard=cpu_copy(policy.reopen_guard.state_dict()),
        controller=cpu_copy(policy.controller.state_dict()),
        ka2_record=cpu_copy(policy.opt_d.record.__dict__))


@contextmanager
def action_hooks(loop, arm, output, require, digest):
    """Process-local native class hooks; no instance attrs or tree changes.

    Hooks select live policy/tester identity, so private structural candidate
    replays keep native actions. Both tester classes are handled; Sequential's
    super call is not captured twice or partially suppressed.
    """
    from particlegan.continuous import SettleTest, SequentialSettleTest
    policy = loop.policy
    policy_type = type(policy)
    native = {name: getattr(policy_type, name) for name in
              ("begin_step", "_reopen_moments", "_anchor_release", "_settle_observe")}
    base_restart, sequential_restart = SettleTest.restart, SequentialSettleTest.restart
    testers = {id(tester): (player, group, role) for player, (row, roles) in
               enumerate(zip(policy.lr_settle.testers, policy.roles))
               for group, (tester, role) in enumerate(zip(row, roles)) if tester is not None}
    records = []
    counts = dict(moments=0, anchor=0, ladder=0, generator_release=0)

    def record(name, before, after, *, omitted=False):
        index = len(records)
        filename = f"action-{index:02d}-{name}.pt"
        import torch
        torch.save(dict(schema="supra_native_event_action_values_v1", arm=arm,
                        action=name, before=before, after=after), output / filename)
        before_fast, after_fast = digest(before["fast"]), digest(after["fast"])
        require(f"{arm}_{name}_{index}_does_not_write_fast", before_fast == after_fast)
        records.append(dict(action=name, omitted=omitted, filename=filename, sha256=sha(output / filename),
            before_fast_digest=before_fast, after_fast_digest=after_fast,
            before_groups={key: {field: value[field] for field in ("role", "lr", "base_lr", "raw_scale")}
                           for key, value in before["groups"].items()},
            after_groups={key: {field: value[field] for field in ("role", "lr", "base_lr", "raw_scale")}
                          for key, value in after["groups"].items()}))

    def begin(owner, *args, **kwargs):
        if owner is not policy:
            return native["begin_step"](owner, *args, **kwargs)
        clock = owner.completed_steps
        watched = clock in (6221, 6222, LATE_CLOCK)
        before = action_values(owner) if watched else None
        value = native["begin_step"](owner, *args, **kwargs)
        if watched:
            after = action_values(owner)
            record(f"begin-row-{clock+1}", before, after)
        return value

    def moments(owner):
        if owner is not policy or owner.completed_steps != LATE_CLOCK:
            return native["_reopen_moments"](owner)
        before = action_values(owner)
        if arm != "cancel_generator_drift_release":
            require(arm + "_moment_hook_is_second_fire", owner.surprise.fires == 2)
        omitted = arm in ("omit_moment_rescale", "omit_all_event_actions")
        value = None if omitted else native["_reopen_moments"](owner)
        counts["moments"] += 1
        record("moments", before, action_values(owner), omitted=omitted)
        return value

    def anchor(owner, reopen):
        if owner is not policy or owner.completed_steps != LATE_CLOCK or not reopen:
            return native["_anchor_release"](owner, reopen)
        before = action_values(owner)
        if arm != "cancel_generator_drift_release":
            require(arm + "_anchor_initial_latch_absent", owner.surprise.anchor_event is None)
        omitted = arm in ("omit_anchor_latch", "omit_all_event_actions")
        value = native["_anchor_release"](owner, False if omitted else reopen)
        counts["anchor"] += 1
        record("anchor", before, action_values(owner), omitted=omitted)
        return value

    def restart(owner, params, reopen, original):
        if id(owner) not in testers or policy.completed_steps != LATE_CLOCK or not reopen:
            return original(owner, params, reopen=reopen)
        _, _, role = testers[id(owner)]
        before = action_values(policy)
        omitted = arm in ("omit_ladder_reopens", "omit_all_event_actions")
        value = owner.begin(params) if omitted else original(owner, params, reopen=reopen)
        counts["ladder"] += 1
        record("ladder-" + role, before, action_values(policy), omitted=omitted)
        return value

    def base(owner, params, reopen=False):
        if isinstance(owner, SequentialSettleTest):
            return base_restart(owner, params, reopen=reopen)
        return restart(owner, params, reopen, base_restart)

    def sequential(owner, params, reopen=False):
        return restart(owner, params, reopen, sequential_restart)

    def observe(owner, player):
        watched = owner is policy and player == 0 and owner.completed_steps + 1 == DRIFT_ROW
        tester = owner.lr_settle.testers[0][0] if watched else None
        before_scale = tester.s if watched else None
        before = action_values(owner) if watched else None
        value = native["_settle_observe"](owner, player)
        if watched:
            require(arm + "_native_generator_release6222", before_scale == .5 and tester.s == 1.
                    and tester.last["decision"] == "drift" and tester.last["step"] == DRIFT_ROW)
            omitted = arm == "cancel_generator_drift_release"
            if omitted:
                tester.s = before_scale
            counts["generator_release"] += 1
            record("generator-release6222", before, action_values(owner), omitted=omitted)
        return value

    policy_type.begin_step = begin
    policy_type._reopen_moments = moments
    policy_type._anchor_release = anchor
    policy_type._settle_observe = observe
    SettleTest.restart, SequentialSettleTest.restart = base, sequential
    try:
        yield dict(records=records, counts=counts)
    finally:
        for name, value in native.items():
            setattr(policy_type, name, value)
        SettleTest.restart, SequentialSettleTest.restart = base_restart, sequential_restart


def game_evaluation(loop, data, critics, *, full, require, digest):
    """Learned RpGAN only, with private immutable panels and FAST owners.

    Complete evaluation reproduces the qualified endpoint's native panels.
    Small before/after panels always consume the complete panel program first,
    then select two fixed contexts per subject; no output errors are scored.
    """
    import torch
    from supra.particle_game import patchify
    from supra.particle_pilot import checkpoint
    from monitor_e22_supra_particle_convergence import evaluation_modes
    before = digest(checkpoint(loop))
    policy = loop.policy
    stream = torch.Generator(device=policy.device).manual_seed(72)
    candidate = policy.routed_control.candidate()
    spec, models = policy.routed_control.spec, policy._training_modules()
    result = dict(step=policy.completed_steps, source="FAST current particle G", full_panel=full,
                  output_sigma=.125, output_metrics_used=False, evaluation_only=True, pools={})
    with torch.no_grad(), evaluation_modes(policy):
        for name in ("fit", "test", "holds", "preservation"):
            pool = data[name]["context"]
            selected = set(range(len(pool))) if full else set()
            if not full:
                for subject in pool[:, 4097].unique(sorted=True):
                    available = (pool[:, 4097] == subject).nonzero().flatten()
                    selected.update(available[[0, len(available)-1]].tolist())
            records = []
            for start in range(0, len(pool), 4):
                length = min(4, len(pool)-start)
                bases = .125 * torch.randn((4, length, 256, 16), device=policy.device,
                                            dtype=torch.float32, generator=stream)
                if not any(index in selected for index in range(start, start+length)):
                    continue
                context = pool[start:start+length].to(policy.device)
                prediction = spec.forward(models, context, candidate)
                condition = policy.encoder.condition(context).repeat(4, 1)
                real = bases.flatten(0, 1)
                scores = {}
                for label, critic in critics.items():
                    fake = (bases + (patchify(prediction).float()/critic.scale).unsqueeze(0)).flatten(0, 1)
                    gap = critic(real, condition)-critic(fake, condition)
                    scores[label] = dict(g_game=torch.nn.functional.softplus(gap).reshape(4,length).mean(0),
                        d_game=torch.nn.functional.softplus(-gap).reshape(4,length).mean(0),
                        score_gap=gap.reshape(4,length).mean(0))
                for row in range(length):
                    if start+row not in selected:
                        continue
                    records.append(dict(index=start+row, source_caption_id=int(context[row,4097]),
                        time=float(context[row,4096]), **{label:{key:float(value[row]) for key,value in values.items()}
                            for label,values in scores.items()}))
            result["pools"][name] = dict(count=len(records), records=records, **{
                label:{key:sum(record[label][key] for record in records)/len(records)
                       for key in ("g_game","d_game","score_gap")} for label in critics})
    require(f"game{policy.completed_steps}_native_immutable", digest(checkpoint(loop)) == before)
    result["native_state_unchanged"] = True
    result["native_digest"] = before
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"outputs/e22-particle-gated-v3-editing-only-6400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent/"ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/e22-particle-gated-late-reopen-causal")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("preserve prior artifacts; choose a fresh output directory")
    pg = args.particlegan_root.resolve()
    if subprocess.check_output(["git","-C",str(pg),"rev-parse","HEAD"],text=True).strip() != PIN:
        raise RuntimeError("unexpected native pin")
    if subprocess.check_output(["git","-C",str(pg),"status","--porcelain","--","particlegan"],text=True).strip():
        raise RuntimeError("native package source is dirty")
    sys.path.insert(0,str(pg))
    import torch
    import particlegan
    from supra.runtime import TARGETS, model_module
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_game import ConditionalTokenCritic
    from review_e22_supra_pr223 import committed_python_hashes
    if Path(particlegan.__file__).resolve().parent.parent != pg:
        raise RuntimeError("unexpected native import")
    device=torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("actual native BF16 replay requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    started=time.perf_counter()
    checks={}
    def require(name, value):
        checks[name]=bool(value)
        if not checks[name]:
            write(args.output/"failed-checks.json",checks)
            raise RuntimeError(name)
    review=json.loads((args.run/"independent-review.json").read_text())
    parent_plan=json.loads((args.run/"plan.json").read_text())
    require("qualified_parent",review["qualified"] is True and review["fixed_updates"]==END)
    inputs=dict(initial=args.run/"checkpoint-06000.pt", final=args.run/"final.pt",
        trace=args.run/"train.jsonl", parent_review=args.run/"independent-review.json",
        parent_plan=args.run/"plan.json", parent_receipt=args.run/"receipt.json",
        final_evaluation=args.run/"evaluation-v3.json",
        data=Path(parent_plan["input_paths"]["data"]), judge=Path(parent_plan["input_paths"]["judge"]),
        control=Path(parent_plan["input_paths"]["control"]))
    input_sha={key:sha(path) for key,path in inputs.items()}
    for name in ("checkpoint-06000.pt","final.pt","train.jsonl","evaluation-v3.json"):
        require("qualified_input_"+name,sha(args.run/name)==review["artifact_sha256"][name])
    for name in ("plan.json","receipt.json"):
        require("qualified_parent_"+name,sha(args.run/name)==review["artifact_sha256"][name])
    for name in ("data","judge","control"):
        require("qualified_auxiliary_"+name,input_sha[name]==parent_plan["input_sha256"][name])
    for name,digest in parent_plan["application_source_sha256"].items():
        require("held_application_"+name,sha(ROOT/name)==digest)
    native_sources=committed_python_hashes(pg,PIN)
    require("qualified_native_tree",native_sources==parent_plan["particlegan_source_sha256"])
    sources={**parent_plan["application_source_sha256"],str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name,digest in sources.items():
        target=args.output/"source"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        require("snapshot_"+name,sha(target)==digest)
    for name,digest in native_sources.items():
        target=args.output/"source/native"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(pg/name,target)
        require("native_snapshot_"+name,sha(target)==sha(pg/name)==digest)
    plan=dict(schema="supra_late_native_reopen_causal_v1",native_pin=PIN,start_step=6000,
        fixed_endpoint=END, drift_decision_row=DRIFT_ROW, late_completed_clock=LATE_CLOCK,
        late_first_affected_row=LATE_CLOCK+1, arms=ARMS, schedule="editing_only_v1",
        input_paths={key:str(path.resolve()) for key,path in inputs.items()},input_sha256=input_sha,
        application_source_sha256=sources,native_source_sha256=native_sources,
        judge_labels=dict(fixed_start_D="fixed common D1856",arm_final_D="fixed common historical V2 D6400; never an intervention arm's evolving critic"),
        raw_output_scoring=False,checkpoint_selection=False,
        native_recipe_changes=False,core_source_changes=False)
    write(args.output/"plan.json",plan)
    emit(event="plan",fixed_endpoint=END,arms=list(ARMS),event_row=LATE_CLOCK+1)
    load=lambda path:torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    initial=load(inputs["initial"])
    expected_final=load(inputs["final"])
    data=load(inputs["data"])
    original=[json.loads(line) for line in inputs["trace"].read_text().splitlines() if line.strip()]
    require("initial_native_digest",state_digest(initial)==review["checkpoints"]["6000"]["native_digest"])
    require("initial_clock",initial["policy"]["completed_steps"]==6000)
    require("final_native_digest",state_digest(expected_final)==review["final_native_digest"])
    teacher={name.removeprefix("teacher."):value for name,value in initial["policy"]["models"]["encoder"].items()
             if name.startswith("teacher.")}
    with torch.random.fork_rng(devices=[]),torch.device("meta"):
        module=model_module()
        base=module.SupraDiT()
        module.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
    base.load_state_dict(teacher,strict=True,assign=True)
    base.to(device).eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[device.index or 0]):
        loop=make_from_saved(base,data,initial,device=device)
        critics={label:ConditionalTokenCritic(data["coordinate_scale"]).to(device)
                 for label in ("fixed_start_D","arm_final_D")}
    for label,path in (("fixed_start_D",inputs["judge"]),("arm_final_D",inputs["control"])):
        saved=load(path)
        critics[label].load_state_dict(saved["policy"]["models"]["critic"],strict=True)
        critics[label].eval().requires_grad_(False)
    plan["critic_digests"]={label:state_digest(critic.state_dict()) for label,critic in critics.items()}
    require("qualified_fixed_D1856",plan["critic_digests"]["fixed_start_D"]==parent_plan["critic_digests"]["D1856"])
    require("qualified_fixed_V2_D6400",plan["critic_digests"]["arm_final_D"]==parent_plan["critic_digests"]["control_final"])
    write(args.output/"plan.json",plan)
    restore(loop,initial)
    require("strict6000_restore",state_digest(checkpoint(loop))==state_digest(initial))
    frozen=frozen_digest(loop)
    forks={}
    results={}

    def save_native(path, law, fork_digest):
        state=cpu_copy(checkpoint(loop))
        torch.save(dict(schema="supra_native_event_diagnostic_state_v1",experimental_law=law,
                        origin_native_digest=fork_digest,native=state),path)
        return dict(path=path.name,file_sha256=sha(path),native_digest=state_digest(state))

    def run_arm(arm,start_state,*,baseline=False):
        directory=args.output/arm
        directory.mkdir()
        restore(loop,start_state)
        origin=state_digest(start_state)
        require(arm+"_exact_fork",state_digest(checkpoint(loop))==origin)
        rows=[]
        games={}
        ready_snapshots={}
        def snapshot_ready():
            step=loop.policy.completed_steps
            filename=f"ready-{step:05d}.pt"
            payload=dict(schema="supra_native_event_ready_values_v1",arm=arm,step=step,
                experimental_law="native_replay_v1" if baseline else ARMS[arm]["law"],
                phase=loop.policy._phase,values=action_values(loop.policy))
            torch.save(payload,directory/filename)
            ready_snapshots[str(step)]=dict(filename=filename,sha256=sha(directory/filename),
                fast_digest=state_digest(payload["values"]["fast"]))
        if loop.policy.completed_steps in (6221,6387):
            snapshot_ready()
        with action_hooks(loop,arm,directory,require,state_digest) as actions:
            with (directory/"train.jsonl").open("w") as trace:
                for step in range(loop.policy.completed_steps+1,END+1):
                    if baseline and loop.policy.completed_steps in (6221,6387):
                        fork=cpu_copy(checkpoint(loop))
                        forks[loop.policy.completed_steps]=fork
                        torch.save(fork,args.output/f"fork-{loop.policy.completed_steps:05d}.pt")
                    if baseline and loop.policy.completed_steps in (6221,6222,6387):
                        value=game_evaluation(loop,data,critics,full=False,require=require,digest=state_digest)
                        games[str(loop.policy.completed_steps)]=value
                        write(directory/f"game-{loop.policy.completed_steps:05d}.json",value)
                    row=training_update(loop)
                    reference=original[step-1]
                    if baseline:
                        require("baseline_row_"+str(step),row==reference)
                    fields=("hold","game_weight","batch_indices","base_noise_sums","paired_rng_digest","dv12_rng_digest")
                    require(arm+"_streams_"+str(step),all(row[name]==reference[name] for name in fields))
                    rows.append(row)
                    trace.write(json.dumps(row,allow_nan=False)+"\n")
                    trace.flush()
                    if step in (6221,6222,6223,6387,6388,END):
                        snapshot_ready()
                    if baseline and step in (6223,6388):
                        value=game_evaluation(loop,data,critics,full=False,require=require,digest=state_digest)
                        games[str(step)]=value
                        write(directory/f"game-{step:05d}.json",value)
                    if step%25==0 or step in (6222,6223,6388,END):
                        emit(event="training",arm=arm,**row)
                        write(args.output/"status.json",dict(phase="training",arm=arm,step=step,fixed_endpoint=END))
        require(arm+"_class_hooks_restored",not any(name in loop.policy.__dict__ for name in
            ("begin_step","_reopen_moments","_anchor_release","_settle_observe")))
        require(arm+"_frozen_owners",frozen_digest(loop)==frozen)
        require(arm+"_no_structural_moves",loop.policy.routed_control.counters["moves"]==0)
        final=checkpoint(loop)
        if baseline:
            require("baseline_full_native_exact6400",state_digest(final)==state_digest(expected_final))
            require("qualified_final_served_FAST",final["policy"]["served_source"]==expected_final["policy"]["served_source"]=="fast")
        evaluation=game_evaluation(loop,data,critics,full=True,require=require,digest=state_digest)
        write(directory/"game-06400.json",evaluation)
        if baseline:
            prior=json.loads(inputs["final_evaluation"].read_text())
            for pool,value in evaluation["pools"].items():
                for got,expected in zip(value["records"],prior[pool]["records"]):
                    require("baseline_game_"+pool+"_"+str(got["index"]),all(got[label]==expected[label] for label in critics))
        result=dict(arm=arm,experimental_law="native_replay_v1" if baseline else ARMS[arm]["law"],
            fork_step=start_state["policy"]["completed_steps"],origin_native_digest=origin,
            rows=len(rows),trace_sha256=sha(directory/"train.jsonl"),actions=actions,
            ready_snapshots=ready_snapshots,
            final=save_native(directory/"final-diagnostic.pt","native_replay_v1" if baseline else ARMS[arm]["law"],origin),
            full_game=evaluation,game_sha256=sha(directory/"game-06400.json"),
            surprise=diagnostic(loop.policy.surprise.state_dict()),
            guard=diagnostic(loop.policy.reopen_guard.state_dict()),
            actual_rates=[[group["lr"] for group in optimizer.param_groups] for optimizer in loop.policy.optimizers],
            frozen_unchanged=True,all_recorded_input_noise_streams_equal=True,structural_moves=0)
        write(directory/"result.json",result)
        emit(event="arm_complete",arm=arm,test_game={label:evaluation["pools"]["test"][label]["g_game"] for label in critics})
        return result

    results["native"]=run_arm("native",initial,baseline=True)
    require("both_forks_captured",set(forks)=={6221,6387})
    require("pre_event_latch_absent",forks[6387]["policy"]["surprise"]["anchor_event"] is None)
    require("baseline_event_hooks",results["native"]["actions"]["counts"]==dict(moments=1,anchor=1,ladder=5,generator_release=1))
    for arm,definition in ARMS.items():
        results[arm]=run_arm(arm,forks[definition["fork"]])
        count=results[arm]["actions"]["counts"]
        require(arm+"_target_hooks_once",count["generator_release"]==(1 if definition["fork"]==6221 else 0))
        if arm!="cancel_generator_drift_release":
            require(arm+"_all_target_hooks",count["moments"]==count["anchor"]==1 and count["ladder"]==5)
    for name,path in inputs.items():
        require("input_unchanged_"+name,sha(path)==input_sha[name])
    for name,digest in sources.items():
        require("source_unchanged_"+name,sha(ROOT/name)==digest)
    write(args.output/"receipt.json",dict(schema="supra_late_native_reopen_causal_v1",qualified=True,
        plan=plan,checks=checks,results=results,forks={str(step):dict(path=f"fork-{step:05d}.pt",
            file_sha256=sha(args.output/f"fork-{step:05d}.pt"),native_digest=state_digest(state))
            for step,state in forks.items()},total_seconds=time.perf_counter()-started,
        raw_output_scoring=False,checkpoint_selection=False,
        limits="One fixed native trajectory and diagnostic action counterfactuals. Later guard/optimizer decisions may diverge as consequences; input/Gaussian/DV12 streams remain matched. Baseline exactness is required before intervention. No ordinary-LoRA optimization or production-law change."))
    write(args.output/"status.json",dict(phase="complete",step=END,arms=list(results),qualified=True))
    emit(event="complete",qualified=True,seconds=time.perf_counter()-started)


if __name__=="__main__":
    main()
