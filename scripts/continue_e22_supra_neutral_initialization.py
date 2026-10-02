#!/usr/bin/env python3
"""Fixed paired editing-only continuation, with public native recovery first.

No quality decision changes the 9600/12800 endpoints, optimizer, or stopping.
--preflight-only checks actual files/configs on CPU without loading CUDA.
Use the predeclared outer timeout as well as the internal charged budgets.
"""
import time
SESSION_STARTED = time.monotonic()

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e22_supra_neutral_continuation_contract import (ARMS, CARD, ENDPOINTS, LOCAL_CHECKPOINTS,
    MODES, PIN, START, STOP, bind_helpers, canonical, exact_recovery, held_manifest,
    ordinary_metadata_contract, paired_row_contract, require_checkpoint_contract, sha)
import experiment_e22_supra_neutral_initialization as held
from experiment_e22_supra_particle_gated import write, emit, diagnostic


def preflight(card_path=CARD, pg_root=None):
    """Full byte binding, then mmap scalar config checks; no model or CUDA calls."""
    started = time.monotonic()
    protocol = json.loads(Path(card_path).read_text())
    pg = Path(pg_root or protocol["particlegan"]["root"]).resolve()
    if pg!=Path(protocol["particlegan"]["root"]).resolve():
        raise ValueError("native checkout path changed")
    if protocol["schema"] != "supra_neutral_continuation_protocol_v1":
        raise ValueError("wrong continuation card")
    expected = dict(start=START, stop=STOP, additional_updates_per_arm=STOP-START,
                    endpoints=list(ENDPOINTS), local_checkpoints=list(LOCAL_CHECKPOINTS),
                    preservation_updates=0, output_objective=False, output_error_guard=False,
                    max_feature_context_harm=0, optimizer_interventions=False)
    if protocol["training"] != expected:
        raise ValueError("continuation training law changed")
    if protocol["budget"] != dict(gpu_seconds=14400, charged_seconds_per_arm=7200,
        external_cpu_review_seconds=1200, total_seconds_including_review=15600,
        disk_free_bytes_required=128*1024**3,
        scope="Startup, input/source qualification, restore/replay, all updates, checkpoints, probes, fixed references, endpoints, exports, and final writes; separate external CPU qualification."):
        raise ValueError("fixed cost law changed")
    if subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("held native HEAD mismatch")
    if subprocess.check_output(["git", "-C", str(pg), "status", "--porcelain", "--", "particlegan"], text=True).strip():
        raise ValueError("held native source dirty")
    cached = {}
    def actual(path):
        key = str(Path(path).resolve())
        if key not in cached:
            cached[key] = sha(key)
        return cached[key]
    for name, digest in protocol["sources"].items():
        if actual(ROOT / name) != digest:
            raise ValueError("held/new source bytes changed: " + name)
    if held_manifest(held) != protocol["orchestration_adapter"]:
        raise ValueError("held helper binding manifest changed")
    parent = Path(protocol["parent"]["run"])
    parent_plan = json.loads((parent / "plan.json").read_text())
    review = json.loads((parent / "independent-review.json").read_text())
    if not (review["qualified"] is True and review["partial"] is False
            and review["common_matched_rows"] == START
            and actual(parent / "independent-review.json") == protocol["parent"]["review_sha256"]
            and actual(parent / "plan.json") == protocol["parent"]["plan_sha256"]
            and actual(parent / "receipt.json") == protocol["parent"]["receipt_sha256"]):
        raise ValueError("parent qualification mismatch")
    for name, digest in parent_plan["application_source_sha256"].items():
        if actual(ROOT / name) != digest or actual(parent / "source" / name) != digest:
            raise ValueError("qualified parent source/archive changed: " + name)
    for name, digest in parent_plan["particlegan_source_sha256"].items():
        if actual(pg / name) != digest or actual(parent / "source/native" / name) != digest:
            raise ValueError("qualified native source/archive changed: " + name)
    for name, entry in protocol["inputs"].items():
        if actual(entry["path"]) != entry["sha256"]:
            raise ValueError("immutable actual input changed: " + name)
    for name,entry in parent_plan["protocol"]["inputs"].items():
        if actual(entry["path"])!=entry["sha256"]:
            raise ValueError("inherited protocol input changed: "+name)
    for arm in ARMS:
        record = protocol["parent"]["checkpoints"][arm]
        witness = review["checkpoint_artifacts"][arm][str(START)]
        if record["sha256"] != witness["file_sha256"] or record["native_digest"] != witness["native_digest"]:
            raise ValueError("6400 input does not match qualified parent: " + arm)
        if actual(record["path"]) != record["sha256"]:
            raise ValueError("actual 6400 native file changed: " + arm)
        initial=protocol["inputs"]["parent_initial_"+arm]
        if initial["sha256"]!=review["checkpoint_artifacts"][arm]["0"]["file_sha256"]:
            raise ValueError("fixed monitor DV12 initial owner is not parent-qualified")
    probe_metadata=json.loads(Path(protocol["inputs"]["parent_probe_metadata"]["path"]).read_text())
    if json.loads(Path(protocol["inputs"]["parent_probe_indices"]["path"]).read_text()) != probe_metadata["indices"]:
        raise ValueError("fixed monitor cohort mismatch")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("source/input preflight must not load CUDA")
    for arm in ARMS:
        state = torch.load(protocol["parent"]["checkpoints"][arm]["path"], mmap=True,
                           map_location="cpu", weights_only=False)
        require_checkpoint_contract(state, parent_plan["configs"][arm], START)
        if state["config"]["particle_init"] != MODES[arm]:
            raise ValueError("trained initializer mode changed")
        del state
        gc.collect()
    from safetensors import safe_open
    expected_host=parent_plan["protocol"]["host"]
    for step in (6400,12800):
        with safe_open(protocol["inputs"]["original"+str(step)]["path"],framework="pt",device="cpu") as handle:
            metadata=handle.metadata()
        ordinary_metadata_contract(metadata,expected_host,step,protocol["inputs"]["ordinary_prompts"]["sha256"])
        state=torch.load(protocol["inputs"]["ordinary_training_state_"+str(step)]["path"],
            mmap=True,map_location="cpu",weights_only=False)
        if state["step"]!=step or not {"optimizer","sampling_rng","torch_rng","cuda_rng"}.issubset(state):
            raise ValueError("historical ordinary training state provenance changed")
        del state
    historical_rows=[]
    for name in ("ordinary_first400_trace","ordinary_continuation_trace"):
        historical_rows.extend(json.loads(line) for line in Path(protocol["inputs"][name]["path"]).read_text().splitlines()
            if json.loads(line)["step"]<=12800)
    if [row["step"] for row in historical_rows]!=list(range(1,12801)) or any(row["hold"]!=(row["step"]%5==0) for row in historical_rows):
        raise ValueError("historical ordinary editing/preservation budget is not fully witnessed")
    if shutil.disk_usage(ROOT).free < protocol["budget"]["disk_free_bytes_required"]:
        raise ValueError("128 GiB free-disk gate failed")
    return dict(schema="supra_neutral_continuation_cpu_preflight_v1", ready=True,
                quality_updates=0, model_forward_calls=0, cuda_initialized=False,
                protocol_sha256=actual(card_path), actual_file_sha256=cached,
                parent_review_sha256=protocol["parent"]["review_sha256"],
                orchestration_adapter=protocol["orchestration_adapter"], seconds=time.monotonic()-started)


def run(args, protocol, readiness):
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from safetensors.torch import load_file
    from supra.runtime import TARGETS, model_module, load_adapter_state
    from supra.particle_adapter import GATED_PARTICLE_V3
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic, patchify, update
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import SHARED_ROUTED_PROFILE, make_training_loop, raw_model_forward, features_for_rows
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, evaluation_modes, rolling_losses
    from evaluate_e22_supra_particle_contribution import mass_only_routing
    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise ValueError("wrong native import")
    device = torch.device(args.device)
    if str(device)!="cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("fixed continuation uses physical GPU0 via CUDA_VISIBLE_DEVICES=0 and cuda:0")
    if not torch.cuda.is_available():
        raise ValueError("held BF16 Supra requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    charged = {arm: 0. for arm in ARMS}
    checks, loops, traces = {}, {}, {}
    started = SESSION_STARTED
    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    def budget():
        if time.monotonic()-started > 14400 or any(value > 7200 for value in charged.values()):
            raise TimeoutError("fixed continuation cost budget exhausted")
    def report(phase, step, **extra):
        write(output / "status.json", dict(phase=phase, step=step, steps=STOP,
            editing_updates=step, preservation_updates=0, continuation_editing_updates=max(0,step-START),
            seconds=time.monotonic()-started,
            arms={arm: dict(charged_seconds=charged[arm], **extra.get(arm, {})) for arm in ARMS}))
    parent = Path(protocol["parent"]["run"])
    parent_plan = json.loads((parent / "plan.json").read_text())
    if torch.__version__!=parent_plan["torch_version"] or torch.cuda.get_device_name(device)!=parent_plan["gpu"]:
        raise ValueError("qualified parent Torch/GPU host changed")
    inherited = parent_plan["protocol"]
    inputs = {name: Path(entry["path"]) for name, entry in protocol["inputs"].items()}
    sources = dict(parent_plan["application_source_sha256"], **protocol["sources"])
    sources[str(CARD.relative_to(ROOT))] = sha(CARD)
    plan = dict(schema="supra_neutral_continuation_plan_v1", protocol=protocol,
        protocol_sha256=sha(CARD), parent_run=str(parent), parent_plan_sha256=sha(parent / "plan.json"),
        fixed_updates=STOP, additional_updates=STOP-START, start_step=START, endpoints=list(ENDPOINTS),
        input_paths={key: str(path) for key,path in inputs.items()},
        input_sha256={key: protocol["inputs"][key]["sha256"] for key in inputs},
        application_source_sha256=sources, particlegan_source_sha256=parent_plan["particlegan_source_sha256"],
        particlegan_commit=PIN, configs=parent_plan["configs"], device=str(device),
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        primary_target="ordinary_lora_6400", secondary_budget_context="ordinary_lora_12800",
        initialization_after_restore=False)
    write(output / "plan.json", plan)
    write(output / "cpu-preflight.json", readiness)
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name in plan["particlegan_source_sha256"]:
        destination = output / "source/native" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.particlegan_root / name, destination)
    try:
        report("preparing", START)
        data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
        require("inherited_data", state_digest(data) == inherited["data"]["digest"])
        teacher = torch.load(inputs["teacher"], map_location="cpu", weights_only=False, mmap=True)
        teacher_state = {key.removeprefix("teacher."):value for key,value in teacher["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            module = model_module()
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher_state, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        del teacher, teacher_state
        gc.collect()
        owners = dict(torch=torch, F=F, device=device, require=require, update=update,
            checkpoint=checkpoint, state_digest=state_digest, frozen_digest=frozen_digest,
            patchify=patchify, protocol=inherited, base=base, data=data, mode_names=MODES,
            GATED_PARTICLE_V3=GATED_PARTICLE_V3, SHARED_ROUTED_PROFILE=SHARED_ROUTED_PROFILE,
            make_training_loop=make_training_loop, load_file=load_file, load_adapter_state=load_adapter_state,
            control=None, inputs=inputs, judges=None, probe_context=None, probe_panels=None)
        helpers = bind_helpers(held, owners)
        fresh, all_edit = helpers["fresh"], helpers["all_edit"]
        def fresh_restore(arm):
            loop = fresh(arm) # initialize_ belongs only to this transient constructor.
            state = torch.load(protocol["parent"]["checkpoints"][arm]["path"],
                               map_location="cpu", weights_only=False, mmap=True)
            require_checkpoint_contract(state, plan["configs"][arm], START)
            restore(loop, state) # strict public native restore; no subsequent initializer.
            require("exact_initial_restore_"+arm,
                state_digest(checkpoint(loop)) == protocol["parent"]["checkpoints"][arm]["native_digest"])
            del state
            gc.collect()
            return loop
        recovery = {}
        for arm in ARMS:
            tick=time.monotonic()
            with torch.random.fork_rng(devices=[device.index or 0]):
                recovery[arm]=exact_recovery(lambda: fresh_restore(arm), all_edit, checkpoint, state_digest)
            recovery[arm]["serialized_row_digest"]=state_digest(diagnostic(recovery[arm]["rows"]))
            charged[arm]+=time.monotonic()-tick
            budget()
        write(output / "recovery-06400-06402.json", recovery)
        for arm in ARMS:
            tick=time.monotonic()
            loops[arm]=fresh_restore(arm)
            (output / arm).mkdir()
            charged[arm]+=time.monotonic()-tick
        control=loops[ARMS[0]]
        judges={}
        for name in ("D1856", "D6400"):
            state=torch.load(inputs[name], map_location="cpu", weights_only=False, mmap=True)
            with torch.random.fork_rng(devices=[device.index or 0]):
                judge=ConditionalTokenCritic(data["coordinate_scale"]).to(device)
            judge.load_state_dict(state["policy"]["models"]["critic"], strict=True)
            key="D1856" if name=="D1856" else "V2D6400"
            require("fixed_judge_"+name,state_digest(judge.state_dict())==inherited["evaluation"]["judges"][key]["critic_tensor_digest"])
            judges[name]=judge.eval().requires_grad_(False)
            del state
        write(output / "judges.json", {key:state_digest(value.state_dict()) for key,value in judges.items()})
        indices=json.loads(inputs["parent_probe_indices"].read_text())
        probe_context=data["test"]["context"][indices].to(device)
        probe_panels=torch.randn(4,len(indices),256,16,generator=torch.Generator().manual_seed(72)).to(device)
        owners.update(control=control,judges=judges,probe_context=probe_context,probe_panels=probe_panels)
        helpers=bind_helpers(held,owners)
        FastReference, full_evaluation=helpers["FastReference"],helpers["full_evaluation"]
        frozen={arm:helpers["immutable_owners"](loop) for arm,loop in loops.items()}
        initial_bank={arm:state_digest(loop.policy.table) for arm,loop in loops.items()}
        initial_router={arm:state_digest(loop.policy.router.state_dict()) for arm,loop in loops.items()}
        references,reference_probes={},{}
        for step in (6400,12800):
            bindings=dict(owners,inputs=dict(inputs,original6400=inputs["original"+str(step)]))
            ordinary=bind_helpers(held,bindings)["OrdinaryReference"]()
            name="ordinary_lora_"+str(step)
            references[name]=full_evaluation(ordinary,data,judges)
            reference_probes[name]=helpers["clean_probe"](ordinary)
            del ordinary
            gc.collect()
        require("primary6400_reference_exact",references["ordinary_lora_6400"]==json.loads((parent / "historical-references.json").read_text())["ordinary_lora_6400"])
        write(output / "historical-references.json", references)
        (output / "historical-probes").mkdir()
        write(output / "historical-probes/references.json",dict(scores=reference_probes, indices=indices,
            context_digest=state_digest(probe_context),panel_digest=state_digest(probe_panels),
            primary="ordinary_lora_6400",secondary="ordinary_lora_12800",
            meaning="Fixed historical MSE/AdamW models; 6400 has5120 editing+1280 preservation, 12800 has10240 editing+2560 preservation; continuation has editing only."))
        rows={arm:[] for arm in ARMS}
        coverage={arm:dict(live_bank_updates=0,dense_128_row_updates=0,moves=0) for arm in ARMS}
        monitors={}
        for arm,loop in loops.items():
            monitor=GameProgressMonitor(loop,data,judges["D1856"],output / arm / "monitor")
            monitor.pools={"edit":(probe_context,probe_panels)}
            monitor.plot=lambda:None
            # The original monitor cohort began at update0; do not reset DV12 to6400.
            initial=torch.load(inputs["parent_initial_"+arm],map_location="cpu",weights_only=False,mmap=True)
            monitor.dv12_state=initial["policy"]["streams"]["noise_generator"].to(device="cpu").clone()
            del initial
            write(output / arm / "monitor/progress-edit-indices.json",indices)
            monitors[arm]=monitor
            traces[arm]=(output / arm / "train.jsonl").open("w",buffering=1)
            with evaluation_modes(loop.policy):
                monitor.evaluate(loop,[])
        shared=time.monotonic()-started-sum(charged.values())
        for arm in ARMS: charged[arm]+=shared/2
        write(output / "shared-preparation.json",dict(seconds=shared,charged_equally_to_arms=True))
        endpoints={}
        for step in range(START+1,STOP+1):
            current={}
            for arm,loop in loops.items():
                tick=time.monotonic()
                row=all_edit(loop)
                rows[arm].append(row);current[arm]=row
                traces[arm].write(json.dumps(diagnostic(row),allow_nan=False)+"\n")
                coverage[arm]["live_bank_updates"]+=int(row["dense_gradient_rows"]>0)
                coverage[arm]["dense_128_row_updates"]+=int(row["dense_gradient_rows"]==128)
                coverage[arm]["moves"]+=int((row.get("move") or {}).get("moves",0))
                if step==6402:
                    require("authoritative_recovery_rows_"+arm,state_digest(rows[arm][:2])==recovery[arm]["row_digest"])
                    require("authoritative_recovery_state_"+arm,state_digest(checkpoint(loop))==recovery[arm]["native_digest"])
                if step in LOCAL_CHECKPOINTS:
                    require("frozen_owners_"+arm,helpers["immutable_owners"](loop)==frozen[arm])
                    helpers["save_state"](loop,output / arm / f"checkpoint-{step:05d}.pt")
                if step%200==0:
                    probe=monitors[arm].evaluate(loop,rows[arm][-100:])
                    emit(event="progress",arm=arm,step=step,test_probe=probe["probes"]["edit"]["clean"]["frozen_start_D"],
                         loss_g=row["loss_g"],loss_d_game=row["loss_d_game"],sigma=row["output_sigma"],seconds=time.monotonic()-started)
                if step in ENDPOINTS:
                    before=state_digest(checkpoint(loop))
                    with evaluation_modes(loop.policy):
                        result=full_evaluation(FastReference(loop),data,judges)
                        ablated=full_evaluation(FastReference(loop,ablate=True),data,judges)
                        with mass_only_routing(loop.policy.router) as calls:
                            mass_only=full_evaluation(FastReference(loop),data,judges)
                    require("endpoint_native_immutable_"+arm+str(step),state_digest(checkpoint(loop))==before)
                    require("mass_only_all_sites_"+arm+str(step),calls[0]>=71)
                    write(output / arm / f"evaluation-{step:05d}.json",result)
                    write(output / arm / f"particle-ablations-{step:05d}.json",dict(code_scores=ablated,mass_only_scores=mass_only,
                        zero_code_minus_live_test_game={name:ablated["test"][name]-result["test"][name] for name in judges},
                        mass_only_minus_live_test_game={name:mass_only["test"][name]-result["test"][name] for name in judges}))
                    endpoints[arm+"@"+str(step)]=result["test"]
                    snapshot=SimpleNamespace(generator=loop.policy.G,router=loop.policy.router,table=loop.policy.table,
                        encoder=loop.policy.encoder,models=loop.policy._training_modules(),routing=loop.policy.routed_control.spec,
                        source="fast",completed_steps=step)
                    export_path=output / arm / f"adapter-{step:05d}.safetensors"
                    export_served_adapter(snapshot,export_path,extra_metadata=dict(particle_init=MODES[arm],training_schedule="fresh_editing_only_v1"))
                    with torch.random.fork_rng(devices=[device.index or 0]):
                        loaded=load_particle_adapter(base,export_path,device=device)
                    with torch.no_grad(),evaluation_modes(loop.policy):
                        context=data["test"]["context"][:4].to(device)
                        z,t,ctx,mask,uctx,umask,strength=loop.policy.encoder.unpack(context)
                        actual=loaded.velocity(z,t,ctx,mask,uctx,umask,strength=strength)
                        raw=particlegan.RoutedRows(model_forward=raw_model_forward,features=features_for_rows,sites=loop.policy.G.sites)
                        expected=raw.forward(loop.policy._training_modules(),context,loop.policy.routed_control.candidate())
                    require("export_reload_raw_exact_"+arm+str(step),torch.equal(actual,expected))
                    require("export_reload_immutable_"+arm+str(step),state_digest(checkpoint(loop))==before)
                    del loaded
                    gc.collect()
                charged[arm]+=time.monotonic()-tick
            paired_row_contract(current[ARMS[0]],current[ARMS[1]],step)
            if step%25==0:
                report("training",step,**{arm:dict(rolling=rolling_losses(rows[arm][-100:]),coverage=coverage[arm]) for arm in ARMS})
            budget()
        final_tick=time.monotonic()
        particles={}
        for arm,loop in loops.items():
            particles[arm]=dict(bank_changed=state_digest(loop.policy.table)!=initial_bank[arm],
                router_changed=state_digest(loop.policy.router.state_dict())!=initial_router[arm],
                C_norms=[float(branch.bridge.weight[:,16:].detach().norm()) for branch in loop.policy.G.particle_branches()],
                H_b_trainable=all(branch.bridge.weight.requires_grad and branch.bridge.bias.requires_grad for branch in loop.policy.G.particle_branches()))
            require("particles_retained_"+arm,particles[arm]["bank_changed"] and particles[arm]["router_changed"]
                and particles[arm]["H_b_trainable"] and all(x>0 for x in particles[arm]["C_norms"]))
        require("sources_unchanged",all(sha(ROOT/name)==value for name,value in sources.items()))
        require("native_unchanged",all(sha(args.particlegan_root/name)==value for name,value in plan["particlegan_source_sha256"].items()))
        require("inputs_unchanged",all(sha(path)==plan["input_sha256"][name] for name,path in inputs.items()))
        final_digests={arm:state_digest(checkpoint(loop)) for arm,loop in loops.items()}
        final_hashes={arm:sha(output / arm / "checkpoint-12800.pt") for arm in ARMS}
        for stream in traces.values(): stream.close()
        remainder=time.monotonic()-started-sum(charged.values())
        for arm in ARMS: charged[arm]+=remainder/2
        budget()
        write(output / "receipt.json",dict(status="complete",plan=plan,checks=checks,recovery=recovery,coverage=coverage,
            particles=particles,seconds=time.monotonic()-started,charged_seconds=charged,
            final_native_digests=final_digests,final_checkpoint_sha256=final_hashes,endpoint_test_scores=endpoints,
            qualification_credit="none until independent CPU review; no optimizer superiority claim from historical comparisons"))
        report("complete",STOP)
        remainder=time.monotonic()-started-sum(charged.values())
        for arm in ARMS: charged[arm]+=remainder/2
        budget()
        completion=dict(complete=False,receipt_sha256=sha(output / "receipt.json"),
            plan_sha256=sha(output / "plan.json"),seconds=time.monotonic()-started,charged_seconds=charged,
            budget_seconds=14400,external_cpu_review_budget_seconds=1200,
            scope="Entire launch from before Torch import through all writes, also bounded by the external14400s timeout; zero launcher exit is mandatory.")
        write(output / "execution-completion.json",completion)
        remainder=time.monotonic()-started-sum(charged.values())
        for arm in ARMS: charged[arm]+=remainder/2
        budget()
        completion.update(complete=True,seconds=time.monotonic()-started,charged_seconds=dict(charged))
        write(output / "execution-completion.json",completion)
        remainder=time.monotonic()-started-sum(charged.values())
        for arm in ARMS: charged[arm]+=remainder/2
        try:
            budget()
        except TimeoutError:
            completion.update(complete=False,seconds=time.monotonic()-started,charged_seconds=dict(charged),overrun=True)
            write(output / "execution-completion.json",completion)
            raise
    except Exception as error:
        write(output / "failure.json",dict(error=type(error).__name__+": "+str(error),checks=checks,
            seconds=time.monotonic()-started,charged_seconds=charged))
        report("incomplete",min((loop.policy.completed_steps for loop in loops.values()),default=START))
        raise
    finally:
        for stream in traces.values(): stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particlegan-root",type=Path,default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/e22-supra-neutral-continuation-12800")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--preflight-output",type=Path)
    args=parser.parse_args()
    if args.output.exists() and not args.preflight_only:
        parser.error("preserve artifacts; use the one predeclared fresh output directory")
    readiness=preflight(CARD,args.particlegan_root)
    protocol=json.loads(CARD.read_text())
    if args.preflight_only:
        if args.preflight_output: write(args.preflight_output,readiness)
        print(json.dumps(dict(ready=True,cuda_initialized=False,protocol_sha256=readiness["protocol_sha256"],seconds=readiness["seconds"])))
        return
    if not protocol["execution_authorized"]:
        parser.error("card remains preparation-only; root must freeze authorization before the single launch")
    if args.output.resolve()!=Path(protocol["run"]["output"]).resolve():
        parser.error("output differs from the one predeclared fresh campaign")
    run(args,protocol,readiness)


if __name__=="__main__":
    main()
