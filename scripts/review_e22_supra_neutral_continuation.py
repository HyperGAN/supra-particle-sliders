#!/usr/bin/env python3
"""Independent CPU qualification of every predeclared continuation artifact.

No native updates, model construction, or forward re-execution. GPU recovery,
endpoint, and export witnesses remain explicit archived-source witnesses.
"""
import time
REVIEW_STARTED=time.monotonic()
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
sys.path.insert(1,str(ROOT))
import torch
import review_e22_supra_neutral_initialization as held_review
from continue_e22_supra_neutral_initialization import preflight
from e22_supra_neutral_continuation_contract import (ARMS,CARD,ENDPOINTS,LOCAL_CHECKPOINTS,
    MODES,PIN,START,STOP,canonical,paired_row_contract,require_checkpoint_contract,sha)
from experiment_e22_supra_particle_gated import write


def read(path): return json.loads(Path(path).read_text())


def load(path): return torch.load(path,map_location="cpu",weights_only=False,mmap=True)


def review_budget():
    if time.monotonic()-REVIEW_STARTED>1200:
        raise TimeoutError("external1200-second CPU review budget exhausted")


def review_continuation(directory,pg):
    directory=Path(directory).resolve()
    readiness=preflight(CARD,pg)
    protocol=read(CARD)
    if directory!=Path(protocol["run"]["output"]).resolve():
        raise ValueError("independent review run is not the predeclared cohort")
    review=held_review.Reviewer()
    plan=read(directory / "plan.json")
    parent=Path(protocol["parent"]["run"])
    parent_plan=read(parent / "plan.json")
    parent_review=read(parent / "independent-review.json")
    inherited=parent_plan["protocol"]
    review.require(plan["schema"]=="supra_neutral_continuation_plan_v1" and plan["protocol"]==protocol
        and plan["protocol_sha256"]==sha(CARD) and protocol["execution_authorized"] is True
        and plan["start_step"]==START and plan["fixed_updates"]==STOP
        and plan["additional_updates"]==STOP-START and plan["endpoints"]==list(ENDPOINTS)
        and plan["configs"]==parent_plan["configs"] and plan["initialization_after_restore"] is False,
        "fixed continuation law/configs")
    review.require(plan["primary_target"]=="ordinary_lora_6400" and plan["secondary_budget_context"]=="ordinary_lora_12800",
                   "fixed primary and secondary comparisons")
    for name,digest in plan["application_source_sha256"].items():
        review_budget()
        review.require(sha(ROOT / name)==sha(directory / "source" / name)==digest,"held/new/archive source "+name)
    review.require(plan["particlegan_source_sha256"]==parent_plan["particlegan_source_sha256"]
        and plan["particlegan_commit"]==PIN,"unchanged native law")
    for name,digest in plan["particlegan_source_sha256"].items():
        review_budget()
        review.require(sha(pg / name)==sha(directory / "source/native" / name)==digest,"held/archive native "+name)
    review.require(plan["input_paths"]=={name:entry["path"] for name,entry in protocol["inputs"].items()}
        and plan["input_sha256"]=={name:entry["sha256"] for name,entry in protocol["inputs"].items()},"all actual inputs bound")
    data=load(plan["input_paths"]["data"])
    digest=held_review.state_digest
    review.require(digest(data)==inherited["data"]["digest"],"full unchanged data")
    judges=read(directory / "judges.json")
    review.require(judges==parent_review["judges"],"unchanged two actual fixed judges")
    stream=torch.Generator().manual_seed(72)
    panels={}
    for pool,count in held_review.COUNTS.items():
        h=hashlib.sha256()
        for start in range(0,count,4):
            raw=torch.randn(4,min(4,count-start),256,16,generator=stream)
            h.update(raw.contiguous().numpy().tobytes(order="C"))
        panels[pool]=h.hexdigest()
    review.require(panels==parent_review["full_evaluation_panel_sha256"],"all private full panels independently regenerated")
    probe=read(directory / "historical-probes/references.json")
    expected_probe=read(parent / "historical-probes.json")
    for key in ("indices","context_digest","panel_digest"):
        review.require(probe[key]==expected_probe[key],"same observational cohort "+key)
    review.require(set(probe["scores"])=={"ordinary_lora_6400","ordinary_lora_12800"}
        and all(math.isfinite(x) for x in probe["scores"].values()),"fixed finite ordinary probes")
    traces={arm:held_review.trace(directory / arm / "train.jsonl") for arm in ARMS}
    review.require(all(len(rows)==STOP-START for rows in traces.values()),"both complete6400 additional update streams")
    cpu=torch.Generator().manual_seed(7)
    for _ in range(START): torch.randint(240,(4,),generator=cpu)
    expected_rng={START:cpu.get_state().clone()}
    for step in range(START+1,STOP+1):
        indices=torch.randint(240,(4,),generator=cpu).tolist()
        a,b=(traces[arm][step-START-1] for arm in ARMS)
        paired_row_contract(a,b,step,indices)
        for row in (a,b):
            review.require(all(math.isfinite(row[name]) for name in
                ("loss_g","loss_d","loss_d_game","penalty","bank_grad_norm","output_sigma")),"finite native row")
            review.require(row["penalty_calls"]==step and row["dense_gradient_rows"]==128
                and row["bank_grad_norm"]>0,"native KA2 and all128 live particle gradients")
        if step in LOCAL_CHECKPOINTS: expected_rng[step]=cpu.get_state().clone()
    recovery=read(directory / "recovery-06400-06402.json")
    metadata,frozen,checkpoints,exports={},{},{},{}
    for arm in ARMS:
        checkpoints[arm]={}
        for step in (START,*LOCAL_CHECKPOINTS):
            review_budget()
            path=Path(protocol["parent"]["checkpoints"][arm]["path"]) if step==START else directory / arm / f"checkpoint-{step:05d}.pt"
            state=load(path)
            policy=state["policy"]
            require_checkpoint_contract(state,plan["configs"][arm],step)
            review.require(state["config"]["particle_init"]==MODES[arm],arm+" exact initialization tag")
            if step==START:
                metadata[arm]=dict(config=state["config"],policy=dict(requires_grad=policy["requires_grad"],roles=policy["roles"]),
                    bank=policy["table"].clone(),router={key:value.clone() for key,value in policy["models"]["router"].items()})
                frozen[arm]=digest(held_review.immutable(state))
            held_review.check_checkpoint_contract(review,state,metadata[arm],plan["configs"][arm],arm,step,inherited["data"]["digest"])
            review.require(digest(held_review.immutable(state))==frozen[arm],arm+" all FAST/EMA frozen parameters unchanged")
            for family in ("models","averages"):
                encoder=policy[family]["encoder"]
                review.require(torch.equal(encoder["contexts"],data["text_contexts"])
                    and torch.equal(encoder["masks"],data["text_masks"]),arm+" frozen text/masks")
            review.require(torch.equal(policy["models"]["critic"]["scale"],data["coordinate_scale"])
                and torch.equal(policy["optimizers"][1]["regularizer"]["ema"]["scale"],data["coordinate_scale"]),arm+" FAST/EMA coordinate scales")
            review.require(torch.equal(state["data_rng"],expected_rng[step]),arm+" exact CPU7 native checkpoint stream")
            for family in ("models","averages"):
                for role in ("generator","router"):
                    for name,tensor in policy[family][role].items():
                        if policy["requires_grad"][role].get(name,False):
                            review.require(torch.isfinite(tensor).all().item(),arm+" finite trainable "+name)
            review.require(torch.isfinite(policy["table"]).all().item() and torch.isfinite(policy["averaged_table"]).all().item(),"finite FAST/EMA bank")
            native_digest=digest(state)
            file_sha=sha(path)
            checkpoints[arm][str(step)]=dict(path=str(path),file_sha256=file_sha,native_digest=native_digest,
                paired_rng_digest=digest(state["paired_noise_rng"]),native_moves=policy["routing"]["counters"]["moves"],
                external_parent_reference=step==START)
            if step==START:
                witness=parent_review["checkpoint_artifacts"][arm][str(START)]
                review.require(native_digest==witness["native_digest"] and file_sha==witness["file_sha256"],"actual qualified6400 native content")
            if step==6402:
                witness=recovery[arm]
                review.require(witness["from_step"]==START and witness["to_step"]==6402
                    and witness["native_replay_updates"]==4 and witness["rows_exact"] is True and witness["state_exact"] is True
                    and witness["serialized_row_digest"]==digest(witness["rows"]) and witness["native_digest"]==native_digest,
                    arm+" actual two-path recovery witnesses")
                review.require([canonical(row) for row in witness["rows"]]==traces[arm][:2],arm+" authoritative first two rows equal witnessed replay")
            if step in ENDPOINTS:
                pins=parent_review["export_artifacts"][arm+"@6400"]["native_pins"]
                exports[arm+"@"+str(step)]=held_review.review_export(review,directory / arm / f"adapter-{step:05d}.safetensors",
                    state,arm,step,pins["backend_sha256"],pins)
            if step==STOP:
                review.require(digest(policy["table"])!=digest(metadata[arm]["bank"])
                    and digest(policy["models"]["router"])!=digest(metadata[arm]["router"]),arm+" live bank/router learned during continuation")
                bridges=[name for name in policy["models"]["generator"] if name.endswith("bridge.weight")]
                review.require(len(bridges)==71 and all(policy["models"]["generator"][name][:,16:].count_nonzero()>0
                    and policy["requires_grad"]["generator"][name] and policy["requires_grad"]["generator"][name.replace("weight","bias")]
                    for name in bridges),"all71 particle C/H/b retained")
            del state,policy,encoder,tensor
            gc.collect()
        progress=held_review.trace(directory / arm / "monitor/progress.jsonl")
        review.require([row["step"] for row in progress]==list(range(START,STOP+1,200)),"fixed33 continuation probes")
        review.require(read(directory / arm / "monitor/progress-edit-indices.json")==probe["indices"],"same held-out probe indices")
        parent_last=parent_review["progress"][arm][-1]
        review.require(progress[0]["probes"]==parent_last["probes"],"6400 observational clean/DV12 boundary unchanged")
        for row in progress:
            review.require(row["source"]=="FAST current particle G" and row["output_metrics_used"] is False
                and row["evaluation_only"] is True and row["native_state_unchanged"] is True
                and set(row["probes"])=={"edit"} and row["probes"]["edit"]["contexts"]==12,"observational probes only")
            review.require(all(math.isfinite(row["probes"]["edit"][mode][judge])
                for mode in ("clean","dv12") for judge in ("frozen_start_D","live_D")),"finite clean/DV12 games")
            if row["step"]>START:
                window=traces[arm][row["step"]-START-100:row["step"]-START]
                rolling=row["rolling"]["edit"]
                review.require(rolling["updates"]==100 and rolling["first_step"]==window[0]["step"]
                    and rolling["last_step"]==window[-1]["step"],"continuation100-update rolling window")
                for field,key in (("g_game","loss_g"),("d_game","loss_d_game"),("penalty","penalty"),("bank_grad_norm","bank_grad_norm")):
                    review.close(rolling[field],sum(item[key] for item in window)/100,"rolling game reduction")
    review.require(all(len(items)==18 for items in checkpoints.values()),"all34 new plus2 parent full states")
    for step in (START,*LOCAL_CHECKPOINTS):
        review.require(checkpoints[ARMS[0]][str(step)]["paired_rng_digest"]==checkpoints[ARMS[1]][str(step)]["paired_rng_digest"],"matched checkpoint Gaussian RNG")
    historical=read(directory / "historical-references.json")
    review.require(set(historical)=={"ordinary_lora_6400","ordinary_lora_12800"},"two predeclared historical comparators")
    review.require(historical["ordinary_lora_6400"]==read(parent / "historical-references.json")["ordinary_lora_6400"],"primary6400 reference exactly inherited")
    evaluations={name:held_review.check_evaluation(review,value,data,name) for name,value in historical.items()}
    comparisons,ablations={},{}
    for step in ENDPOINTS:
        review_budget()
        results={}
        for arm in ARMS:
            review_budget()
            label=arm+"@"+str(step)
            result=read(directory / arm / f"evaluation-{step:05d}.json")
            results[arm]=result
            evaluations[label]=held_review.check_evaluation(review,result,data,label)
            for name,reference in historical.items():
                comparisons[label+"-"+name]=held_review.paired_summary(review,reference,result,label+"-"+name)
            raw=read(directory / arm / f"particle-ablations-{step:05d}.json")
            ablations[label]={}
            for kind,key in (("zero_code","code_scores"),("mass_only","mass_only_scores")):
                ablation=raw[key]
                evaluations[label+"/"+kind]=held_review.check_evaluation(review,ablation,data,label+"/"+kind)
                ablations[label][kind]=held_review.paired_summary(review,result,ablation,label+"/"+kind)
                declared=raw[("zero_code" if kind=="zero_code" else "mass_only")+"_minus_live_test_game"]
                for judge in held_review.JUDGES:
                    review.close(declared[judge],ablation["test"][judge]-result["test"][judge],"actual ablation gain")
        comparisons["neutral-control@"+str(step)]=held_review.paired_summary(review,results[ARMS[0]],results[ARMS[1]],"matched neutral-control")
    receipt,status,completion,launcher=(read(directory / name) for name in
        ("receipt.json","status.json","execution-completion.json","launcher-exit.json"))
    review.require(receipt["status"]==status["phase"]=="complete" and status["step"]==status["steps"]==STOP
        and status["editing_updates"]==STOP and status["preservation_updates"]==0
        and receipt["plan"]==plan and all(value is True for value in receipt["checks"].values()),"complete held GPU witnesses")
    review.require(completion["complete"] is True and completion["receipt_sha256"]==sha(directory / "receipt.json")
        and completion["plan_sha256"]==sha(directory / "plan.json") and completion["seconds"]<=14400
        and all(x<=7200 for x in completion["charged_seconds"].values()) and receipt["seconds"]<=14400,
        "complete bounded execution receipt")
    review.require(launcher["complete"] is True and launcher["exit_code"]==0 and launcher["timed_out"] is False
        and launcher["seconds"]<=14400 and launcher["budget_seconds"]==14400,"actual external bounded zero exit")
    for arm in ARMS:
        actual=dict(live_bank_updates=sum(row["dense_gradient_rows"]>0 for row in traces[arm]),
            dense_128_row_updates=sum(row["dense_gradient_rows"]==128 for row in traces[arm]),
            moves=sum((row.get("move") or {}).get("moves",0) for row in traces[arm]))
        review.require(receipt["coverage"][arm]==actual and actual["moves"]==checkpoints[arm][str(STOP)]["native_moves"]-checkpoints[arm][str(START)]["native_moves"],"actual continuation particle coverage")
        review.require(receipt["final_native_digests"][arm]==checkpoints[arm][str(STOP)]["native_digest"]
            and receipt["final_checkpoint_sha256"][arm]==checkpoints[arm][str(STOP)]["file_sha256"],"actual final checkpoint file/content")
        for step in ENDPOINTS:
            label=arm+"@"+str(step)
            review.require(receipt["endpoint_test_scores"][label]==read(directory / arm / f"evaluation-{step:05d}.json")["test"],"complete endpoint receipt records")
            for prefix in ("endpoint_native_immutable_","mass_only_all_sites_","export_reload_raw_exact_","export_reload_immutable_"):
                review.require(receipt["checks"][prefix+arm+str(step)] is True,"held GPU endpoint/reload witness")
    target={arm+"@"+str(step):all(comparisons[arm+"@"+str(step)+"-ordinary_lora_6400"]["test"][judge]["new_minus_reference"]<0
        for judge in held_review.JUDGES) for arm in ARMS for step in ENDPOINTS}
    return dict(schema="supra_neutral_continuation_independent_review_v1",qualified=True,partial=False,
        checks=review.checks,reviewer_sha256=sha(__file__),protocol_sha256=sha(CARD),plan_sha256=sha(directory / "plan.json"),
        parent_review_sha256=protocol["parent"]["review_sha256"],application_source_sha256=plan["application_source_sha256"],
        input_sha256=plan["input_sha256"],native_source_digest=parent_review["native_source_digest"],
        trace_rows={arm:len(rows) for arm,rows in traces.items()},common_matched_rows=STOP-START,
        checkpoint_artifacts=checkpoints,export_artifacts=exports,judges=judges,
        full_evaluation_panel_sha256=panels,evaluation_summaries=evaluations,comparisons=comparisons,particle_ablations=ablations,
        beats_fixed_primary_two_judges=target,
        recovery_qualification="Exact actual6400→6402 direct/replay and authoritative native GPU witnesses, fully byte/content bound; CPU did not re-execute native updates.",
        gpu_runtime_witnesses=dict(launcher=launcher,completion=completion,receipt_sha256=sha(directory / "receipt.json")),
        external_cpu_review_excluded_from_gpu_runner_budget=True,external_cpu_review_budget_seconds=1200,
        native_or_generator_updates_by_reviewer=0,model_forward_calls_by_reviewer=0,
        output_metrics_used_for_optimizer_or_selection=False,
        qualification_credit="This cohort only. Matched control/neutral comparison; historical MSE ordinary references have different initialization, editing/preservation composition, and objective. Extra-budget win over6400 is not matched optimizer superiority.",
        review_seconds=time.monotonic()-REVIEW_STARTED)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT / "outputs/e22-supra-neutral-continuation-12800")
    parser.add_argument("--particlegan-root",type=Path,default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.preflight_only:
        report=preflight(CARD,args.particlegan_root)
        target=args.output or ROOT / "outputs/e22-supra-neutral-continuation-preflight.json"
        write(target,report)
        print(json.dumps(dict(ready=True,quality_updates=0,cuda_initialized=False,output=str(target))))
        return
    target=args.output or args.run / "independent-review.json"
    if target.exists() or Path(str(target)+".completion.json").exists():
        parser.error("preserve existing independent review; no unrequested rerun")
    report=review_continuation(args.run,args.particlegan_root)
    write(target,report)
    completion=dict(complete=False,report_sha256=sha(target),review_seconds=time.monotonic()-REVIEW_STARTED,
        budget_seconds=1200,combined_seconds=report["gpu_runtime_witnesses"]["launcher"]["seconds"]+time.monotonic()-REVIEW_STARTED)
    write(Path(str(target)+".completion.json"),completion)
    if time.monotonic()-REVIEW_STARTED>1200 or completion["combined_seconds"]>15600:
        raise TimeoutError("external CPU/combined review overrun; report remains unqualified without companion")
    completion.update(complete=True,review_seconds=time.monotonic()-REVIEW_STARTED,
        combined_seconds=report["gpu_runtime_witnesses"]["launcher"]["seconds"]+time.monotonic()-REVIEW_STARTED)
    write(Path(str(target)+".completion.json"),completion)
    if time.monotonic()-REVIEW_STARTED>1200 or report["gpu_runtime_witnesses"]["launcher"]["seconds"]+time.monotonic()-REVIEW_STARTED>15600:
        completion.update(complete=False,overrun=True,review_seconds=time.monotonic()-REVIEW_STARTED)
        write(Path(str(target)+".completion.json"),completion)
        raise TimeoutError("external CPU review final-write overrun")
    print(json.dumps(dict(qualified=True,checks=report["checks"],review_seconds=completion["review_seconds"],output=str(target))))


if __name__=="__main__": main()
