#!/usr/bin/env python3
"""Prospective root-only GPU300 full12800 data migration/restore prerequisite.

Zero native/optimizer updates. This qualifies full14-block owner/data glue;
it cannot restore full12800 into a reduced depth-one architecture or qualify
any convergence candidate. A separately qualified expanded-data artifact is
required, and the root must freeze actual card/input/source identities first.
"""
import time
STARTED=time.monotonic()
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

TASK="supra_latent_coverage_full12800_recovery_v1"
LIMIT=300


def deadline():
    if time.monotonic()-STARTED>LIMIT:raise TimeoutError("full12800 data recovery GPU300 cap")


def sha(path,charged=True):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
            if charged:deadline()
    return h.hexdigest()


def write(path,value,replace=False):
    path=Path(path);tmp=path.with_name(path.name+".tmp") if replace else path
    with tmp.open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
    if replace:tmp.replace(path)


def bindings(card):
    files={}
    for family in ("sources","dependencies","inputs"):
        for item in card[family].values():
            path=str(Path(item["path"]).resolve(strict=True))
            if path not in files:files[path]=sha(path)
            if files[path]!=item["sha256"]:raise ValueError("fixed bytes differ: "+path)
    if files.get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError("qualifier source must be pinned")
    return files


def run(card,out,torch):
    from supra.particle_final_precision_task import load_task,fresh_restored
    from supra.particle_final_precision import checkpoint_precision,restore_precision,precision_rng_scope
    from supra.particle_latent_coverage import (BASE_SEEDS,TRAIN_SEEDS,attach_latent_coverage,
        checkpoint_latent_coverage,restore_latent_coverage,validate_latent_coverage_data)
    from supra.particle_pilot import state_digest
    data,parent,base,judges=load_task(card);del judges
    artifact=torch.load(card["inputs"]["coverage_data"]["path"],map_location="cpu",weights_only=False,mmap=True)
    validate_latent_coverage_data(data,artifact);deadline()
    completed=[]; captures=[];common=data["fit"]["context"][[0,40,80,120]].clone()
    model_forwards=0
    def observed(loop):
        nonlocal model_forwards
        before=checkpoint_precision(loop)
        try:
            with precision_rng_scope(loop),torch.no_grad():
                value=loop.policy.routed_generate(common.to(loop.policy.device),sigma=0,perturb=False,averaged=False)
            if state_digest(checkpoint_precision(loop))!=state_digest(before):
                raise AssertionError("common FIT observation changed native/caller state")
            model_forwards+=1
            return value.cpu().clone()
        finally:restore_precision(loop,before)
    for seeds in (BASE_SEEDS,TRAIN_SEEDS):
        loop,bootstrap=fresh_restored(base,data,parent,card["precision"],card["neutral_native_digest"])
        original=checkpoint_precision(loop)
        try:
            initial=observed(loop)
            tag=attach_latent_coverage(loop,data,artifact,seeds=seeds,bootstrap_proof=bootstrap)
            state=checkpoint_latent_coverage(loop);migrated=observed(loop)
            if not torch.equal(initial,migrated):raise AssertionError("FIT migration changed serving prediction")
            path=out/("support-"+str(len(seeds))+"-12800.pt")
            with path.open("xb") as stream:torch.save(state,stream)
            saved=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
            own=checkpoint_latent_coverage(loop)
            bad=deepcopy(saved);bad["tag"]["seeds"]=[-1]
            try:restore_latent_coverage(loop,bad)
            except ValueError:pass
            else:raise AssertionError("wrong data tag accepted")
            if state_digest(checkpoint_latent_coverage(loop))!=state_digest(own):
                raise AssertionError("wrong-tag rejection changed state")
            fresh,proof=fresh_restored(base,data,parent,card["precision"],card["neutral_native_digest"])
            fresh_original=checkpoint_precision(fresh)
            try:
                attach_latent_coverage(fresh,data,artifact,seeds=seeds,bootstrap_proof=proof)
                restore_latent_coverage(fresh,saved)
                if state_digest(checkpoint_latent_coverage(fresh))!=state_digest(state):
                    raise AssertionError("fresh public own-state reload differs")
                replay=observed(fresh)
                if not torch.equal(replay,initial):raise AssertionError("reloaded raw serving differs")
                captures.append(dict(seeds=list(seeds),before=initial,after_migration=migrated,after_reload=replay))
            finally:
                # Original config/FIT references are restored explicitly before
                # original precision restore; no initializer is called here.
                fresh.config=deepcopy(fresh_original["pilot"]["config"])
                fresh.fit_context=data["fit"]["context"].to(fresh.policy.device)
                fresh.fit_targets=torch.zeros(240,4,32,32,device=fresh.policy.device,dtype=torch.float32)
                restore_precision(fresh,fresh_original)
                if state_digest(checkpoint_precision(fresh))!=state_digest(fresh_original):
                    raise AssertionError("fresh owner finally rollback differs")
            completed.append(dict(seeds=list(seeds),tag=tag,bootstrap=bootstrap,
                exact_initial_migrated_reload=True,wrong_tag_rejected_neutral=True,
                checkpoint_sha256=sha(path),native_updates=0,optimizer_steps=0))
        finally:
            loop.config=deepcopy(original["pilot"]["config"])
            loop.fit_context=data["fit"]["context"].to(loop.policy.device)
            loop.fit_targets=torch.zeros(240,4,32,32,device=loop.policy.device,dtype=torch.float32)
            restore_precision(loop,original)
            if state_digest(checkpoint_precision(loop))!=state_digest(original):
                raise AssertionError("original owner finally rollback differs")
        deadline()
    if not torch.equal(captures[0]["before"],captures[1]["before"]):
        raise AssertionError("two supports do not share original12800 prediction")
    with (out/"recovery-observations.pt").open("xb") as stream:torch.save(captures,stream)
    return dict(task=TASK,complete=True,qualification="PASS",scientific_status=None,
        arms=completed,observed_sha256=sha(out/"recovery-observations.pt"),
        counts=dict(public_routed_generate=model_forwards,student_forwards=model_forwards,
                    teacher_forwards=model_forwards,native_updates=0,optimizer_steps=0),
        full_clock=12800,whole_rollback_exact=True,initialized_after_trained_restore=False,
        scope="full14-block data migration prerequisite only; reduced trial/full quality unapproved")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();out=None;report=None;error=None;code=2;files={};card={};card_hash=None
    try:
        card=json.loads(args.card.read_text());card_hash=sha(args.card)
        if card["id"]!=TASK or card["seconds"]!=LIMIT or Path(card["output"]).resolve()!=args.out.resolve():
            raise ValueError("fixed root recovery contract differs")
        if args.out.exists():raise ValueError("exclusive new output required")
        files=bindings(card)
        generation=json.loads(Path(card["inputs"]["coverage_report"]["path"]).read_text())
        done=json.loads(Path(card["inputs"]["coverage_completion"]["path"]).read_text())
        if not (generation["complete"] is True and generation["qualification"]=="PASS"
                and done["complete"] is True and done["qualification"]=="PASS"
                and done["report_sha256"]==card["inputs"]["coverage_report"]["sha256"]
                and generation["artifact_sha256"]==done["artifact_sha256"]==card["inputs"]["coverage_data"]["sha256"]):
            raise ValueError("qualified full-teacher data preparation required")
        args.out.mkdir(parents=True,exist_ok=False);out=args.out
        sys.path.insert(0,card["particlegan_root"]);sys.path.insert(1,card["application_root"])
        import torch
        import particlegan
        import supra
        if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available()
                or Path(particlegan.__file__).resolve().parent.parent!=Path(card["particlegan_root"]).resolve()
                or Path(supra.__file__).resolve().parent.parent!=Path(card["application_root"]).resolve()):
            raise ValueError("root-only physicalGPU0/qualified imports required")
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone()
        try:
            with torch.random.fork_rng(devices=[0]):report=run(card,out,torch)
        finally:
            if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(cuda,torch.cuda.get_rng_state(0)):
                raise AssertionError("ambient CPU/CUDA RNG changed")
        if bindings(card)!=files or sha(args.card)!=card_hash:raise AssertionError("fixed bytes changed")
        deadline();code=0
    except BaseException as exc:
        traceback.print_exc();error=dict(type=type(exc).__name__,message=str(exc))
        report=dict(report or {},task=TASK,complete=False,qualification="INCOMPLETE",scientific_status=None,error=error)
    if out is None:return code
    report.update(protocol_sha256=card_hash,source_input_bindings=files,seconds=time.monotonic()-STARTED,limit_seconds=LIMIT)
    write(out/"report.json",report)
    done=dict(task=TASK,complete=report["complete"],qualification=report["qualification"],scientific_status=None,
        exit_code=code,report_sha256=sha(out/"report.json",False),protocol_sha256=card_hash,
        seconds=time.monotonic()-STARTED,limit_seconds=LIMIT,error=error)
    write(out/"completion.json",done)
    if time.monotonic()-STARTED>LIMIT:
        error=dict(type="FinalWriteDeadline",message="GPU300 startup-through-final-writes cap exceeded")
        report.update(complete=False,qualification="INCOMPLETE",error=error,seconds=time.monotonic()-STARTED)
        write(out/"report.json",report,True)
        done.update(complete=False,qualification="INCOMPLETE",exit_code=2,error=error,
                    report_sha256=sha(out/"report.json",False),seconds=time.monotonic()-STARTED)
        write(out/"completion.json",done,True);code=2
    return code


if __name__=="__main__":raise SystemExit(main())
