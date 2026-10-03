#!/usr/bin/env python3
"""Source-only prototype for separately bounded full-teacher data preparation.

Root must freeze a new card/qualification before executing. This command builds
data only: no training/update/evaluation-selection or new experiment horizon.
"""
import time
STARTED=time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

TASK="supra_latent_coverage_data_preparation_v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def write(path,value,replace=False):
    path=Path(path);tmp=path.with_name(path.name+".tmp") if replace else path
    with tmp.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
    if replace:tmp.replace(path)


def pins(card,deadline):
    result={}
    for group in ("sources","dependencies","inputs"):
        for item in card[group].values():
            path=str(Path(item["path"]).resolve(strict=True))
            if path not in result:result[path]=sha(path)
            deadline()
            if result[path]!=item["sha256"]:raise ValueError("bound bytes differ: "+path)
    if result.get(str(Path(__file__).resolve()))!=sha(__file__):
        raise ValueError("preparation source must be card-bound")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();out=None;report=None;error=None;code=2;bound={};card={};card_hash=None
    limit=None
    def deadline():
        if limit is not None and time.monotonic()-STARTED>limit:
            raise TimeoutError("root-declared startup-through-final-writes preparation cap")
    try:
        card=json.loads(args.card.read_text());card_hash=sha(args.card);limit=card["seconds"]
        if (card["id"]!=TASK or not isinstance(limit,(int,float)) or limit<=0
                or Path(card["output"]).resolve()!=args.out.resolve()):
            raise ValueError("explicit root-frozen data preparation contract required")
        if args.out.exists():raise ValueError("existing output preserved; exclusive new directory required")
        bound=pins(card,deadline);args.out.mkdir(parents=True,exist_ok=False);out=args.out
        sys.path.insert(0,card["particlegan_root"]);sys.path.insert(1,card["application_root"])
        import torch
        import particlegan
        import supra
        from supra.particle_final_precision_task import load_task,fresh_restored
        from supra.particle_final_precision import checkpoint_precision,restore_precision
        from supra.particle_latent_coverage import build_latent_coverage_data,validate_latent_coverage_data
        from supra.particle_pilot import state_digest
        if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available()
                or Path(particlegan.__file__).resolve().parent.parent!=Path(card["particlegan_root"]).resolve()
                or Path(supra.__file__).resolve().parent.parent!=Path(card["application_root"]).resolve()):
            raise ValueError("root-only physicalGPU0/authenticated source imports required")
        torch.cuda.set_device(0)
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone()
        try:
            with torch.random.fork_rng(devices=[0]):
                original,legacy,base,judges=load_task(card);del judges
                loop,bootstrap=fresh_restored(base,original,legacy,card["precision"],card["neutral_native_digest"])
                before=checkpoint_precision(loop)
                try:
                    artifact=build_latent_coverage_data(original,loop.policy.encoder,
                        seconds=max(.001,limit-(time.monotonic()-STARTED)),
                        provenance=dict(protocol_sha256=card_hash,input_bindings=card["inputs"],
                                        source_identity=card["sources"],bootstrap=bootstrap),
                        progress=lambda row:print(json.dumps(row),flush=True))
                    validate_latent_coverage_data(original,artifact)
                    if state_digest(checkpoint_precision(loop))!=state_digest(before):
                        raise AssertionError("generation changed complete native/caller state before restore")
                finally:
                    restore_precision(loop,before)
                    if state_digest(checkpoint_precision(loop))!=state_digest(before):
                        raise AssertionError("generation finally restore is not exact")
                deadline()
                with (out/"latent-coverage-data.pt").open("xb") as stream:torch.save(artifact,stream)
                report=dict(task=TASK,complete=True,qualification="PASS",scientific_status=None,
                    artifact_sha256=sha(out/"latent-coverage-data.pt"),manifest=artifact["manifest"],
                    original12800_bootstrap=bootstrap,native_before_after_restore_exact=True,
                    native_updates=0,optimizer_steps=0,training_student_forwards=0,
                    future_training_eligible=False,scope="source/data prerequisite only; no quality trial")
        finally:
            if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(cuda,torch.cuda.get_rng_state(0)):
                raise AssertionError("ambient CPU/CUDA RNG changed")
        if pins(card,deadline)!=bound or sha(args.card)!=card_hash:
            raise AssertionError("source/input/card bytes changed")
        deadline();code=0
    except BaseException as exc:
        traceback.print_exc();error=dict(type=type(exc).__name__,message=str(exc))
        report=dict(report or {},task=TASK,complete=False,qualification="INCOMPLETE",scientific_status=None,error=error)
    if out is None:return code
    report.update(protocol_sha256=card_hash,source_input_bindings=bound,
                  seconds=time.monotonic()-STARTED,limit_seconds=limit)
    write(out/"report.json",report)
    completion=dict(task=TASK,complete=report["complete"],qualification=report["qualification"],
        scientific_status=None,exit_code=code,report_sha256=sha(out/"report.json"),
        protocol_sha256=card_hash,artifact_sha256=report.get("artifact_sha256"),
        seconds=time.monotonic()-STARTED,limit_seconds=limit,error=error)
    write(out/"completion.json",completion)
    if time.monotonic()-STARTED>limit:
        error=dict(type="FinalWriteDeadline",message="root-declared preparation cap exceeded")
        report.update(complete=False,qualification="INCOMPLETE",error=error,seconds=time.monotonic()-STARTED)
        write(out/"report.json",report,replace=True)
        completion.update(complete=False,qualification="INCOMPLETE",exit_code=2,error=error,
                          report_sha256=sha(out/"report.json"),seconds=time.monotonic()-STARTED)
        write(out/"completion.json",completion,replace=True);code=2
    return code


if __name__=="__main__":raise SystemExit(main())
