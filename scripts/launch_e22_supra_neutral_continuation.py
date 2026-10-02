#!/usr/bin/env python3
"""Single bounded subprocess launch; external exit witness is required by review."""
import time
STARTED = time.monotonic()
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/e22-supra-neutral-continuation-12800")
    parser.add_argument("--particlegan-root",type=Path,default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    card=json.loads((ROOT / "docs/e22_supra_neutral_continuation_protocol.json").read_text())
    if args.output.resolve()!=Path(card["run"]["output"]).resolve():
        parser.error("output differs from the one fixed predeclared cohort")
    if args.device!="cuda:0": parser.error("fixed physical GPU0 campaign")
    if args.output.exists(): parser.error("preserve existing artifacts; one fresh launch only")
    command=[sys.executable,str(ROOT / "scripts/continue_e22_supra_neutral_initialization.py"),
             "--output",str(args.output),"--particlegan-root",str(args.particlegan_root),"--device",args.device]
    deadline=STARTED+14400
    timed_out=False
    with Path(str(args.output)+".launcher.log").open("x",buffering=1) as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0"))
        try:
            exit_code=process.wait(timeout=max(0,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True
            process.terminate()
            try: exit_code=process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill();exit_code=process.wait()
    seconds=time.monotonic()-STARTED
    report=dict(schema="supra_neutral_continuation_launcher_exit_v1",exit_code=exit_code,
        timed_out=timed_out,seconds=seconds,budget_seconds=14400,command=command,
        cuda_visible_devices="0",
        complete=exit_code==0 and not timed_out and seconds<=14400,
        cleanup_overrun_seconds=max(0,seconds-14400))
    target=args.output / "launcher-exit.json" if args.output.exists() else Path(str(args.output)+".launcher-exit.json")
    target.write_text(json.dumps(report,indent=2)+"\n")
    if time.monotonic()-STARTED>14400 and report["complete"]:
        report.update(complete=False,seconds=time.monotonic()-STARTED)
        target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    if not report["complete"]: raise SystemExit(exit_code or 124)


if __name__=="__main__": main()
