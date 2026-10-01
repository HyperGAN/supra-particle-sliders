#!/usr/bin/env python3
"""Native f459 CPU state witness: learned noise can veto its floor release."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'outputs/e22-convergence-gap/noise-floor-witness.json')
args=parser.parse_args()
ARCHIVE=ROOT/'outputs/e22-convergence-gap/particlegan-f459-source'
sys.path.insert(0,str(ARCHIVE))
import torch
from torch import nn
from particlegan import E22Policy,get_recipe,init

torch.set_num_threads(2)
recipe=get_recipe('e22',num_particles=8,z_dim=2,batch_size=4,output_noise_std=.125,
                  row_evidence_gate=False,particle_birth_death=False,
                  birth_death_feature_scale='none',birth_death_isolation=False)
G=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,2)).double()
D=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,1)).double()
prior=recipe.make_prior().double()
for module,key in ((G,0),(D,1),(prior,2)):
    init.deterministic_orthogonal_(module,seed=key)
opt_g=recipe.make_generator_optimizer([dict(params=G.parameters()),dict(params=[prior.z],lr=recipe.lr*recipe.prior_lr_mult)],latent_table=prior.z,foreach=False)
opt_d=recipe.make_critic_optimizer(D,ema_critic=deepcopy(D),foreach=False)
policy=E22Policy(recipe,G,D,prior=prior,generator_optimizer=opt_g,critic_optimizer=opt_d,
                 roles=[['generator','table'],['critic']],seed=21)
policy.controller.mobility=.09037799875328119
for row,roles in zip(policy.lr_settle.testers,policy.roles):
    for tester,role in zip(row,roles):
        if tester is not None and role!='critic':
            tester.s=1. if role=='noise' else 1./64.
noise_tester=policy.lr_settle.testers[0][-1]
assert policy.roles[0][-1]=='noise'
base=recipe.output_noise_std

cases=[]
for raw in (.0625,.005):
    with torch.no_grad():policy.log_output_sigma.fill_(math.log(raw))
    policy.opt_g.zero_grad(set_to_none=True)
    native_sigma=policy._output_sigma(base,detach=False)
    native_sigma.backward()
    native_grad=float(policy.log_output_sigma.grad)
    # Reporting-only counterfactual eligibility rule. No library function is
    # patched and no physical noise law is installed in a training loop.
    hypothetical_log=policy.log_output_sigma.detach().clone().requires_grad_(True)
    eligible_floor=base*policy.controller.mobility
    hypothetical_sigma=torch.maximum(hypothetical_log.exp(),torch.tensor(eligible_floor,dtype=hypothetical_log.dtype))
    hypothetical_sigma.backward()
    cases.append(dict(raw_sigma=raw,native_floor=base,native_physical_sigma=float(native_sigma.detach()),
        native_log_sigma_gradient=native_grad,
        hypothetical_floor_excluding_noise_eligibility=eligible_floor,
        hypothetical_physical_sigma=float(hypothetical_sigma.detach()),
        hypothetical_log_sigma_gradient=float(hypothetical_log.grad)))

# Native optimizer and native noise-group tester, with an unchanged clamped
# scalar. This is a state-level witness, not a GAN training experiment.
with torch.no_grad():policy.log_output_sigma.fill_(math.log(.0625))
noise_tester.begin([policy.log_output_sigma])
before=policy.log_output_sigma.detach().clone()
decisions=[]
for step in range(24):
    opt_g.zero_grad(set_to_none=True)
    policy._output_sigma(base,detach=False).backward()
    opt_g.step()
    result=noise_tester.observe([policy.log_output_sigma],ratio=1.,step=step+1)
    if result is not None:decisions.append(result)
after=policy.log_output_sigma.detach().clone()

actual=torch.load(ROOT/'outputs/e22-full-f459cb6d-6400/final.pt',map_location='cpu',weights_only=False,mmap=True)['policy']
actual_roles=[]
for roles,row in zip(actual['roles'],actual['lr_settle']):
    for role,tester in zip(roles,row):
        if role!='critic':actual_roles.append(dict(role=role,scale=None if tester is None else tester['s']))
source=ARCHIVE/'particlegan/policy.py'
result=dict(
    metadata=dict(revision='f459cb6d',native_owner=str(source),
        policy_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        controller_owner=str(ARCHIVE/'particlegan/continuous.py'),
        synthetic_state=True,trained_condition=False,
        setup='all other noncritic group scales 1/64; noise scale1; raw sigma below native floor; mobility .090378',
        no_native_source_edits=True,no_G_or_D_updates=True,no_output_metrics=True),
    cases=cases,
    native_frozen_noise_tester=dict(updates=24,decisions=decisions,scale=noise_tester.s,
        log_sigma_changed=not torch.equal(before,after),counts=noise_tester.counts,
        physical_sigma_after=float(policy._output_sigma(base)),
        raw_sigma_after=float(after.exp())),
    actual_6400=dict(noncritic_scales=actual_roles,mobility=actual['controller']['mobility'],
        raw_sigma=float(actual['output_noise'].exp()),physical_sigma=actual['last_output_sigma'],
        caveat='router/table scale1 and generator scale.5 already bind the floor independently; this witness does not establish cause of the current accuracy gap'),
    conclusion='Including the clamped/frozen noise group in all-groups floor-release eligibility creates a self-veto: its gradient and parameter motion vanish, its tester remains scale1, and it prevents its own floor release even when other groups have settled.')
out=args.output
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
