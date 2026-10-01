#!/usr/bin/env python3
"""Independent CPU review of exact-addition DV12 table-displacement profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FRACTIONS=[0.,.01,.1,.25,.5,1.]
MODES=['native_dv12','fixed_offset_dv12','clean']


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/e22-table-dv12-exact-offset-6400')
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/e22-particle-v2-6400')
    parser.add_argument('--role-autopsy',type=Path,default=ROOT/'outputs/e22-native-step-roles-6400')
    args=parser.parse_args()
    report=json.loads((args.output/'profiles.json').read_text())
    plan=report['plan']
    pg=Path(plan['imported_particlegan']).parent.parent
    sys.path.insert(0,str(pg))
    sys.path.insert(1,str(ROOT))
    import torch
    import torch.nn.functional as F
    from supra.particle_pilot import state_digest
    torch.set_num_threads(4)
    payload=torch.load(args.output/'profile-evidence.pt',map_location='cpu',weights_only=False)
    native=json.loads((args.output/'native-training-rows.json').read_text())
    saved=torch.load(args.run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    data=torch.load(args.run/'data.pt',map_location='cpu',weights_only=False)
    qualification=json.loads((args.run/'qualification-review.json').read_text())
    checks={}

    def check(name,value):
        checks[name]=bool(value)
        if not value:
            raise RuntimeError('qualification failed: '+name)

    check('evidence_sha256',sha(args.output/'profile-evidence.pt')==report['evidence_sha256'])
    check('raw_plan_exact',payload['plan']==plan)
    check('inputs_sha256',all(sha(args.run/name)==value for name,value in plan['input_sha256'].items()))
    check('archived_application_sources',all(sha(args.output/'source'/name)==value
        for name,value in plan['application_source_sha256'].items()))
    check('current_application_sources',all(sha(ROOT/name)==value for name,value in plan['application_source_sha256'].items()))
    check('particlegan_sources',all(sha(pg/name)==value for name,value in plan['particlegan_source_sha256'].items()))
    check('qualified_native_initial_state',qualification['qualified'] and
        state_digest(saved)==qualification['final_native_digest']==plan['initial_native_digest'])
    check('fixed_plan',plan['schema']=='supra_table_dv12_exact_offset_v1' and plan['start_step']==6400 and plan['updates']==5 and
        plan['fractions']==FRACTIONS and plan['modes']==MODES)
    check('fixed_horizon',len(report['rows'])==len(payload['rows'])==len(native)==5 and
        [row['step'] for row in report['rows']]==list(range(6401,6406)))
    check('runtime_qualified_native_replay',all(report[name] is True for name in
        ('completed','exact_native_rows','exact_native_full_replay','frozen_unchanged','source_and_inputs_unchanged')) and
        report['final_native_digest']==report['reference_final_native_digest'])
    expected_data=torch.Generator().set_state(saved['data_rng'].cpu())
    maximum_score_error=0.
    score_count=0
    summary=[]
    for row,raw,training in zip(report['rows'],payload['rows'],native):
        step=row['step']
        prefix=str(step)+'_'
        hold=step%5==0
        weight=.1 if hold else 1.
        indices=torch.randint(len(data['holds' if hold else 'fit']['context']),(4,),generator=expected_data)
        check(prefix+'native_sampling_task',training['step']==step==raw['step'] and
            row['hold']==training['hold']==hold and row['game_weight']==training['game_weight']==weight and
            training['batch_indices']==indices.tolist())
        check(prefix+'state_unchanged_witness',row['native_state_unchanged_by_profiles'] is True)
        check(prefix+'actual_site_capture_shapes',row['site_calls']==len(raw['sites'])==len(saved['config']['sites'])==71 and
            all(site['codes'].shape==site['perturbed'].shape==site['displacement'].shape and
                site['codes'].ndim==2 and site['codes'].shape[1]==4 and
                site['codes'].dtype==site['perturbed'].dtype==site['displacement'].dtype==torch.float32 and
                torch.isfinite(site['codes']).all() and torch.isfinite(site['displacement']).all() and
                torch.isfinite(site['perturbed']).all() for site in raw['sites']))
        check(prefix+'actual_offset_native_addition_bitexact',all(
            torch.equal(site['codes']+site['displacement'],site['perturbed']) for site in raw['sites']))
        check(prefix+'table_geometry_shape_finite',all(value.shape==(128,4) and torch.isfinite(value).all()
            for value in (raw['table_before'],raw['table_gradient'],raw['displacement'])))
        dot=float((raw['table_gradient'].double()*raw['displacement'].double()).sum())
        check(prefix+'gradient_geometry_reconstruction',abs(dot-row['weighted_gradient_dot_displacement'])<1e-12 and
            abs(dot/weight-row['unweighted_gradient_dot_displacement'])<1e-12 and
            abs(float(raw['table_gradient'].double().norm())-row['table_gradient_norm'])<1e-12 and
            abs(float(raw['displacement'].double().norm())-row['displacement_norm'])<1e-12)
        check(prefix+'all_profile_modes',list(row['profiles'])==list(raw['profiles'])==MODES)
        for mode in MODES:
            reported=row['profiles'][mode]
            values=raw['profiles'][mode]
            check(prefix+mode+'_fixed_fractions',[value['fraction'] for value in reported]==
                [value['fraction'] for value in values]==FRACTIONS)
            baseline=values[0]['prediction']
            for record,point in zip(reported,values):
                gap,prediction=point['gap'],point['prediction']
                check(prefix+mode+'_'+str(record['fraction'])+'_panel_shape_finite',
                    gap.shape==(4,1) and gap.dtype==torch.float32 and torch.isfinite(gap).all() and
                    prediction.shape==(4,4,32,32) and prediction.dtype==torch.float32 and torch.isfinite(prediction).all())
                score=float(F.softplus(gap).double().mean())
                error=abs(score-record['g_game'])
                maximum_score_error=max(maximum_score_error,error)
                score_count+=1
                check(prefix+mode+'_'+str(record['fraction'])+'_score',error<=3e-7)
                check(prefix+mode+'_'+str(record['fraction'])+'_paired_delta',
                    record['delta_from_mode_baseline']==record['g_game']-reported[0]['g_game'])
                fraction=float((prediction!=baseline).float().mean())
                check(prefix+mode+'_'+str(record['fraction'])+'_prediction_change',
                    fraction==record['prediction_changed_fraction'] and
                    torch.equal(prediction,baseline)==record['prediction_equal_to_mode_baseline'])
        native_zero=raw['profiles']['native_dv12'][0]
        fixed_zero=raw['profiles']['fixed_offset_dv12'][0]
        check(prefix+'both_DV12_zero_endpoints_bitexact',torch.equal(native_zero['prediction'],raw['actual_prediction']) and
            torch.equal(fixed_zero['prediction'],raw['actual_prediction']) and
            torch.equal(native_zero['gap'],fixed_zero['gap']))
        check(prefix+'actual_Gpass_game_weight_exact',
            float(torch.tensor(row['profiles']['native_dv12'][0]['g_game'],dtype=torch.float32)*weight)==training['loss_g'])
        check(prefix+'fixed_offset_outputs_used_as_prescribed_witness',
            all(point['native_vs_fixed_offset_site_output_changed_fraction']==0 for point in row['profiles']['fixed_offset_dv12']))
        summary.append(dict(step=step,hold=hold,unweighted_gradient_dot_displacement=dot/weight,
            fraction_one_game_deltas={mode:row['profiles'][mode][-1]['delta_from_mode_baseline'] for mode in MODES},
            fraction_0_01_game_deltas={mode:row['profiles'][mode][1]['delta_from_mode_baseline'] for mode in MODES},
            native_minus_fixed_offset_fraction_one=row['profiles']['native_dv12'][-1]['g_game']-
                                              row['profiles']['fixed_offset_dv12'][-1]['g_game'],
            native_fraction_0_01_prediction_change=row['profiles']['native_dv12'][1]['prediction_changed_fraction']))
    if (args.role_autopsy/'roles.json').exists():
        autopsy=json.loads((args.role_autopsy/'roles.json').read_text())
        check('same_native_updates_as_role_autopsy',autopsy['final_native_digest']==report['final_native_digest'])
        check('native_table_endpoint_matches_role_autopsy',all(
            row['profiles']['native_dv12'][-1]['g_game']==old['arms']['table']['training_native_dv12']['g_game'] and
            row['profiles']['native_dv12'][0]['g_game']==old['arms']['none']['training_native_dv12']['g_game']
            for row,old in zip(report['rows'],autopsy['rows'])))
    result=dict(qualified=True,checks=checks,scores_reconstructed=score_count,
        maximum_cross_device_FP32_score_error=maximum_score_error,summary=summary,
        limits='CPU recomputes all raw game panels, paired deltas, prediction equality/change fractions, gradient/displacement geometry and native data draws. Actual native CUDA/BF16 forward replay and unchanged in-step/full state are bounded runtime witnesses. Fractions are diagnostics, not selected optimizer settings.')
    write_path=args.output/'independent-review.json'
    write_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
