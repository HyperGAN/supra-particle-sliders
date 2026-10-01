#!/usr/bin/env python3
"""Independent CPU qualification of the structural phase candidate receipt."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/e22-structural-phase-causal-6400')
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/e22-particle-v2-6400')
    parser.add_argument('--particlegan-root', type=Path,
                        default=ROOT/'outputs/e22-convergence-gap/particlegan-cabe2084-source')
    args = parser.parse_args()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    from supra.particle_pilot import state_digest
    torch.set_num_threads(4)
    result = json.loads((args.output/'result.json').read_text())
    plan = result['plan']
    payload_path = args.output/'candidate-losses.pt'
    payload = torch.load(payload_path, map_location='cpu', weights_only=False)
    saved = torch.load(args.run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise RuntimeError(f'qualification failed: {name}')

    check('payload_sha256', sha(payload_path) == result['payload_sha256'])
    check('input_sha256', all(sha(args.run/name) == digest for name,digest in plan['input_sha256'].items()))
    check('source_sha256', all(sha(path) == digest for path,digest in plan['source_sha256'].items()))
    check('native_state_digest', state_digest(saved) == plan['native_state_digest'])
    check('zero_feature_harm_no_output_guard', plan['max_context_harm'] == 0 and plan['output_error_guard'] is False)
    control = saved['policy']['routing']
    spec = control['config']
    evidence = control['evidence']['tensors']
    table, average = saved['policy']['table'], saved['policy']['averaged_table']
    effect = evidence['effect_sum'] / evidence['effect_weight'].clamp_min(1e-30)
    effective = evidence['effect_weight'].square() / evidence['effect_weight_sq'].clamp_min(1e-300)
    enough = effective >= spec['min_observations']
    fresh = evidence['last_probe'] >= max(1,control['counters']['evals']-math.ceil(len(table)/spec['probe_budget']))
    children = (enough & fresh & (effect > spec['min_effect'])).nonzero().flatten()
    parents = (enough & fresh & (effect < -spec['min_effect'])).nonzero().flatten()
    children = children[effect[children].argsort(descending=True,stable=True)]
    parents = parents[effect[parents].argsort(stable=True)]
    pairs = [[int(child),int(parent)] for child in children for parent in parents][:spec['candidate_budget']]
    check('native_pair_eligibility_order_budget', pairs == plan['pairs'] == [list(pair) for pair in payload['pairs']])
    check('source_tables_exact', torch.equal(table,payload['source_table']) and torch.equal(average,payload['source_averaged_table']))
    check('cached_actual_gradient_exact', torch.equal(control['latest_gradient'], payload['gradients']['actual_cached_hold']))
    check('cached_actual_gradient_beta0_moment', torch.equal(control['latest_gradient'],
        saved['policy']['optimizers'][0]['state'][saved['policy']['optimizers'][0]['param_groups'][2]['params'][0]]['exp_avg']))
    check('candidate_count', len(payload['candidates']) == len(pairs)*4 == len(result['entries']))
    baselines = payload['baselines']
    check('loss_shapes_finite', all(value.dtype == torch.float64 and value.shape == (64,) and torch.isfinite(value).all()
        for value in [*baselines.values(), *[loss for item in payload['candidates'] for loss in item['values'].values()]]))
    check('directions_dense_finite', all(value.shape == table.shape and torch.isfinite(value).all() and (value.norm(dim=1)>0).all()
                                      for value in payload['gradients'].values()))
    entries = []
    maximum_scalar_error = 0.
    maximum_delta_relative_error = 0.
    for item, reported in zip(payload['candidates'], result['entries']):
        record, values, delta = item['record'], item['values'], item['delta']
        check('candidate_record_' + str(len(entries)), record == reported)
        child,parent = record['child'],record['parent']
        check('candidate_pair_' + str(len(entries)), [child,parent] in pairs)
        if record['variant'] == 'duplicate':
            check('duplicate_zero_delta_' + str(len(entries)), not bool(delta.any()) and record['direction']=='shared')
        else:
            gradient = payload['gradients'][record['direction']][parent]
            def radius(bank):
                distance = (bank-bank[parent]).norm(dim=1)
                return distance[distance>0].min()*.5
            norm_bound = torch.minimum(radius(table),radius(average))*spec['split_scale']
            expected = -gradient/gradient.norm().clamp_min(1e-30)*norm_bound
            relative = float((delta-expected).norm()/expected.norm().clamp_min(1e-30))
            maximum_delta_relative_error = max(maximum_delta_relative_error,relative)
            check('native_radius_direction_' + str(len(entries)), relative < 1e-6)
        fast = baselines['guard']-values['guard']
        avg = baselines['average_guard']-values['average_guard']
        scalars = dict(fit_error=float(values['fit'].mean()),
                       fit_gain=float(baselines['fit'].mean())-float(values['fit'].mean()),
                       guard_gain=float(fast.mean()), average_guard_gain=float(avg.mean()),
                       max_context_harm=float(-fast.min()), average_max_context_harm=float(-avg.min()))
        maximum_scalar_error=max(maximum_scalar_error, max(abs(record[key]-value) for key,value in scalars.items()))
        fit_pass=scalars['fit_gain'] >= spec['improvement_margin']
        guard_pass=(scalars['guard_gain']>=spec['improvement_margin'] and scalars['average_guard_gain']>=-1e-12
                    and max(scalars['max_context_harm'],scalars['average_max_context_harm'])<=1e-12)
        check('native_acceptance_' + str(len(entries)),
            record['fit_pass']==fit_pass and record['guard_pass']==guard_pass and record['accepted']==(fit_pass and guard_pass)
            and record['harmed_fast']==int((fast < -1e-12).sum())
            and record['harmed_average']==int((avg < -1e-12).sum()))
        entries.append(record)
    check('per_context_scalar_recomputation', maximum_scalar_error < 1e-12)
    for direction, decision in result['decisions'].items():
        fit_only=sorted([entry for entry in entries if entry['direction'] in ('shared',direction)],
                        key=lambda entry:(entry['fit_error'],entry['variant']!='antisymmetric'))[0]
        check('fit_only_selection_'+direction, fit_only==decision)
    actual=result['decisions']['actual_cached_hold']
    check('native_selected_guard_exact_witness', all(actual[key]==control['last'][key]
        for key in ('child','parent','variant','guard_gain','average_guard_gain','max_context_harm',
                    'average_max_context_harm','accepted')))
    check('native_baseline_exact_witness', abs(float(baselines['fit'].mean())-control['last']['fit_error']) < 1e-12)
    check('bounded_gpu_witnesses', all(result[name] is True for name in
        ('qualified_gpu_capture','native_baseline_and_selected_guard_replay_exact','full_checkpoint_unchanged',
         'global_rng_unchanged','gradient_buffers_unchanged','input_and_source_unchanged')))
    candidate_summary={direction:dict(total=sum(entry['direction'] in ('shared',direction) for entry in entries),
        fit_improved=sum(entry['fit_pass'] for entry in entries if entry['direction'] in ('shared',direction)),
        any_guard_passing=sum(entry['guard_pass'] for entry in entries if entry['direction'] in ('shared',direction)),
        any_fully_eligible=sum(entry['accepted'] for entry in entries if entry['direction'] in ('shared',direction)))
        for direction in result['decisions']}
    review=dict(qualified=True,gpu_used=False,checks=checks,maximum_scalar_recompute_error=maximum_scalar_error,
        maximum_CPU_CUDA_delta_relative_error=maximum_delta_relative_error,
        decisions=result['decisions'],candidate_summary=candidate_summary,parent_gradient_cosines=result['parent_gradient_cosines'],
        limits='CPU recomputes eligibility, candidate radius/direction, all feature-gain/harm flags and fit-only decisions. CUDA BF16 full-model losses, gradients and unchanged full state are qualified capture witnesses; CPU does not rerun the host.')
    (args.output/'independent-review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2))


if __name__ == '__main__':
    main()
