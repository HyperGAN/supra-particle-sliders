#!/usr/bin/env python3
"""Read-only dashboard for the fixed-budget, game-only bank trust experiment.

This server reads named JSON artifacts only. It does not import the training
framework, open checkpoints, or reuse scores from the old step-1856 judge.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('native', 'bank_game_trust_v1')
LABELS = {'native': 'V2 · native continuation',
          'bank_game_trust_v1': 'V2 · game-only bank trust'}
JUDGE = 'common qualified particle V2 step6400 critic'
CHECKS = ('initial_native_state_exact', 'two_update_replay_exact', 'frozen_unchanged',
          'evaluation_state_unchanged', 'source_and_inputs_unchanged', 'matched_owned_streams')

HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra · game-only particle bank continuation</title>
<style>
:root{font:16px system-ui,sans-serif;color:#eaf0f8;background:#101723}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0 0 9px}.muted,footer{color:#9babbe;font-size:13px;line-height:1.6}header{display:flex;justify-content:space-between;gap:20px;align-items:center}.pill{border:1px solid #3b516d;border-radius:24px;padding:8px 14px;white-space:nowrap}.cards,.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.charts{grid-template-columns:1fr 1fr;margin-bottom:0}.card,.panel{background:#162131;border:1px solid #2b3b51;border-radius:14px;padding:20px}.panel{margin:18px 0}.label{font-size:13px;color:#9babbe}.value{font-size:28px;margin:8px 0}.delta{color:#a5ceee;font-size:13px;line-height:1.5}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend span:before{content:'';display:inline-block;width:22px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}label{font-size:13px;margin-right:16px}.chart svg{display:block;width:100%}table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}th,td{padding:11px 7px;border-bottom:1px solid #2b3b51}th{font-size:12px;color:#9babbe}th:first-child,td:first-child{text-align:left}a{color:#9dd4ff}footer{margin-top:15px}.overflow{overflow-x:auto}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#c2d2e4;line-height:1.5}@media(max-width:700px){body{padding:16px}.cards,.charts{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:13px}th,td{font-size:12px;padding:8px 3px}}
</style>
<header><div><h1>Supra · game-only particle bank continuation</h1><div class="muted">Matched particle continuations · start at 6,400 · fixed 256 further updates</div></div><div id="status" class="pill">Preparing</div></header>
<div class="panel"><h2>Ordinary LoRA remains the overall reference to beat</h2><div class="muted">V2 at 6,400 updates is our best particle result. PR223 tied old V2 at 1,600 updates. This experiment compares two continuations from the same qualified V2 state, on the same native source and random streams. It does not establish a new overall best.</div><div id="referenceStatus" class="muted" style="margin-top:8px">Awaiting ordinary LoRA probes under this experiment's step-6,400 critic. Earlier step-1,856 scores are not plotted here.</div></div>
<div class="cards"><div class="card"><div class="label">Current arm · additional updates</div><div id="updates" class="value">— / 256</div><div id="arm" class="muted">Preparing native control</div></div><div class="card"><div class="label">Edit · matched clean G game</div><div id="edit" class="value">—</div><div id="editDelta" class="delta">Waiting for matched probes</div></div><div class="card"><div class="label">Preservation · matched clean G game</div><div id="preservation" class="value">—</div><div id="preservationDelta" class="delta">Waiting for matched probes</div></div></div>
<div class="panel"><h2>Continuation game vs ordinary LoRA reference</h2><div class="muted">Common frozen V2 step-6,400 critic · identical 12-context probes per task and four paired output-noise panels. Both particle curves share the private DV12 replay stream. Lower G game loss is better. Solid: clean serving. Dashed: DV12. Gold is the unchanged original ordinary LoRA trained for 6,400 updates, shown as a constant quality target.</div><div class="legend"><span style="--color:#f2c474">Ordinary LoRA · trained 6,400 · constant clean reference</span><span style="--color:#94d5b1">V2 · native · 6,400 → 6,656</span><span style="--color:#62b6f0">V2 · bank trust · 6,400 → 6,656</span></div><label><input id="clean" type="checkbox" checked onchange="render(current)">Clean</label><label><input id="dv12" type="checkbox" checked onchange="render(current)">DV12</label><div class="charts"><div id="editChart" class="chart"></div><div id="preservationChart" class="chart"></div></div><div class="muted">The gold line is a historical 6,400-update reference, not a model trained for 6,656 updates. Progress probes do not choose, stop, or alter training.</div></div>
<div class="panel"><h2>Rolling training game · changing training critic</h2><div class="muted">Last 100 updates of each task within this continuation, separately. G/D game values exclude the preservation task weight; critic penalty and bank gradient retain native units.</div><div class="overflow"><table><thead><tr><th>Arm / task</th><th>Updates</th><th>G game</th><th>D game</th><th>Critic penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table></div></div>
<div class="panel"><h2>Bank step projection · training game only</h2><div class="muted">Only the candidate arm scales the proposed bank displacement. Native Adam moments and the other parameter updates remain native. It tries predeclared fractions 1, ½, ¼, ⅛, 0 against the same training batch. Both clean and replayed native noisy games must avoid harm relative to the post-generator/router state with no bank displacement. Native structural feature guards remain separate.</div><div class="cards"><div class="card"><div class="label">Mean applied bank fraction</div><div id="fraction" class="value">—</div><div id="fractionNote" class="muted">Awaiting candidate updates</div></div><div class="card"><div class="label">Full bank proposals with game harm</div><div id="proposalHarm" class="value">—</div><div class="muted">Clean or native noisy game; before projection</div></div><div class="card"><div class="label">Applied bank steps with game harm</div><div id="acceptedHarm" class="value">—</div><div class="muted">Either game; zero is required by this experiment</div></div></div><pre id="fractions">No candidate records yet</pre><div class="muted">These are same-batch local checks. They do not promise held-out improvement. Output error is never used by the projection or structural criterion.</div></div>
<div class="panel"><h2>Final fixed-budget evaluation · common judges</h2><div id="finalStatus" class="muted">Awaiting both continuations and final evaluation.</div><div class="overflow"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Judge</th><th>LoRA 6,400</th><th>Native 6,656</th><th>Trust 6,656</th><th>Trust − native</th></tr></thead><tbody id="finalMetrics"></tbody></table></div><div class="muted" style="margin-top:18px">Output RMSE · final diagnostic only</div><table><thead><tr><th>Pool</th><th>LoRA 6,400</th><th>Native 6,656</th><th>Trust 6,656</th><th>Trust − native</th></tr></thead><tbody id="finalOutput"></tbody></table><pre id="qualification">Runner checks pending</pre></div>
<footer id="updated">Waiting for first refresh</footer><footer>One late continuation on one task and stream. Learned critic scores are measures of this game, not a general quality guarantee. <a href="http://pop-os:8765">Overall LoRA vs best particle benchmark</a> · <a href="http://pop-os:8766">PR223 matched 1,600-update comparison</a></footer>
<script>
let current=null;const num=(x,d=6)=>Number.isFinite(x)?x.toFixed(d):'—',signed=x=>Number.isFinite(x)?(x>0?'+':'')+num(x):'—',ns='http://www.w3.org/2000/svg',arms=['native','bank_game_trust_v1'],labels={native:'V2 native',bank_game_trust_v1:'V2 bank trust'};
function node(tag,attrs={},text=''){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function score(row,task,mode='clean'){return row?.probes?.[task]?.[mode]?.frozen_start_D}
function row(values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}return tr}
function plot(id,title,series,start,end){const host=document.getElementById(id),svg=node('svg',{viewBox:'0 0 530 300',role:'img','aria-label':title+' common step-6400 critic generator game loss'}),points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y)&&p.x>=start&&p.x<=end);svg.append(node('text',{x:68,y:22,fill:'#eaf0f8','font-size':16},title));if(!points.length){svg.append(node('text',{x:68,y:135,fill:'#9babbe'},'Awaiting recorded probes'));host.replaceChildren(svg);return}let lo=Math.min(...points.map(p=>p.y)),hi=Math.max(...points.map(p=>p.y)),pad=Math.max(.001,(hi-lo)*.12);lo=Math.max(0,lo-pad);hi+=pad;const X=x=>68+(x-start)/(end-start)*444,Y=y=>242-(y-lo)/(hi-lo)*190;for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4,x=start+(end-start)*i/4;svg.append(node('line',{x1:68,x2:512,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),node('text',{x:60,y:Y(y)+4,fill:'#9babbe','text-anchor':'end','font-size':11},num(y,3)),node('text',{x:X(x),y:263,fill:'#9babbe','text-anchor':'middle','font-size':11},Math.round(x).toLocaleString()))}svg.append(node('text',{x:68,y:39,fill:'#9babbe','font-size':11},'G game loss · frozen V2 critic at 6,400'),node('text',{x:290,y:287,fill:'#9babbe','text-anchor':'middle','font-size':12},'Completed particle updates · continuation starts at 6,400'));for(const s of series){const p=s.points.filter(p=>Number.isFinite(p.y)&&p.x>=start&&p.x<=end).sort((a,b)=>a.x-b.x);if(!p.length)continue;const path=node('path',{d:p.map((v,i)=>(i?'L':'M')+X(v.x)+','+Y(v.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':s.reference?3:2,'stroke-dasharray':s.dashed?'6 4':'none'});path.append(node('title',{},s.label));svg.append(path);for(const v of s.reference?p.slice(0,1):p){const dot=node('circle',{cx:X(v.x),cy:Y(v.y),r:s.reference?4:3,fill:s.color});dot.append(node('title',{},s.reference?s.label+' · trained 6,400; unchanged reference · frozen-6400 G '+num(v.y):s.label+' · '+v.x.toLocaleString()+' completed updates ('+(v.x-start)+' additional) · frozen-6400 G '+num(v.y)));svg.append(dot)}}host.replaceChildren(svg)}
function render(s){if(!s)return;const st=s.status||{},start=s.start_step,end=s.final_step,step=st.step??start;document.getElementById('status').textContent=st.phase==='complete'?'Both arms complete':st.phase==='training'?(step>=end?'Evaluating ':'Training ')+(labels[st.arm]||st.arm):st.phase==='loading'?'Loading native control':'Preparing';document.getElementById('updates').textContent=Math.max(0,step-start).toLocaleString()+' / '+(end-start).toLocaleString();document.getElementById('arm').textContent=st.phase==='complete'?'Both particle endpoints: '+end.toLocaleString()+' total updates':(labels[st.arm]||'Native control')+' · '+step.toLocaleString()+' total updates';document.getElementById('referenceStatus').textContent=s.original_reference_compatible?'Gold reference: original ordinary LoRA trained 6,400, re-evaluated under the common step-6,400 critic. Its score is held constant across the continuation.':'Awaiting compatible ordinary LoRA probes under this experiment’s step-6,400 critic. Earlier step-1,856 scores are not plotted here.';
for(const task of ['edit','preservation']){const native=s.arms.native,candidate=s.arms.bank_game_trust_v1,matches=candidate.history.filter(r=>native.history.some(n=>n.step===r.step)).sort((a,b)=>b.step-a.step),p=matches[0],old=p&&native.history.find(n=>n.step===p.step),delta=score(p,task)-score(old,task);document.getElementById(task).textContent=num(score(p,task));document.getElementById(task+'Delta').textContent=p?'At '+p.step.toLocaleString()+': trust − native '+signed(delta)+' · '+(delta<0?'trust lower':delta>0?'native lower':'exact tie'):'Awaiting both arms at the same probe step';const series=[];for(const [arm,color]of [['native','#94d5b1'],['bank_game_trust_v1','#62b6f0']])for(const mode of ['clean','dv12'])if(document.getElementById(mode).checked)series.push({label:labels[arm]+' · '+mode,color,dashed:mode==='dv12',points:s.arms[arm].history.map(r=>({x:r.step,y:score(r,task,mode)}))});if(document.getElementById('clean').checked&&s.original_reference_compatible){const value=score(s.original_reference.points[0],task);series.push({label:'Ordinary LoRA · original 6,400-update clean reference',color:'#f2c474',reference:true,points:[{x:start,y:value},{x:end,y:value}]})}plot(task+'Chart',task==='edit'?'Editing':'Preservation',series,start,end)}
document.getElementById('rolling').replaceChildren(...arms.flatMap(arm=>['edit','preservation'].filter(task=>s.arms[arm].rolling?.[task]).map(task=>{const r=s.arms[arm].rolling[task];return row([labels[arm]+' / '+task,r.updates,num(r.g_game),num(r.d_game),num(r.penalty),Number.isFinite(r.bank_grad_norm)?r.bank_grad_norm.toExponential(3):'—'])})));const trust=s.arms.bank_game_trust_v1.bank_trust;document.getElementById('fraction').textContent=num(trust.mean_applied_fraction,3);document.getElementById('fractionNote').textContent=trust.updates?trust.updates+' candidate updates · '+trust.zero_fraction_updates+' with no applied bank displacement':'Awaiting candidate updates';document.getElementById('proposalHarm').textContent=trust.updates?trust.full_proposal_harm_updates+' / '+trust.updates:'—';document.getElementById('acceptedHarm').textContent=trust.updates?trust.accepted_harm_updates+' / '+trust.updates:'—';document.getElementById('fractions').textContent=trust.updates?'Applied fraction counts: '+JSON.stringify(trust.fraction_counts)+'\nFull proposal harm: clean '+trust.full_proposal_clean_harm_updates+', noisy '+trust.full_proposal_noisy_harm_updates+'\nLatest applied fraction: '+num(trust.latest?.selected_fraction,3)+'; accepted game delta '+JSON.stringify(trust.latest?.accepted_new_minus_baseline):'No candidate records yet';
const final=s.final;document.getElementById('finalStatus').textContent=final.runner_qualified?'Both fixed-horizon endpoints recorded; runner state, replay, source and stream checks passed. Independent artifact review is separate. Both final adapters and the historical LoRA reference use both common judges.':final.available?'Endpoint evaluations recorded; final runtime receipt pending.':'Awaiting both continuations and final evaluation.';const gameRows=[],outputRows=[];for(const pool of ['fit','test','holds','preservation']){const old=final.native?.[pool],candidate=final.candidate?.[pool],original=final.original?.[pool];if(!old&&!candidate&&!original)continue;for(const judge of ['fixed_start_D','arm_final_D'])gameRows.push(row([pool,old?.count??candidate?.count??original?.count,judge==='fixed_start_D'?'Common frozen D at 6,400':'Common native final D at '+end.toLocaleString(),num(original?.[judge]?.g_game),num(old?.[judge]?.g_game),num(candidate?.[judge]?.g_game),signed(candidate?.[judge]?.g_game-old?.[judge]?.g_game)]));outputRows.push(row([pool,num(original?.rmse),num(old?.rmse),num(candidate?.rmse),signed(candidate?.rmse-old?.rmse)]))}document.getElementById('finalMetrics').replaceChildren(...gameRows);document.getElementById('finalOutput').replaceChildren(...outputRows);document.getElementById('qualification').textContent=Object.keys(final.checks).length?JSON.stringify(final.checks,null,2):'Runner checks pending';document.getElementById('updated').textContent='Updated '+new Date().toLocaleTimeString()+' · read-only experiment files';}
async function refresh(){try{const response=await fetch('/api/state',{cache:'no-store'}),s=await response.json();if(!response.ok){document.getElementById('status').textContent='Preparing';document.getElementById('updated').textContent=s.message||'Experiment files are not ready';return}current=s;render(s)}catch(e){document.getElementById('status').textContent='Refresh unavailable';document.getElementById('updated').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></html>'''


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_history(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_metrics(path):
    value = read_json(path)
    if not isinstance(value, dict):
        return None
    return {name: {key: item for key, item in pool.items() if key != 'records'}
            for name, pool in value.items() if isinstance(pool, dict)}


def trust_summary(records):
    """Report applied fractions and local game harm in native score units."""
    def harmed(values, mode=None):
        selected = [values.get(mode)] if mode else values.values()
        return any(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                   for value in selected)

    full = [next((candidate.get('new_minus_baseline', {})
                  for candidate in record.get('candidates', []) if candidate.get('fraction') == 1), {})
            for record in records]
    fractions = [record.get('selected_fraction') for record in records]
    fractions = [value for value in fractions if isinstance(value, (int, float)) and math.isfinite(value)]
    return dict(updates=len(records), fraction_counts={str(fraction): fractions.count(fraction)
                    for fraction in (1., .5, .25, .125, 0.)},
                mean_applied_fraction=sum(fractions) / len(fractions) if fractions else None,
                zero_fraction_updates=fractions.count(0.),
                full_proposal_harm_updates=sum(harmed(value) for value in full),
                full_proposal_clean_harm_updates=sum(harmed(value, 'clean') for value in full),
                full_proposal_noisy_harm_updates=sum(harmed(value, 'noisy') for value in full),
                accepted_harm_updates=sum(harmed(record.get('accepted_new_minus_baseline', {}))
                                          for record in records),
                latest=records[-1] if records else None)


def snapshot(run):
    plan = read_json(run / 'plan.json') or {}
    status = read_json(run / 'status.json')
    receipt = read_json(run / 'receipt.json') or {}
    arms = {}
    for name in ARMS:
        directory = run / name
        history = read_history(directory / 'progress.jsonl')
        latest = read_json(directory / 'progress-latest.json') or (history[-1] if history else None)
        arm_status = read_json(directory / 'status.json') or {}
        training = read_history(directory / 'train.jsonl')
        arm_receipt = read_json(directory / 'receipt.json') or {}
        records = arm_receipt.get('trust_rows') or [row['bank_trust'] for row in training
                                                   if isinstance(row.get('bank_trust'), dict)]
        arms[name] = dict(label=LABELS[name], status=arm_status, history=history, latest=latest,
                          rolling=arm_status.get('rolling') or (latest or {}).get('rolling') or {},
                          bank_trust=trust_summary(records), evaluation=read_metrics(directory / 'evaluation.json'))
    reference = read_json(run / 'original-reference-progress.json')
    compatible = bool(reference and reference.get('judge') == JUDGE
        and reference.get('reference_update') == plan.get('start_step') == 6400
        and reference.get('evaluation_only') is True
        and reference.get('original_sha256') == plan.get('input_sha256', {}).get('original_reference')
        and len(reference.get('points', [])) == 1 and reference['points'][0].get('step') == 6400)
    if compatible:
        for task in ('edit', 'preservation'):
            selected = read_json(run / 'reference-probes' / f'progress-{task}-indices.json')
            if not isinstance(selected, list) or reference['points'][0].get('probes', {}).get(task, {}).get('contexts') != len(selected):
                compatible = False
            for name in ARMS:
                actual = read_json(run / name / f'progress-{task}-indices.json')
                if actual is not None and actual != selected:
                    compatible = False
    checks = {name: receipt.get(name) for name in CHECKS} if receipt else {}
    original = read_metrics(run / 'evaluation-original.json')
    final = dict(available=all(arms[name]['evaluation'] is not None for name in ARMS) and original is not None,
                 runner_qualified=bool(status and status.get('phase') == 'complete'
                                       and checks and all(value is True for value in checks.values())),
                 checks=checks, native=arms['native']['evaluation'], candidate=arms['bank_game_trust_v1']['evaluation'],
                 original=original, game_deltas_candidate_minus_native=receipt.get('game_deltas_candidate_minus_native'),
                 game_gaps_to_original_6400_reference=receipt.get('game_gaps_to_original_6400_reference'))
    return bool(status or any(arm['history'] for arm in arms.values())), dict(
        run=str(run), updated_at=datetime.now(timezone.utc).isoformat(), plan=plan, status=status,
        start_step=plan.get('start_step', 6400), final_step=plan.get('final_step', 6656),
        judge=JUDGE, arms=arms, original_reference=reference if compatible else None,
        original_reference_compatible=compatible, final=final,
        evaluation_only=True, output_metrics_used_for_optimization_or_selection=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/e22-bank-game-trust-256')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    run = args.run.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                self.respond(200, HTML.encode(), 'text/html; charset=utf-8')
                return
            if path not in ('/api/state', '/api/status', '/api/history', '/api/references', '/api/final'):
                self.send_error(404)
                return
            ready, state = snapshot(run)
            payload = {'/api/state': state, '/api/status': state['status'],
                       '/api/history': {name: state['arms'][name]['history'] for name in ARMS},
                       '/api/references': dict(compatible=state['original_reference_compatible'],
                                               reference=state['original_reference']),
                       '/api/final': state['final']}[path]
            if not ready:
                self.respond(503, json.dumps(dict(message='Experiment files are not ready', run=str(run))).encode())
            else:
                self.respond(200, json.dumps(payload, allow_nan=False).encode())

        def respond(self, code, body, content_type='application/json; charset=utf-8'):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Dashboard listening on http://{args.host}:{args.port}; reading {run}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
