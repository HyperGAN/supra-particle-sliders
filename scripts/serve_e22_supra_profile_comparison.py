#!/usr/bin/env python3
"""Read-only live comparison of the PR223 shared profile and archived V2.

Only named experiment artifacts are exposed. This dashboard neither imports
the training framework nor reads model checkpoints or changes training state.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra · PR223 shared profile comparison</title>
<style>
:root{font:16px system-ui,sans-serif;color:#eaf0f8;background:#101723}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0 0 9px}.muted,footer{color:#9babbe;font-size:13px;line-height:1.6}header{display:flex;justify-content:space-between;align-items:center;gap:20px}.pill{border:1px solid #3b516d;border-radius:25px;padding:8px 14px;white-space:nowrap}.cards,.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.charts{grid-template-columns:1fr 1fr;margin-bottom:0}.card,.panel{background:#162131;border:1px solid #2b3b51;border-radius:14px;padding:20px}.panel{margin:18px 0}.label{font-size:13px;color:#9babbe}.value{font-size:29px;margin:8px 0}.delta{color:#a5ceee;font-size:13px;line-height:1.5}.bar{background:#2b3b51;height:5px;border-radius:3px;margin-top:15px}.bar span{display:block;background:#62b6f0;height:100%;border-radius:3px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin-top:14px}.legend span:before{content:'';display:inline-block;width:22px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}label{display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:13px}.chart svg{display:block;width:100%}table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}th,td{padding:11px 7px;border-bottom:1px solid #2b3b51}th{font-size:12px;color:#9babbe}th:first-child,td:first-child{text-align:left}.activity{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;color:#c2d2e4;max-height:260px;overflow:auto}.error{color:#f0b7a4}a{color:#9dd4ff}footer{margin-top:15px}@media(max-width:700px){body{padding:16px}.cards,.charts,.activity{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:13px}th,td{font-size:12px;padding:8px 3px}}
</style>
<header><div><h1>Supra · PR223 shared profile</h1><div class="muted">Matched 1,600-update comparison · particles retained at all 71 sites · refreshes every 5 seconds</div></div><div id="status" class="pill">Waiting for run</div></header>
<div class="cards"><div class="card"><div class="label">Completed updates</div><div id="updates" class="value">— / 1,600</div><div id="probe" class="muted">Waiting for the first fixed probe</div><div class="bar"><span id="bar"></span></div></div><div class="card"><div class="label">Edit · frozen-critic G game</div><div id="edit" class="value">—</div><div id="editDelta" class="delta">Waiting for a matched old/new checkpoint</div></div><div class="card"><div class="label">Preservation · frozen-critic G game</div><div id="preservation" class="value">—</div><div id="preservationDelta" class="delta">Waiting for a matched old/new checkpoint</div></div></div>
<div class="panel"><h2>Does the shared profile improve the game?</h2><div class="muted">Archived f459 ParticleGAN and PR223 share the V2 adapter, probe contexts, frozen V2 step-1,856 critic, and paired output noise. Particle control curves also share the private DV12 replay stream. Lower generator game loss is better. Solid lines show clean serving; dashed lines show DV12 perturbations. The gold ordinary-LoRA reference uses its historical AdamW/MSE formulation.</div><div class="legend"><span style="--color:#94d5b1">Old ParticleGAN · f459 · V2 control</span><span style="--color:#62b6f0">PR223 · shared profile</span><span style="--color:#f2c474">Original ordinary LoRA</span></div><div style="margin-top:14px"><label><input id="clean" type="checkbox" checked onchange="render(current)">Clean</label><label><input id="dv12" type="checkbox" checked onchange="render(current)">DV12</label></div><div id="references" class="muted" style="margin-top:10px">Waiting for archived control probes</div><div class="charts"><div id="editChart" class="chart"></div><div id="preservationChart" class="chart"></div></div></div>
<div class="panel"><h2>Latest rolling training game</h2><div class="muted">Last 100 updates of each task, separately. G/D game values are shown before the preservation task weight; critic penalty and bank gradient retain native units. These use the changing training critic.</div><table><thead><tr><th>Task</th><th>Updates</th><th>G game</th><th>D game</th><th>Critic penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table></div>
<div class="panel"><h2>Backend and game controls</h2><div id="backend" class="muted">Waiting for runtime backend evidence</div><div class="activity"><div><div class="label">Routed structural activity</div><pre id="routing">Not recorded yet</pre></div><div><div class="label">Settled reopen guard</div><pre id="guard">Not recorded yet</pre></div></div><div class="muted">The shared profile requests automatic backend selection. This run uses native routed ownership when reported below; Atlas cells are not implied by the profile name.</div></div>
<div class="panel"><h2>Final fixed-budget evaluation</h2><div id="finalStatus" class="muted">Waiting for all 1,600 updates and complete evaluation. Progress probes do not stop training or select a checkpoint.</div><div style="overflow-x:auto"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Judge</th><th>Old f459 G</th><th>PR223 G</th><th>New − old</th></tr></thead><tbody id="finalMetrics"></tbody></table></div><div class="muted" style="margin-top:16px">Output RMSE · final diagnostic only</div><table><thead><tr><th>Pool</th><th>Old f459</th><th>PR223</th><th>New − old</th></tr></thead><tbody id="finalOutput"></tbody></table></div>
<footer id="updated">Waiting for first refresh</footer><footer>Learned critic scores are endogenous measures of this game. Output errors remain evaluation only. This single task and stream do not establish general superiority. The existing dashboard remains at <a href="http://pop-os:8765">pop-os:8765</a>.</footer>
<script>
let current=null;const num=(x,d=6)=>Number.isFinite(x)?x.toFixed(d):'—', signed=x=>(x>0?'+':'')+num(x),ns='http://www.w3.org/2000/svg';
function svgNode(tag,attrs={},text=''){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function plot(id,title,series,max){const host=document.getElementById(id),svg=svgNode('svg',{viewBox:'0 0 530 300',role:'img','aria-label':title+' frozen-critic generator game loss'}),points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max);svg.append(svgNode('text',{x:68,y:22,fill:'#eaf0f8','font-size':16},title));if(!points.length){svg.append(svgNode('text',{x:68,y:130,fill:'#9babbe'},'Waiting for recorded probes'));host.replaceChildren(svg);return}let lo=Math.min(...points.map(p=>p.y)),hi=Math.max(...points.map(p=>p.y)),pad=Math.max(.001,(hi-lo)*.12);lo=Math.max(0,lo-pad);hi+=pad;const X=x=>68+x/max*444,Y=y=>242-(y-lo)/(hi-lo)*190;
for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4,x=max*i/4;svg.append(svgNode('line',{x1:68,x2:512,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),svgNode('text',{x:60,y:Y(y)+4,fill:'#9babbe','text-anchor':'end','font-size':11},num(y,3)),svgNode('text',{x:X(x),y:263,fill:'#9babbe','text-anchor':'middle','font-size':11},Math.round(x).toLocaleString()))}svg.append(svgNode('text',{x:68,y:39,fill:'#9babbe','font-size':11},'G game loss · common frozen critic'),svgNode('text',{x:290,y:287,fill:'#9babbe','text-anchor':'middle','font-size':12},'Completed updates'));
for(const s of series){const p=s.points.filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max).sort((a,b)=>a.x-b.x);if(!p.length)continue;svg.append(svgNode('path',{d:p.map((v,i)=>(i?'L':'M')+X(v.x)+','+Y(v.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dashed?'6 4':'none'}));for(const v of p){const dot=svgNode('circle',{cx:X(v.x),cy:Y(v.y),r:3,fill:s.color});dot.append(svgNode('title',{},s.label+' · update '+v.x+' · '+num(v.y)));svg.append(dot)}}host.replaceChildren(svg)}
function tableRow(values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}return tr}
function score(row,task,mode='clean'){return row?.probes?.[task]?.[mode]?.frozen_start_D}
function render(s){if(!s)return;const p=s.latest||s.history.at(-1),status=s.status||{},step=status.step??p?.step??0,horizon=status.steps??s.horizon??1600;document.getElementById('status').textContent=status.phase==='complete'?'Complete':step>=horizon?'Evaluating':p?'Running':'Preparing';document.getElementById('updates').textContent=step.toLocaleString()+' / '+horizon.toLocaleString();document.getElementById('bar').style.width=Math.min(100,100*step/horizon)+'%';document.getElementById('probe').textContent=p?'Latest fixed probe: '+p.step.toLocaleString()+' · training noise '+num(p.training_output_sigma,3):'Waiting for the first fixed probe';
const control=s.control||[],history=s.history||[],gold=s.references?.curves||s.references?.series||[];document.getElementById('references').textContent=control.length?'Archived V2 control: '+control.length+' checkpoints · original LoRA: '+gold.reduce((n,c)=>n+c.points.filter(p=>p.step<=horizon).length,0)+' checkpoints':'Waiting for the archived V2 control probes';
for(const task of ['edit','preservation']){document.getElementById(task).textContent=num(score(p,task));const matches=history.filter(r=>control.some(o=>o.step===r.step)).sort((a,b)=>b.step-a.step),matched=matches[0],old=matched&&control.find(o=>o.step===matched.step),delta=score(matched,task)-score(old,task);document.getElementById(task+'Delta').textContent=Number.isFinite(delta)?'Matched update '+matched.step+': new − old '+signed(delta)+' · '+(delta<0?'shared profile lower':delta>0?'old control lower':'equal'):'Waiting for a matched old/new checkpoint';const series=[];for(const [points,label,color]of [[control,'Old ParticleGAN','#94d5b1'],[history,'PR223 shared','#62b6f0']])for(const mode of ['clean','dv12'])if(document.getElementById(mode).checked)series.push({label:label+' · '+mode,color,dashed:mode==='dv12',points:points.map(r=>({x:r.step,y:score(r,task,mode)}))});if(document.getElementById('clean').checked)for(const reference of gold)series.push({label:'Original ordinary LoRA',color:'#f2c474',points:reference.points.map(r=>({x:r.step,y:score(r,task)}))});plot(task+'Chart',task==='edit'?'Editing':'Preservation',series,horizon)}
const rolling=status.rolling??p?.rolling??{};document.getElementById('rolling').replaceChildren(...['edit','preservation'].filter(n=>rolling[n]).map(n=>{const r=rolling[n];return tableRow([n==='edit'?'Edit':'Preservation',r.updates,num(r.g_game),num(r.d_game),num(r.penalty),Number.isFinite(r.bank_grad_norm)?r.bank_grad_norm.toExponential(3):'—'])}));
const backend=s.backend_selection;document.getElementById('backend').textContent=backend?'Requested: '+(backend.requested_backend??backend.requested??'auto')+' · Actual: '+(backend.actual_backend??'pending')+' · Sampling: '+(backend.sampling_backend??'pending')+' · Profile: '+(s.profile??'shared'):'Profile: '+(s.profile??'shared')+' · Requested: auto · Actual: '+(s.backend??'pending');for(const[id,value]of [['routing',s.routed_diagnostics],['guard',s.guard_activity??s.reopen_guard]])document.getElementById(id).textContent=value==null?'Not recorded yet':JSON.stringify(value,null,2).slice(0,12000);
document.getElementById('finalStatus').textContent=s.final.qualified?'Fixed-budget old/new evaluation recorded; runtime qualification and evaluation immutability checks passed. Both judges score both adapters. RMSE is diagnostic only.':s.final.available?'Evaluation written; waiting for its final runtime receipt.':step>=horizon?'Training horizon reached; final evaluation is pending.':'Waiting for all '+horizon.toLocaleString()+' updates and complete evaluation. Progress probes do not stop training or select a checkpoint.';const comparisons=s.final.comparison||{};document.getElementById('finalMetrics').replaceChildren(...['fit','test','holds','preservation'].filter(n=>comparisons[n]).flatMap(n=>['fixed_start_D','arm_final_D'].map(judge=>{const r=comparisons[n],g=r[judge]||{};return tableRow([n,r.count,judge==='fixed_start_D'?'Frozen at 1,856':'PR223 final at 1,600',num(g.old_g_game),num(g.new_g_game),signed(g.new_minus_old)])})));document.getElementById('finalOutput').replaceChildren(...['fit','test','holds','preservation'].filter(n=>comparisons[n]).map(n=>{const r=comparisons[n];return tableRow([n,num(r.old_rmse),num(r.new_rmse),signed(r.rmse_change)])}));document.getElementById('updated').textContent='Updated '+new Date().toLocaleTimeString()+' · read-only experiment files';}
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'}),s=await r.json();if(!r.ok){document.getElementById('status').textContent='Waiting for run';document.getElementById('updated').textContent=s.message||'Progress not ready';return}current=s;render(s)}catch(e){document.getElementById('status').textContent='Refresh unavailable';document.getElementById('updated').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></html>'''


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_history(path):
    try:
        content = path.read_text()
    except FileNotFoundError:
        return []
    result = []
    for line in content.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live writer may not have finished its final line.
        if isinstance(row, dict):
            result.append(row)
    return result


def last_training_row(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    return None


def first_field(name, *records):
    for record in records:
        if isinstance(record, dict) and record.get(name) is not None:
            return record[name]
    return None


def snapshot(run, baselines):
    status = read_json(run / 'status.json')
    history = read_history(run / 'progress.jsonl')
    latest = read_json(run / 'progress-latest.json') or (history[-1] if history else None)
    plan = read_json(run / 'plan.json') or {}
    config = (read_json(run / 'run.json') or {}).get('config', {})
    profile = read_json(run / 'profile.json') or {}
    receipt = read_json(run / 'receipt.json') or {}
    evaluation = read_json(run / 'evaluation-new.json')
    old_evaluation = read_json(run / 'evaluation-old.json')
    training = last_training_row(run / 'train.jsonl') or {}
    control = read_json(run / 'comparison-control-progress.json')
    if isinstance(control, dict):
        control = control.get('progress', control.get('points', control.get('rows', [])))
    ready = bool(status or latest)
    records = (status, profile, training, receipt, plan)
    routed = first_field('routed_diagnostics', *records) or first_field('routing', *records)
    if routed is None and training.get('move') is not None:
        routed = dict(latest_update=training.get('step'), latest_structural_result=training['move'])
    checks = receipt.get('checks') or {}
    qualified = bool(receipt.get('qualified') is True and checks
                     and all(value is True for value in checks.values())
                     and receipt.get('evaluation_state_unchanged') is True
                     and status and status.get('phase') == 'complete')
    guard_activity = first_field('guard_activity', *records) or {}
    return ready, dict(
        run=str(run), updated_at=datetime.now(timezone.utc).isoformat(), status=status,
        horizon=plan.get('fixed_updates', 1600), history=history, latest=latest,
        control=control if isinstance(control, list) else [], references=read_json(baselines),
        profile=first_field('particle_profile', status, config, plan, profile) or first_field('profile', plan, profile),
        control_commit=plan.get('control_commit'), backend=first_field('backend', *records),
        backend_selection=first_field('backend_selection', *records),
        routed_diagnostics=routed, reopen_guard=first_field('reopen_guard', guard_activity, *records),
        guard_activity=guard_activity or None,
        final=dict(available=evaluation is not None and old_evaluation is not None, qualified=qualified,
                   comparison=receipt.get('comparison') or (status or {}).get('comparison'),
                   new=evaluation, old=old_evaluation),
        evaluation_only=True, output_metrics_used_for_optimization_or_selection=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/e22-pr223-shared-1600')
    parser.add_argument('--baselines', type=Path,
                        default=ROOT / 'outputs/e22-dashboard-baselines/fixed-reference-baselines.json')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    run = args.run.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                self.respond(200, HTML.encode(), 'text/html; charset=utf-8')
                return
            supported = {'/api/state', '/api/status', '/api/progress', '/api/history',
                         '/api/control', '/api/references', '/api/final'}
            if path not in supported:
                self.send_error(404)
                return
            ready, state = snapshot(run, args.baselines)
            fields = {'/api/status': 'status', '/api/progress': 'latest', '/api/history': 'history',
                      '/api/control': 'control', '/api/references': 'references', '/api/final': 'final'}
            payload = state if path == '/api/state' else state[fields[path]]
            unavailable = ((path == '/api/state' and not ready)
                           or (path == '/api/history' and not state['history']) or payload is None)
            if unavailable:
                self.respond(503, json.dumps(dict(message='Experiment progress is not ready', run=str(run))).encode())
            else:
                self.respond(200, json.dumps(payload, allow_nan=False).encode())

        def respond(self, status, body, kind='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps(dict(event='profile_comparison_dashboard_ready', host=args.host,
                          port=args.port, run=str(run), baselines=str(args.baselines))), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
