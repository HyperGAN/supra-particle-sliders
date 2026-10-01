#!/usr/bin/env python3
"""Read-only fresh 400-update gated-particle experiment dashboard."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

try:
    from .serve_e22_supra_bank_game_trust import read_json, read_history, read_metrics
except ImportError:
    from serve_e22_supra_bank_game_trust import read_json, read_history, read_metrics


ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra · fresh particle gating · 400 updates</title>
<style>
:root{font:16px system-ui,sans-serif;color:#eaf0f8;background:#101723}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0 0 9px}.muted,footer{color:#9babbe;font-size:13px;line-height:1.6}header{display:flex;justify-content:space-between;gap:20px;align-items:center}.pill{border:1px solid #3b516d;border-radius:24px;padding:8px 14px;white-space:nowrap}.cards,.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.charts{grid-template-columns:1fr 1fr;margin-bottom:0}.card,.panel{background:#162131;border:1px solid #2b3b51;border-radius:14px;padding:20px}.panel{margin:18px 0}.label{font-size:13px;color:#9babbe}.value{font-size:28px;margin:8px 0}.delta{color:#a5ceee;font-size:13px;line-height:1.5}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend span:before{content:'';display:inline-block;width:22px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}label{font-size:13px;margin-right:16px}.chart svg{display:block;width:100%}table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}th,td{padding:11px 7px;border-bottom:1px solid #2b3b51}th{font-size:12px;color:#9babbe}th:first-child,td:first-child{text-align:left}a{color:#9dd4ff}footer{margin-top:15px}.overflow{overflow-x:auto}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#c2d2e4;line-height:1.5;max-height:240px;overflow:auto}@media(max-width:700px){body{padding:16px}.cards,.charts{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:13px}th,td{font-size:12px;padding:8px 3px}}
</style>
<header><div><h1>Supra · fresh particle gating · 400 updates</h1><div class="muted">V3 gate vs native V2 · both fresh initialization · same shared ParticleGAN profile</div></div><div id="status" class="pill">Preparing</div></header>
<div class="panel"><h2>Ordinary LoRA remains the overall reference to beat</h2><div class="muted">Our best particle result is V2 trained for 6,400 updates. This fresh 400-update test changes the particle-conditioned branch formula and retains the particles. Both arms use the same PR223 shared profile. It is separate from the 6,400-update benchmark and the late bank-trust continuation.</div><div id="references" class="muted" style="margin-top:8px">Awaiting compatible ordinary LoRA probes under the common step-1,856 critic.</div></div>
<div class="cards"><div class="card"><div class="label">Fresh V3 completed updates</div><div id="updates" class="value">— / 400</div><div id="probe" class="muted">Waiting for the first probe</div></div><div class="card"><div class="label">Edit · clean G game</div><div id="edit" class="value">—</div><div id="editDelta" class="delta">Matched V2 difference appears at 400</div></div><div class="card"><div class="label">Preservation · clean G game</div><div id="preservation" class="value">—</div><div id="preservationDelta" class="delta">Matched V2 difference appears at 400</div></div></div>
<div class="panel"><h2>Fresh V3 progress vs matched 400-update references</h2><div class="muted">Common frozen V2 step-1,856 critic · identical probe contexts and private paired output-noise panels. The particle probes use a common V2 step-2 DV12 replay stream. Lower G game loss is better. Solid: clean serving. Dashed: DV12. Green: measured native V2 endpoint at 400. Gold: ordinary LoRA at the same 400-update budget.</div><div class="legend"><span style="--color:#f2c474">Ordinary LoRA · 400-update reference</span><span style="--color:#94d5b1">V2 · shared profile · 400 endpoint</span><span style="--color:#62b6f0">V3 · particle gate · fresh 400</span></div><label><input id="clean" type="checkbox" checked onchange="render(current)">Clean</label><label><input id="dv12" type="checkbox" checked onchange="render(current)">DV12</label><div class="charts"><div id="editChart" class="chart"></div><div id="preservationChart" class="chart"></div></div><div class="muted">The V2 reference is an endpoint marker; an earlier V2 trajectory is not inferred. This page excludes step-6,400-critic scores and longer-budget endpoints.</div></div>
<div class="panel"><h2>Rolling V3 training game · changing critic</h2><div class="muted">Last 100 updates of each task. G/D games exclude the preservation task weight. Penalty and bank gradient retain native units.</div><table><thead><tr><th>Task</th><th>Updates</th><th>G game</th><th>D game</th><th>Penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table><div id="plateau" class="muted" style="margin-top:12px">Waiting for four fixed probes to report the game plateau diagnostic.</div></div>
<div class="panel"><h2>Particle connectivity and native controls</h2><div id="audit" class="muted">Awaiting the read-only 71-site gated-basis audit.</div><pre id="activity">Native controller activity pending</pre><div class="muted">FP32 branch effects and gradients show connectivity. Some small per-site changes can quantize away at the frozen BF16 host boundary. Full-model ablations below assess game dependence separately. Output error is not an optimizer or structural criterion.</div></div>
<div class="panel"><h2>Final matched 400-update evaluation</h2><div id="finalStatus" class="muted">Awaiting the fixed horizon, final evaluation and runtime receipt.</div><div class="overflow"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Common judge</th><th>V2 G</th><th>V3 G</th><th>V3 − V2</th></tr></thead><tbody id="finalMetrics"></tbody></table></div><div class="muted" style="margin-top:16px">Output RMSE · final diagnostic only</div><table><thead><tr><th>Pool</th><th>V2</th><th>V3</th><th>V3 − V2</th></tr></thead><tbody id="finalOutput"></tbody></table></div>
<div class="panel"><h2>Final V3 particle contribution · paired game ablations</h2><div class="muted">All 570 contexts, with common critics and paired noise within each ablation. Positive ablation − clean means the intact particle computation has lower game loss for this fixture. These CPU-generated panels differ from the endpoint evaluator's CUDA panels; absolute values belong to this table.</div><div class="overflow"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Common judge</th><th>Zero codes − clean</th><th>Mass-only routing − clean</th></tr></thead><tbody id="contribution"></tbody></table></div><div class="muted">Ablations are observational; they do not select, stop, or change training. Native source and random-stream checks are reported in the final receipt.</div></div>
<footer id="updated">Waiting for first refresh</footer><footer>One task and stream; the gate has not established general superiority. <a href="http://pop-os:8765">Overall LoRA vs best particle</a> · <a href="http://pop-os:8766">PR223 vs old V2 at 1,600</a> · <a href="http://pop-os:8767">Separate step-6,400 bank-trust continuation</a></footer>
<script>
let current=null;const num=(x,d=6)=>Number.isFinite(x)?x.toFixed(d):'—',signed=x=>Number.isFinite(x)?(x>0?'+':'')+num(x):'—',ns='http://www.w3.org/2000/svg';
function node(tag,attrs={},text=''){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function score(p,task,mode='clean'){return p?.probes?.[task]?.[mode]?.frozen_start_D}
function row(values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}return tr}
function plot(id,title,series,max){const host=document.getElementById(id),svg=node('svg',{viewBox:'0 0 530 300',role:'img','aria-label':title+' common step-1856 critic generator game loss'}),points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max);svg.append(node('text',{x:68,y:22,fill:'#eaf0f8','font-size':16},title));if(!points.length){svg.append(node('text',{x:68,y:135,fill:'#9babbe'},'Awaiting recorded probes'));host.replaceChildren(svg);return}let lo=Math.min(...points.map(p=>p.y)),hi=Math.max(...points.map(p=>p.y)),pad=Math.max(.001,(hi-lo)*.12);lo=Math.max(0,lo-pad);hi+=pad;const X=x=>68+x/max*444,Y=y=>242-(y-lo)/(hi-lo)*190;for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4,x=max*i/4;svg.append(node('line',{x1:68,x2:512,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),node('text',{x:60,y:Y(y)+4,fill:'#9babbe','text-anchor':'end','font-size':11},num(y,3)),node('text',{x:X(x),y:263,fill:'#9babbe','text-anchor':'middle','font-size':11},Math.round(x).toLocaleString()))}svg.append(node('text',{x:68,y:39,fill:'#9babbe','font-size':11},'G game loss · frozen V2 critic at 1,856'),node('text',{x:290,y:287,fill:'#9babbe','text-anchor':'middle','font-size':12},'Completed fresh training updates'));for(const s of series){const p=s.points.filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max).sort((a,b)=>a.x-b.x);if(!p.length)continue;svg.append(node('path',{d:p.map((v,i)=>(i?'L':'M')+X(v.x)+','+Y(v.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':s.color==='#f2c474'?3:2,'stroke-dasharray':s.dashed?'6 4':'none'}));for(const v of p){const dot=node('circle',{cx:X(v.x),cy:Y(v.y),r:s.radius??3,fill:s.color});dot.append(node('title',{},s.label+' · '+v.x.toLocaleString()+' fresh updates · frozen-1856 G '+num(v.y)));svg.append(dot)}}host.replaceChildren(svg)}
function render(s){if(!s)return;const st=s.status||{},p=s.latest||s.history.at(-1),step=st.step??p?.step??0,max=s.horizon;document.getElementById('status').textContent=st.phase==='complete'?'Complete':st.phase==='evaluating'||step>=max?'Evaluating':st.phase==='preparing_control'?'Qualifying V2 control':st.phase==='training'?'Training V3':'Preparing';document.getElementById('updates').textContent=step.toLocaleString()+' / '+max.toLocaleString();document.getElementById('probe').textContent=p?'Latest fixed probe: '+p.step.toLocaleString():'Waiting for the first V3 probe';document.getElementById('references').textContent=s.gold_compatible?'Ordinary LoRA reference uses the same D1856, cached data, probe contexts and paired panels; only checkpoints through 400 are shown.':'Awaiting compatible ordinary LoRA probes under the common step-1,856 critic.';
for(const task of ['edit','preservation']){document.getElementById(task).textContent=num(score(p,task));const match=s.history.find(r=>r.step===400),delta=score(match,task)-score(s.control,task);document.getElementById(task+'Delta').textContent=Number.isFinite(delta)?'Matched 400: V3 − V2 '+signed(delta)+' · '+(delta<0?'V3 lower':delta>0?'V2 lower':'exact tie'):'Matched V2 difference appears at 400';const series=[];for(const mode of ['clean','dv12'])if(document.getElementById(mode).checked){series.push({label:'V3 particle gate · '+mode,color:'#62b6f0',dashed:mode==='dv12',points:s.history.map(r=>({x:r.step,y:score(r,task,mode)}))});if(s.control)series.push({label:'V2 shared-profile endpoint · '+mode,color:'#94d5b1',dashed:mode==='dv12',radius:5,points:[{x:400,y:score(s.control,task,mode)}]})}if(s.gold_compatible&&document.getElementById('clean').checked)for(const gold of s.gold)series.push({label:'Ordinary LoRA · matched-budget reference · clean',color:'#f2c474',radius:5,points:gold.points.filter(r=>r.step<=max).map(r=>({x:r.step,y:score(r,task)}))});plot(task+'Chart',task==='edit'?'Editing':'Preservation',series,max)}
const rolling=st.rolling??p?.rolling??{};document.getElementById('rolling').replaceChildren(...['edit','preservation'].filter(task=>rolling[task]).map(task=>{const r=rolling[task];return row([task,r.updates,num(r.g_game),num(r.d_game),num(r.penalty),Number.isFinite(r.bank_grad_norm)?r.bank_grad_norm.toExponential(3):'—'])}));document.getElementById('plateau').textContent=p?.plateau_diagnostic?'Last four frozen-critic clean probes: '+Object.entries(p.plateau_diagnostic).map(([task,r])=>task+' '+num(100*r.relative_improvement,3)+'% relative game improvement over '+r.steps+' updates'+(r.possible_plateau?' · possible plateau':r.regressed?' · regressed':'')).join('; ')+'. Diagnostic only; fixed training horizon.':'Waiting for four fixed probes to report the game plateau diagnostic.';document.getElementById('activity').textContent=st.guard_activity?JSON.stringify(st.guard_activity,null,2):'Native controller activity pending';const audit=s.audit;document.getElementById('audit').textContent=audit?'Audit at '+audit.step+' updates: '+audit.branch_count+' exact gated-basis sites · clean bank rows '+audit.clean_bank_rows+' / 128 · DV12 bank rows '+audit.dv12_bank_rows+' / 128 · '+audit.site_effects_nonzero+' final-output site interventions nonzero. State/RNG/gradients unchanged: '+audit.immutable:'Awaiting the read-only 71-site gated-basis audit.';
document.getElementById('finalStatus').textContent=s.final.runner_qualified?'Final fixed400 receipt recorded; runner initialization, replay, source, stream and export checks passed. Independent artifact review is separate. Both endpoints use both common judges.':s.final.available?'Endpoint evaluations recorded; runtime receipt pending.':'Awaiting the fixed horizon, final evaluation and runtime receipt.';const gameRows=[],outputRows=[];for(const pool of ['fit','test','holds','preservation']){const a=s.final.control?.[pool],b=s.final.gated?.[pool];if(!a&&!b)continue;for(const judge of ['fixed_start_D','arm_final_D'])gameRows.push(row([pool,a?.count??b?.count,judge==='fixed_start_D'?'Frozen V2 D1,856':'Shared V2 D400',num(a?.[judge]?.g_game),num(b?.[judge]?.g_game),signed(b?.[judge]?.g_game-a?.[judge]?.g_game)]));outputRows.push(row([pool,num(a?.rmse),num(b?.rmse),signed(b?.rmse-a?.rmse)]))}document.getElementById('finalMetrics').replaceChildren(...gameRows);document.getElementById('finalOutput').replaceChildren(...outputRows);document.getElementById('contribution').replaceChildren(...['fit','test','holds','preservation'].filter(pool=>s.contribution?.[pool]).flatMap(pool=>['common_D1856','common_D400'].map(judge=>{const r=s.contribution[pool];return row([pool,r.contexts,judge==='common_D1856'?'Frozen V2 D1,856':'Shared V2 D400',signed(r.per_arm?.zero_particle_codes?.[judge]?.g_game_delta_from_clean),signed(r.per_arm?.mass_only_routing?.[judge]?.g_game_delta_from_clean)])})));document.getElementById('updated').textContent='Updated '+new Date().toLocaleTimeString()+' · read-only experiment files';}
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'}),s=await r.json();if(!r.ok){document.getElementById('status').textContent='Preparing';document.getElementById('updated').textContent=s.message||'Experiment files not ready';return}current=s;render(s)}catch(e){document.getElementById('status').textContent='Refresh unavailable';document.getElementById('updated').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></html>'''


def compact_contribution(path):
    result = read_json(path) or {}
    return {name: dict(contexts=pool.get('contexts'), per_arm={arm:
                {key: value for key, value in metrics.items() if key not in ('records', 'per_subject')}
                for arm, metrics in pool.get('per_arm', {}).items()}) for name, pool in result.items()}


def snapshot(run, baselines):
    plan = read_json(run / 'plan.json') or {}
    status = read_json(run / 'status.json')
    history = read_history(run / 'progress.jsonl')
    latest = read_json(run / 'progress-latest.json') or (history[-1] if history else None)
    control = read_json(run / 'control-progress400.json')
    reference = read_json(baselines) or {}
    metadata = reference.get('reference', {})
    indices = {task: read_json(run / 'control-probes400' / f'progress-{task}-indices.json')
               for task in ('edit', 'preservation')}
    gold_compatible = bool(control and control.get('step') == 400 and control.get('output_sigma') == .125
        and metadata.get('critic_checkpoint_sha256') == plan.get('input_sha256', {}).get('judge')
        and metadata.get('data_sha256') == plan.get('input_sha256', {}).get('data')
        and metadata.get('output_sigma') == .125 and metadata.get('probe_indices') == indices)
    if gold_compatible:
        for task, expected in indices.items():
            actual = read_json(run / f'progress-{task}-indices.json')
            if actual is not None and actual != expected:
                gold_compatible = False
    receipt = read_json(run / 'receipt.json') or {}
    checks = receipt.get('checks') or {}
    control_metrics = read_metrics(run / 'evaluation-v2.json')
    gated_metrics = read_metrics(run / 'evaluation-v3.json')
    audit = read_json(run / 'audit-final.json') or read_json(run / 'audit-step-two.json')
    audit_summary = None
    if audit:
        audit_summary = dict(step=audit.get('step'), branch_count=len(audit.get('branches', {})),
            clean_bank_rows=audit.get('clean_gradient', {}).get('bank_rows_nonzero'),
            dv12_bank_rows=audit.get('dv12_gradient', {}).get('bank_rows_nonzero'),
            site_effects_nonzero=audit.get('final_output_site_effects_nonzero_diagnostic'),
            immutable=audit.get('native_state_rng_and_gradients_unchanged'))
    return bool(status or history), dict(run=str(run), updated_at=datetime.now(timezone.utc).isoformat(),
        plan=plan, status=status, horizon=plan.get('fixed_updates', 400), history=history, latest=latest,
        control=control, gold_compatible=gold_compatible,
        gold=[dict(label=curve.get('label'), points=[point for point in curve.get('points', [])
                                                  if point.get('step', 401) <= 400])
              for curve in reference.get('curves', [])] if gold_compatible else [],
        audit=audit_summary, contribution=compact_contribution(run / 'particle-contribution.json'),
        final=dict(available=control_metrics is not None and gated_metrics is not None,
            runner_qualified=bool(receipt.get('qualified') is True and checks
                and all(value is True for value in checks.values()) and receipt.get('source_immutable') is True
                and status and status.get('phase') == 'complete'),
            checks=checks, control=control_metrics, gated=gated_metrics, comparison=receipt.get('comparison')),
        evaluation_only=True, output_metrics_used_for_optimization_or_selection=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/e22-particle-gated-v3-400')
    parser.add_argument('--baselines', type=Path,
                        default=ROOT / 'outputs/e22-dashboard-baselines/fixed-reference-baselines.json')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8782)
    args = parser.parse_args()
    run = args.run.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                self.respond(200, HTML.encode(), 'text/html; charset=utf-8')
                return
            if path not in ('/api/state', '/api/status', '/api/history', '/api/references', '/api/final', '/api/contribution'):
                self.send_error(404)
                return
            ready, state = snapshot(run, args.baselines)
            payload = {'/api/state': state, '/api/status': state['status'], '/api/history': state['history'],
                '/api/references': dict(compatible=state['gold_compatible'], gold=state['gold'], control=state['control']),
                '/api/final': state['final'], '/api/contribution': state['contribution']}[path]
            if not ready:
                self.respond(503, json.dumps(dict(message='Gated experiment files are not ready', run=str(run))).encode())
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
