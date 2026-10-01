#!/usr/bin/env python3
"""Serve the live particle training dashboard; only progress files are exposed."""
import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra particle training</title>
<style>
:root{font:16px system-ui,sans-serif;color:#e9eef7;background:#101723}body{margin:auto;padding:30px;max-width:1120px}
h1{font-size:26px;margin:0 0 8px}.muted{color:#99abc4;font-size:14px}header{display:flex;justify-content:space-between;gap:20px;align-items:center}
.pill{border:1px solid #37516d;border-radius:30px;padding:8px 14px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.card,.panel{border:1px solid #2b3b51;background:#162131;border-radius:14px;padding:20px}.label{color:#99abc4;font-size:13px}.value{font-size:29px;margin:6px 0}.trend{font-size:13px;color:#92d1c2}
.bar{height:5px;background:#26374f;border-radius:4px;margin-top:14px}.bar span{display:block;height:100%;background:#62b6f0;border-radius:4px}
.panel{margin:18px 0}.panel h2{font-size:18px;margin:0 0 8px}img{width:100%;border-radius:8px;background:white;margin-top:12px}
table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}td,th{padding:12px 8px;border-bottom:1px solid #2b3b51}th{font-size:13px;color:#99abc4}td:first-child,th:first-child{text-align:left}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}.chart svg{width:100%;display:block}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:12px 0}.legend span:before{content:'';display:inline-block;width:18px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}.target{margin-top:10px;font-size:14px;color:#f2c474}select{background:#162131;color:#e9eef7;border:1px solid #37516d;padding:6px;border-radius:6px}.error{color:#efb099}footer{margin:18px 0;font-size:13px;color:#99abc4}@media(max-width:650px){body{padding:18px}.cards,.charts{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:12px}td,th{padding:9px 4px;font-size:12px}}
</style>
<header><div><h1>Supra particle training</h1><div class="muted">Shared particles, native game · refreshes every 5 seconds</div></div><div id="status" class="pill">Connecting</div></header>
<div class="cards"><div class="card"><div class="label">Completed updates</div><div id="updates" class="value">—</div><div id="probeStep" class="muted">Waiting for progress</div><div class="bar"><span id="bar"></span></div></div>
<div class="card"><div class="label">Edit game loss · frozen critic</div><div id="edit" class="value">—</div><div id="editTrend" class="trend">Waiting for probes</div></div>
<div class="card"><div class="label">Preservation game loss · frozen critic</div><div id="preservation" class="value">—</div><div id="preservationTrend" class="trend">Waiting for probes</div></div></div>
<div class="panel"><h2>Can the particles beat the original LoRA?</h2><div class="muted">Saved original LoRA checkpoints and the current particle run use the same training probes, frozen critic and paired noise. Lower game loss is better. Dashed blue: particle perturbations. Dotted gold: the original 6,400-update target.</div><div id="referenceStatus" class="target">Loading the original formulation reference…</div><div class="legend"><span style="--color:#62b6f0">Current particles</span><span style="--color:#f2c474">Original LoRA · reference to beat</span></div><label class="muted">Graph range <select id="range" onchange="refresh()"><option value="all">All updates</option><option value="current">From current run start</option><option value="recent">Last 1,200 updates</option></select></label><div class="charts"><div id="editChart" class="chart"></div><div id="preservationChart" class="chart"></div></div><div id="target" class="target"></div></div>
<div class="panel"><h2>Original formulation’s training curve</h2><div class="muted">Archived ordinary LoRA training, through 6,400 updates. Each line averages the last 100 updates of that task. Its MSE objective has different units from the game losses above.</div><div class="legend"><span style="--color:#f2c474">Edit</span><span style="--color:#92d1c2">Preservation</span></div><div id="originalChart" class="chart"></div></div>
<div class="panel"><h2>Recent training losses</h2><div class="muted">Edit and preservation game values are shown in the same units, before task weighting. The live critic changes during training.</div><table><thead><tr><th>Task</th><th>Updates</th><th>G game</th><th>D game</th><th>Critic penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table></div>
<div class="panel"><h2>Final held-out comparison · 6,400 updates</h2><div id="validationStatus" class="muted">Waiting for the fixed-horizon run and complete evaluation.</div><div style="overflow-x:auto"><table><thead><tr><th>Pool</th><th>Judge</th><th>Original LoRA</th><th>Particles</th><th>Particles − original</th></tr></thead><tbody id="validation"></tbody></table></div></div>
<footer id="updated">Waiting for first refresh</footer><footer>A possible plateau means less than 0.2% net change across four probes. It is a diagnostic, not an automatic stopping rule.</footer>
<script>
const number=(x,d=6)=>Number.isFinite(x)?x.toFixed(d):'—';
function trend(p,name){const t=p.plateau_diagnostic?.[name];if(!t)return 'Collecting four probes';const pct=100*t.relative_improvement;return `${t.regressed?'Regressed':t.possible_plateau?'Possible plateau':'Improving'} · ${number(pct,2)}% decrease over ${t.steps} updates`;}
const ns='http://www.w3.org/2000/svg';
function element(tag,attrs={},text=''){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e;}
function chart(id,title,series,ylabel,options={}){const host=document.getElementById(id),svg=element('svg',{viewBox:'0 0 520 280',role:'img','aria-label':title+' '+ylabel});svg.append(element('text',{x:66,y:20,fill:'#e9eef7','font-size':16},title));const xMin=options.min??0,xMax=options.max??6400,goals=options.goals||[],visible=p=>Number.isFinite(p.y)&&p.x>=xMin&&p.x<=xMax,points=series.flatMap(s=>s.points).filter(visible);if(!points.length){svg.append(element('text',{x:66,y:120,fill:'#99abc4'},'Waiting for recorded checkpoints'));host.replaceChildren(svg);return;}
const values=[...points.map(p=>p.y),...goals.map(g=>g.y)];let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.12,.001);lo=Math.max(0,lo-pad);hi+=pad;const X=x=>66+(x-xMin)/(xMax-xMin)*438,Y=y=>225-(y-lo)/(hi-lo)*185;
for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4;svg.append(element('line',{x1:66,x2:504,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),element('text',{x:58,y:Y(y)+4,fill:'#99abc4','text-anchor':'end','font-size':11},number(y,3)));const x=xMin+(xMax-xMin)*i/4;svg.append(element('text',{x:X(x),y:244,fill:'#99abc4','text-anchor':'middle','font-size':11},Math.round(x).toLocaleString()));}
svg.append(element('text',{x:285,y:267,fill:'#99abc4','text-anchor':'middle','font-size':12},'Completed updates'),element('text',{x:66,y:36,fill:'#99abc4','font-size':11},ylabel));
for(const g of goals){const line=element('line',{x1:66,x2:504,y1:Y(g.y),y2:Y(g.y),stroke:g.color,'stroke-width':1.5,'stroke-dasharray':'2 5'});line.append(element('title',{},g.label+': '+number(g.y)));svg.append(line);}
for(const s of series){const p=s.points.filter(visible).sort((a,b)=>a.x-b.x);if(!p.length)continue;svg.append(element('path',{d:p.map((v,i)=>(i?'L':'M')+X(v.x)+','+Y(v.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dashed?'5 4':'none'}));for(const v of p){const c=element('circle',{cx:X(v.x),cy:Y(v.y),r:3,fill:s.color});c.append(element('title',{},s.label+' · update '+v.x+': '+number(v.y)));svg.append(c);}}
host.replaceChildren(svg);}
async function refresh(){try{const [s,p,references,historyText,validation]=await Promise.all(['/api/status','/api/progress','/api/references','/api/history','/api/comparison'].map(async u=>{const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw Error('Progress not ready');return u==='/api/history'?r.text():r.json()}));const history=historyText.trim().split('\n').filter(Boolean).map(x=>JSON.parse(x));
document.querySelector('#status').textContent=s.phase==='complete'?'Training complete':s.step===s.steps?'Evaluating':'Running';document.querySelector('#updates').textContent=`${s.step.toLocaleString()} / ${s.steps.toLocaleString()}`;document.querySelector('#bar').style.width=`${100*s.step/s.steps}%`;
document.querySelector('#probeStep').textContent=`Latest fixed probe: update ${p.step.toLocaleString()} · noise ${number(p.training_output_sigma,3)}`;
for(const name of ['edit','preservation']){document.querySelector('#'+name).textContent=number(p.probes[name].clean.frozen_start_D);document.querySelector('#'+name+'Trend').textContent=trend(p,name);}
document.querySelector('#rolling').replaceChildren(...['edit','preservation'].filter(n=>p.rolling[n]).map(n=>{const a=p.rolling[n],tr=document.createElement('tr');for(const v of [n==='edit'?'Edit':'Preservation',a.updates,number(a.g_game),number(a.d_game),number(a.penalty),a.bank_grad_norm.toExponential(3)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}return tr}));
const range=document.querySelector('#range').value;for(const name of ['edit','preservation']){const series=[{label:'Current particles · clean',color:'#62b6f0',points:history.map(r=>({x:r.step,y:r.probes[name].clean.frozen_start_D}))},{label:'Current particles · DV12',color:'#62b6f0',dashed:true,points:history.map(r=>({x:r.step,y:r.probes[name].dv12.frozen_start_D}))}];for(const r of references.fixed?.series||[])series.push({label:r.label,color:'#f2c474',points:r.points.map(p=>({x:p.step,y:p.probes[name].clean.frozen_start_D}))});const goals=(references.fixed?.series||[]).flatMap(r=>r.points.filter(p=>p.step===6400).map(p=>({label:'Original LoRA at 6,400 updates',color:'#f2c474',y:p.probes[name].clean.frozen_start_D})));chart(name+'Chart',name==='edit'?'Editing':'Preservation',series,'G game loss · frozen critic',{min:range==='recent'?Math.max(0,s.step-1200):range==='current'?history[0].step:0,max:range==='recent'?s.step:s.steps,goals});}
const fixed=references.fixed,baseline=fixed?.series?.[0],count=baseline?.points?.length||0;document.querySelector('#referenceStatus').textContent=count?`Original reference: ${count} saved checkpoints scored${fixed.complete?' · complete':' · still scoring'}`:'Scoring archived original checkpoints under the same frozen critic…';const end=baseline?.points?.find(p=>p.step===6400);document.querySelector('#target').textContent=end?`Original at 6,400 updates: edit ${number(end.probes.edit.clean.frozen_start_D)}, preservation ${number(end.probes.preservation.clean.frozen_start_D)}. Compare both; an edit win alone does not establish a preservation win.`:'';
chart('originalChart','Original LoRA · archived training loss',[{label:'Edit',color:'#f2c474',points:references.training.points.map(p=>({x:p.step,y:p.edit}))},{label:'Preservation',color:'#92d1c2',points:references.training.points.map(p=>({x:p.step,y:p.preservation}))}],'Unweighted MSE · rolling 100 per task');
if(validation){document.querySelector('#validationStatus').textContent='Complete held-out pools, four shared paired-noise panels and two common judges. Lower generator game loss is better; a positive difference favors the original. These scores are distinct from the small training probes above.';document.querySelector('#validation').replaceChildren(...['test','preservation'].flatMap(pool=>['fixed_start_D','arm_final_D'].map(judge=>{const a=validation[pool][judge],tr=document.createElement('tr');for(const v of [pool==='test'?`Editing · ${validation[pool].count} contexts`:`Preservation · ${validation[pool].count} contexts`,judge==='fixed_start_D'?'Frozen at 1,856':'Final at 6,400',number(a.original_g_game),number(a.particle_g_game),(a.particle_minus_original>0?'+':'')+number(a.particle_minus_original)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}return tr})));}
document.querySelector('#updated').textContent='Updated '+new Date().toLocaleTimeString()+' · probe state and training RNGs verified unchanged';
}catch(e){document.querySelector('#status').textContent='Waiting for progress';document.querySelector('#updated').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></html>"""


def original_training_curve(path, horizon=6400):
    """Show the original recorded objective separately from game comparisons."""
    pools = {False: deque(maxlen=100), True: deque(maxlen=100)}
    points = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["step"] > horizon:
            continue
        pools[row["hold"]].append(float(row["mse"]))
        if row["step"] % 25 == 0:
            points.append(dict(step=row["step"], **{
                name: sum(pools[hold]) / len(pools[hold]) if pools[hold] else None
                for name, hold in (("edit", False), ("preservation", True))}))
    return dict(label="Original ordinary LoRA", objective="unweighted MSE", window=100,
                source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--baselines", type=Path, default=ROOT / "outputs/e22-dashboard-baselines/fixed-reference-baselines.json")
    parser.add_argument("--original-log", type=Path, default=Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/train.jsonl"))
    args = parser.parse_args()
    run = args.run.resolve()
    training = original_training_curve(args.original_log)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                data, kind = HTML.encode(), "text/html; charset=utf-8"
            elif path == "/api/references":
                try:
                    fixed = json.loads(args.baselines.read_text())
                    fixed["series"] = fixed.get("series", fixed.get("curves", []))
                except FileNotFoundError:
                    fixed = None
                data, kind = json.dumps(dict(training=training, fixed=fixed)).encode(), "application/json"
            elif path == "/api/comparison":
                try:
                    receipt = json.loads((run / "evaluation-original-6400-game/receipt.json").read_text())
                    comparison = receipt["comparison"] if receipt.get("evaluation_state_unchanged") else None
                except FileNotFoundError:
                    comparison = None
                data, kind = json.dumps(comparison).encode(), "application/json"
            else:
                mapped = {"/api/status": ("status.json", "application/json"),
                          "/api/progress": ("progress-latest.json", "application/json"),
                          "/api/history": ("progress.jsonl", "application/x-ndjson"),
                          "/plot.png": ("progress.png", "image/png")}.get(path)
                if mapped is None:
                    self.send_error(404)
                    return
                filename, kind = mapped
                try:
                    data = (run / filename).read_bytes()
                except FileNotFoundError:
                    self.send_error(503, "Progress not ready")
                    return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps(dict(event="dashboard_ready", host=args.host, port=args.port, run=str(run))), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
