#!/usr/bin/env python3
"""Read-only live view of the fixed Supra sampled-initialization comparison."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


ARMS = ("sampled_control", "sampled_hb_neutral")
ENDPOINTS = (5120, 6400)


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def read_history(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A writer may still be appending the final line.
        if isinstance(row, dict):
            rows.append(row)
    return rows


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def state(run):
    """Read observation files only; never import models or training code."""
    return finite_json(dict(
        updated=datetime.now(timezone.utc).isoformat(),
        status=read_json(run / "status.json", {}),
        historical_probes=read_json(run / "historical-probes.json", {}),
        historical_references=read_json(run / "historical-references.json", {}),
        arms={arm: dict(history=read_history(run / arm / "monitor/progress.jsonl"),
                        evaluations={str(step): read_json(run / arm / f"evaluation-{step:05d}.json")
                                     for step in ENDPOINTS}) for arm in ARMS},
        receipt=read_json(run / "receipt.json"),
        failure=read_json(run / "failure.json"),
    ))


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra · sampled particle initialization</title>
<style>
:root{font:16px system-ui,sans-serif;color:#eaf0f8;background:#101723}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0 0 10px}header{display:flex;justify-content:space-between;gap:20px;align-items:center}.muted,footer{color:#a5b4c7;font-size:13px;line-height:1.6}.pill{border:1px solid #3b516d;border-radius:24px;padding:8px 14px;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:20px 0}.card,.panel{background:#162131;border:1px solid #2b3b51;border-radius:14px;padding:20px}.panel{margin:18px 0}.value{font-size:28px;margin:9px 0}.label{font-size:13px;color:#a5b4c7}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend span:before{content:'';display:inline-block;width:22px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}.chart svg{display:block;width:100%}table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}th,td{padding:10px 8px;border-bottom:1px solid #2b3b51}th{font-size:12px;color:#a5b4c7}th:first-child,td:first-child{text-align:left}.overflow{overflow-x:auto}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#c2d2e4;line-height:1.5;max-height:180px;overflow:auto}.good{color:#94d5b1}.gold{color:#f2c474}.bad{color:#f0a5a5}a{color:#9dd4ff}footer{margin-top:12px}@media(max-width:700px){body{padding:16px}.cards{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:13px}th,td{font-size:12px;padding:8px 4px}}
</style>
<header><div><h1>Supra · particle initialization comparison</h1><div class="muted">71 gated particle sites · shared 128 × 4 bank · two fresh public sampled-initialization runs</div></div><div id="status" class="pill">Preparing</div></header>
<div class="panel"><h2 class="gold">Original ordinary LoRA is the target to beat</h2><div id="outcome">Awaiting matched held-out probes.</div><div class="muted">Gold: the historical ordinary LoRA checkpoint trained for 6,400 total updates, including 5,120 editing updates. The fresh runs train on editing only through 6,400 updates. Historical references use a different initialization cohort and training schedule.</div></div>
<div class="cards"><div class="card"><div class="label">Sampled control · original H/b initialization</div><div id="controlValue" class="value">—</div><div id="controlDetail" class="muted">Waiting for a probe</div></div><div class="card"><div class="label">Sampled neutral · initial H = 0, b = 0</div><div id="neutralValue" class="value">—</div><div id="neutralDetail" class="muted">Waiting for a probe</div></div></div>
<div class="panel"><h2>Held-out progress · clean FAST · lower is better</h2><div class="muted">Same fixed D1856 critic, the same 12 held-out editing contexts, and the same four private CPU72 Gaussian panels for both curves and all horizontal reference levels. These probes do not choose or stop training.</div><div class="legend"><span style="--color:#f2c474">Ordinary LoRA · target</span><span style="--color:#62b6f0">Sampled control</span><span style="--color:#94d5b1">Sampled H/b neutral</span><span style="--color:#b491d9">Historical particle V2</span><span style="--color:#d68fb6">Historical particle V3</span></div><div id="gameChart" class="chart"></div><div id="panelIdentity" class="muted"></div><div id="plateau" class="muted" style="margin-top:10px"></div></div>
<div class="panel"><h2>Rolling training game · changing critic</h2><div class="muted">Mean of the most recent 100 editing updates. G and D training losses use each run’s live critic; their absolute values belong to this chart. A flattening training curve alone does not establish convergence.</div><div class="legend"><span style="--color:#62b6f0">Control G · solid / D · dashed</span><span style="--color:#94d5b1">Neutral G · solid / D · dashed</span></div><div id="trainingChart" class="chart"></div><div class="overflow"><table><thead><tr><th>Fresh run</th><th>Editing updates</th><th>G game</th><th>D game</th><th>KA2 penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table></div></div>
<div class="panel"><h2>Both fixed endpoints · full held-out editing set</h2><div class="muted">All 240 held-out editing contexts, independently rescored under both D1856 and D6400. Full-set means appear only in this table; the live chart uses its fixed 12-context probe. Historical ordinary LoRA has 5,120 editing updates. Both fresh runs have exactly the editing updates shown.</div><div class="overflow"><table><thead><tr><th>Editing updates</th><th>Common judge</th><th>Contexts</th><th>Ordinary target</th><th>Sampled control</th><th>H/b neutral</th><th>Neutral − control</th><th>Neutral − target</th></tr></thead><tbody id="endpoints"></tbody></table></div><div id="endpointStatus" class="muted" style="margin-top:10px"></div></div>
<div class="panel"><h2>Run verification</h2><div id="verification" class="muted">Waiting for the runtime receipt.</div><div id="activity" class="muted"></div><pre id="failure" hidden></pre><div class="muted">Both runs retain learned H/b, C, routing and particle-bank controls. Output error is not used as a particle training or structural criterion. This comparison tests one declared Supra task and initialization change.</div></div>
<footer id="updated">Waiting for refresh</footer><footer>Refreshes every 5 seconds. The dashboard reads observation files and does not access training state or GPU resources.</footer>
<script>
const names={sampled_control:'Sampled control',sampled_hb_neutral:'H/b neutral'},colors={sampled_control:'#62b6f0',sampled_hb_neutral:'#94d5b1'},NS='http://www.w3.org/2000/svg';
const num=(v,n=6)=>Number.isFinite(v)?v.toFixed(n):'—', signed=v=>Number.isFinite(v)?(v>0?'+':'')+num(v):'—', finite=v=>Number.isFinite(v);
function node(tag,attrs={},text=''){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);n.textContent=text;return n}
function row(values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}return tr}
function game(p){return p?.probes?.edit?.clean?.frozen_start_D}
function chart(id,series,refs=[],ylabel='G game loss',maximum=6400){const host=document.getElementById(id),svg=node('svg',{viewBox:'0 0 1000 350',role:'img','aria-label':ylabel+' versus completed editing updates'}),points=series.flatMap(s=>s.points).filter(p=>finite(p.y)&&finite(p.x)),levels=refs.filter(r=>finite(r.y));host.replaceChildren(svg);if(!points.length&&!levels.length){svg.append(node('text',{x:80,y:165,fill:'#a5b4c7'},'Awaiting recorded probes'));return}const values=[...points.map(p=>p.y),...levels.map(r=>r.y)];let lo=Math.min(...values),hi=Math.max(...values),padding=Math.max(.002,(hi-lo)*.12);lo-=padding;hi+=padding;const X=x=>78+x/maximum*895,Y=y=>287-(y-lo)/(hi-lo)*235;for(let i=0;i<=5;i++){let y=lo+(hi-lo)*i/5,x=maximum*i/5;svg.append(node('line',{x1:78,x2:973,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),node('text',{x:69,y:Y(y)+4,fill:'#a5b4c7','text-anchor':'end','font-size':12},num(y,3)),node('text',{x:X(x),y:310,fill:'#a5b4c7','text-anchor':'middle','font-size':12},Math.round(x).toLocaleString()))}svg.append(node('text',{x:78,y:25,fill:'#a5b4c7','font-size':13},ylabel),node('text',{x:525,y:338,fill:'#a5b4c7','text-anchor':'middle','font-size':13},'Completed editing updates · fresh runs use editing only'));for(const r of levels){svg.append(node('line',{x1:78,x2:973,y1:Y(r.y),y2:Y(r.y),stroke:r.color,'stroke-width':r.target?2.5:1.4,'stroke-dasharray':r.target?'8 4':'3 6'}));svg.append(node('text',{x:965,y:Y(r.y)-6,fill:r.color,'text-anchor':'end','font-size':12},r.name+' '+num(r.y,4)))}for(const s of series){const p=s.points.filter(p=>finite(p.y)&&finite(p.x)).sort((a,b)=>a.x-b.x);if(p.length)svg.append(node('polyline',{points:p.map(p=>X(p.x)+','+Y(p.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':2.5,'stroke-dasharray':s.dash?'6 4':''}));for(const q of p)svg.append(node('circle',{cx:X(q.x),cy:Y(q.y),r:2.3,fill:s.color}))}}
function render(s){const status=s.status||{},arms=s.arms||{},gold=s.historical_probes?.scores?.ordinary_lora_6400;document.getElementById('status').textContent=(status.phase||'preparing')+' · '+(status.editing_updates??status.step??0).toLocaleString()+' / 6,400 edits';const referenceLevels=[{name:'Ordinary LoRA target',y:gold,color:'#f2c474',target:true},{name:'Historical particle V2',y:s.historical_probes?.scores?.historical_particle_v2,color:'#b491d9'},{name:'Historical particle V3',y:s.historical_probes?.scores?.historical_particle_v3,color:'#d68fb6'}];let gameSeries=[],trainSeries=[],latest={};document.getElementById('rolling').replaceChildren();let plateau=[];for(const arm of Object.keys(names)){const history=arms[arm]?.history||[],last=history.at(-1);latest[arm]=last;const key=arm==='sampled_control'?'control':'neutral';document.getElementById(key+'Value').textContent=num(game(last));document.getElementById(key+'Detail').textContent=last?'At '+last.step.toLocaleString()+' editing updates · '+(finite(gold)?'difference from ordinary target '+signed(game(last)-gold):'historical target pending'):'Waiting for a probe';gameSeries.push({color:colors[arm],points:history.map(p=>({x:p.step,y:game(p)}))});for(const metric of ['g_game','d_game'])trainSeries.push({color:colors[arm],dash:metric==='d_game',points:history.map(p=>({x:p.step,y:p.rolling?.edit?.[metric]}))});const rolling=last?.rolling?.edit||status.arms?.[arm]?.rolling?.edit;if(rolling)document.getElementById('rolling').append(row([names[arm],String(last?.step??status.step??0),num(rolling.g_game),num(rolling.d_game),num(rolling.penalty),num(rolling.bank_grad_norm)]));const flag=last?.plateau_diagnostic?.edit;if(flag)plateau.push(names[arm]+': '+(flag.regressed?'fixed-probe game has regressed':flag.possible_plateau?'possible plateau':'fixed-probe game is changing')+' over '+flag.steps+' editing updates (relative improvement '+num(100*flag.relative_improvement,2)+'%).')}
chart('gameChart',gameSeries,referenceLevels,'Clean FAST G game · frozen D1856 · fixed held-out probe');chart('trainingChart',trainSeries,[],'Rolling G/D game · each run’s changing critic');document.getElementById('plateau').textContent=plateau.join(' ')||'Plateau diagnostic begins after four fixed probes. It does not stop or change training.';document.getElementById('panelIdentity').textContent=finite(gold)?'Historical levels were rescored on exactly the live probe contexts and paired panels. '+(s.historical_probes?.indices?.length||12)+' contexts; four panels; σ = 0.125.':'Historical levels appear only after their matching live-probe rescoring is available.';const ng=game(latest.sampled_hb_neutral),cg=game(latest.sampled_control);document.getElementById('outcome').textContent=finite(gold)&&finite(ng)?'Latest H/b-neutral probe '+num(ng)+(ng<gold?' is below ':' is above ')+'the ordinary target '+num(gold)+'. '+(finite(cg)?'Fresh sampled control: '+num(cg)+'. ':'')+'Full-set endpoint results are shown separately below.':'Awaiting matched held-out probes and the ordinary target.';
const body=document.getElementById('endpoints');body.replaceChildren();let complete=0;for(const step of [5120,6400])for(const judge of ['D1856','D6400']){const control=arms.sampled_control?.evaluations?.[step]?.test,neutral=arms.sampled_hb_neutral?.evaluations?.[step]?.test,target=s.historical_references?.ordinary_lora_6400?.test,cv=control?.[judge],nv=neutral?.[judge],tv=target?.[judge];if(finite(cv)&&finite(nv))complete++;body.append(row([String(step),judge,String(neutral?.count??control?.count??240),num(tv),num(cv),num(nv),signed(nv-cv),signed(nv-tv)]))}document.getElementById('endpointStatus').textContent=complete===4?'Both declared endpoints and both judges are available.':'Waiting for '+(4-complete)+' endpoint/judge comparisons. The 5,120 and 6,400 results are both required.';const receipt=s.receipt;document.getElementById('verification').textContent=receipt?'Receipt: '+receipt.status+' · runtime '+num(receipt.seconds,1)+' s · '+(receipt.qualification_credit||'Independent qualification pending'):'Runtime verification is still pending. '+(finite(status.seconds)?'Elapsed '+num(status.seconds,1)+' s.':'');document.getElementById('activity').textContent=Object.keys(names).map(arm=>{const c=receipt?.coverage?.[arm]||status.arms?.[arm]?.coverage;return c?names[arm]+': '+c.live_bank_updates+' live-bank updates, '+c.dense_128_row_updates+' dense 128-row updates, '+c.moves+' accepted moves.':''}).filter(Boolean).join(' ');document.getElementById('failure').hidden=!s.failure;if(s.failure)document.getElementById('failure').textContent=s.failure.error||JSON.stringify(s.failure,null,2);document.getElementById('updated').textContent='Last read: '+s.updated+' · refresh every 5 seconds.'}
async function refresh(){try{const response=await fetch('/api',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);render(await response.json())}catch(error){document.getElementById('updated').textContent='Refresh failed: '+error.message+' · retrying in 5 seconds.'}}refresh();setInterval(refresh,5000);
</script></html>'''


def make_handler(run):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                payload = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif path in ("/api", "/api/state"):
                payload = json.dumps(state(run), allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8784, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.run.resolve()))
    print(f"Serving read-only Supra initialization dashboard on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
