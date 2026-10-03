#!/usr/bin/env python3
"""Read-only, stdlib dashboard for the matched final-precision continuation.

Reads saved JSON/JSONL only. It never imports a model, ParticleGAN or Torch,
and never writes training inputs or changes training decisions.
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time

ARMS = ("BF16", "FP32_final")
TRACE_FIELDS = ("loss_g", "loss_d_game", "penalty", "output_sigma", "bank_grad_norm")


def read_json(path, notes):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        notes.append(f"{path.name}: temporarily unreadable ({type(exc).__name__})")
        return None


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def read_trace(path, notes):
    rows, previous = [], 12800
    try:
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue  # A concurrent producer may still be writing this row.
                try:
                    row = json.loads(line)
                    step = row.get("step")
                    if type(step) is not int or not 12800 < step <= 19200 or step <= previous:
                        notes.append(f"{path.parent.name}: unexpected trace clock; row omitted")
                        continue
                    rows.append(dict(step=step, penalty_phase=row.get("penalty_phase"),
                                     **{key: number(row.get(key)) for key in TRACE_FIELDS}))
                    previous = step
                except (ValueError, AttributeError):
                    notes.append(f"{path.parent.name}: malformed completed trace row omitted")
    except FileNotFoundError:
        pass
    except OSError as exc:
        notes.append(f"{path.parent.name}: trace unreadable ({type(exc).__name__})")
    return rows


def rolling_trace(rows):
    """Missing observations stay missing; each metric uses its own finite count."""
    windows = {key: deque() for key in TRACE_FIELDS}
    totals = dict.fromkeys(TRACE_FIELDS, 0.)
    counts = dict.fromkeys(TRACE_FIELDS, 0)
    result = []
    for row in rows:
        point = dict(step=row["step"])
        for key in TRACE_FIELDS:
            value = row[key]
            windows[key].append(value)
            if value is not None:
                totals[key] += value; counts[key] += 1
            if len(windows[key]) > 100:
                removed = windows[key].popleft()
                if removed is not None:
                    totals[key] -= removed; counts[key] -= 1
            point[key] = totals[key] / counts[key] if counts[key] else None
        result.append(point)
    return result


def snapshot(run_dir, initial_report):
    notes = []
    status = read_json(run_dir / "status.json", notes) or {}
    report = read_json(run_dir / "report.json", notes) or {}
    completion = read_json(run_dir / "completion.json", notes) or {}
    initial = read_json(initial_report, notes) if initial_report is not None else None
    initial = initial or {}
    summaries = initial.get("summaries", {})
    original = summaries.get("original28000_BF16") or status.get("fixed_original") or report.get("fixed_original") or {}
    traces = {arm: read_trace(run_dir / arm / "train.jsonl", notes) for arm in ARMS}
    endpoints = status.get("endpoints") or report.get("endpoint_results") or {}
    error = completion.get("error") or report.get("error") or status.get("error")
    phase = ("INCOMPLETE" if error else status.get("phase", "waiting for run"))
    if report.get("complete") is True:
        phase = f"complete · producer {report.get('scientific_status', 'pending')} · qualification pending"
    return dict(updated=time.time(), status=status, report=report, completion=completion,
                phase=phase, error=error, traces=traces,
                rolling={arm: rolling_trace(rows) for arm, rows in traces.items()},
                endpoints=endpoints, initial=summaries, fixed_original=original,
                notes=list(dict.fromkeys(notes)))


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra final precision continuation</title><style>
:root{color-scheme:dark;--bg:#10141c;--card:#19202c;--ink:#e5ecf5;--muted:#a5b2c5;--grid:#344052}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 6px}h2{font-size:19px;margin:0 0 8px}
p{margin:6px 0 14px}.muted{color:var(--muted)}.card{padding:18px;background:var(--card);border:1px solid #2b3546;border-radius:12px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.grid .card{margin:0}.wide{grid-column:1/-1}
.chart{position:relative;width:100%;min-height:270px}svg{display:block;width:100%;height:auto}svg text{font-size:12px;fill:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:9px 18px;font-size:13px}.legend span{white-space:nowrap}.swatch{display:inline-block;width:20px;border-top:3px solid;margin-right:6px;vertical-align:middle}
.tip{display:none;position:absolute;pointer-events:none;background:#05080fef;border:1px solid #62718b;border-radius:7px;padding:9px;font-size:12px;max-width:330px;z-index:2;white-space:pre-line}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--grid)}th{font-size:13px;color:var(--muted)}
.bad{color:#ffb8aa}.pill{display:inline-block;border:1px solid #718198;border-radius:20px;padding:3px 10px;margin-right:10px}
progress{width:100%;height:10px;accent-color:#58b4ff}label{cursor:pointer}.small{font-size:13px}.value{font-size:21px;font-variant-numeric:tabular-nums}
@media(max-width:850px){main{padding:14px}.grid{grid-template-columns:1fr}table{font-size:12px}th,td{padding:6px}}
</style></head><body><main>
<h1>Supra: matched final-head precision continuation</h1>
<p class="muted">Shared-Up particles · native game · editing only · fixed 12,800 → 19,200 updates per arm</p>
<div class="card"><span id="phase" class="pill">Waiting for saved status</span><span id="clock" class="muted"></span>
<p id="error" class="bad"></p><div id="progress"></div><p id="notes" class="muted small"></p></div>
<div class="grid" id="arms"></div>
<div class="card"><h2>Native training losses</h2><p class="muted small">Each point is a real native update. Solid lines average the last 100 updates; faint lines show individual updates. These learned game losses are distinct from held-out task scores; their scale can move as the critic changes. D excludes the KA2 penalty.</p>
<label class="small"><input id="raw" type="checkbox" checked> Show individual updates alongside the rolling mean</label>
<label class="small" style="margin-left:18px">Loss window <select id="loss-window"><option value="all">All observed updates</option><option value="1000">Most recent 1,000 updates</option></select></label>
<div class="grid"><div><h2>Generator game loss</h2><div id="loss_g" class="chart"></div></div><div><h2>Critic game loss</h2><div id="loss_d_game" class="chart"></div></div></div>
<h2>Native output sigma</h2><p class="muted small">Logged after each native update; an unchanged trace alone does not establish a gradient defect.</p><div id="output_sigma" class="chart"></div></div>
<div class="card"><h2>Fixed full TEST240 endpoints</h2>
<p>Solid curves compare <strong>BF16-trained and FP32-final-trained particles under the same FP32 final-head arithmetic</strong>. That difference measures the training effect. A single checkpoint's solid-versus-dashed scores measure its serving precision effect. Both arms start from the same trained 12,800-update particle checkpoint.</p>
<p class="muted small">All captures use the same 240 contexts, live target teacher, and fixed judge panels.</p>
<p class="muted small">16,000 is a mandatory diagnostic; 19,200 is the fixed terminal comparison. Displayed scores are producer observations pending independent qualification. Output accuracy never changes the optimizer, guards, horizon, or checkpoint selection.</p>
<div id="endpoint-legend" class="legend"></div>
<div id="rmse_diagnostic" class="chart"></div><div id="D1856" class="chart"></div><div id="D6400" class="chart"></div>
<p class="muted small" title="Selected historical ordinary LoRA: standard low-rank adapters without the particle bank/router; MSE/AdamW objective, 22,400 editing + 5,600 preservation updates. Its training budget and objective differ from this native-game editing-only continuation.">The horizontal target is the selected original ordinary LoRA at 28,000, using its original BF16 arithmetic. It is a fixed quality target with a different objective and mixed training schedule, not a matched-budget optimizer control. Ordinary FP32 values below are descriptive shared-arithmetic controls and do not replace that target.</p>
<div id="initial-table"></div><div id="endpoint-table"></div></div>
<p class="muted small">Read-only saved-artifact view · refreshes every 5 seconds · no model execution or training controls</p>
</main><script>
const COLORS={BF16:'#58b4ff',FP32_final:'#ff95cd'},METRICS=['rmse_diagnostic','D1856','D6400'];
const METRIC_NAMES={rmse_diagnostic:'Physical residual RMSE ↓',D1856:'Fixed judge D1856 paired game ↓',D6400:'Fixed judge D6400 paired game ↓'};
const END_SERIES=[
 {arm:'BF16',eval:'FP32_final',label:'BF16-trained → FP32 final (common arithmetic)',color:COLORS.BF16},
 {arm:'FP32_final',eval:'FP32_final',label:'FP32-final-trained → FP32 final (common arithmetic)',color:COLORS.FP32_final},
 {arm:'BF16',eval:'BF16',label:'BF16-trained → BF16 final',color:'#69d7c2',dash:'6 5'},
 {arm:'FP32_final',eval:'BF16',label:'FP32-final-trained → BF16 final',color:'#e7bb74',dash:'6 5'}];
let saved=null;
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=(x,n=6)=>finite(x)?x.toFixed(n):'pending';
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const duration=x=>finite(x)?`${Math.floor(x/3600)}h ${Math.floor(x%3600/60)}m ${Math.floor(x%60)}s`:'pending';
const name=a=>a==='BF16'?'BF16-trained particles':'FP32-final-trained particles';
function chart(id,series,{target=null,reference=null,title='',endpoints=false,xmin=12800,xmax=19200}={}){
 const el=document.getElementById(id),w=1100,h=310,L=72,R=24,T=31,B=43;
 const points=series.flatMap(s=>s.points.filter(p=>finite(p.y)));
 const ys=points.map(p=>p.y);if(finite(target))ys.push(target);if(finite(reference))ys.push(reference);
 if(!ys.length){el.innerHTML=`<p class="muted">${esc(title)}: pending saved observations.</p>`;return;}
 let lo=Math.min(...ys),hi=Math.max(...ys),pad=(hi-lo)*.12||Math.max(Math.abs(hi)*.08,.001);lo-=pad;hi+=pad;
 const x=v=>L+(v-xmin)/(xmax-xmin)*(w-L-R),y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><text x="${L}" y="17">${esc(title)}</text>`;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);svg+=`<line x1="${L}" x2="${w-R}" y1="${yy}" y2="${yy}" stroke="#344052"/><text x="${L-9}" y="${yy+4}" text-anchor="end">${v.toFixed(4)}</text>`;}
 const ticks=endpoints?[12800,14400,16000,17600,19200]:Array.from({length:5},(_,i)=>Math.round(xmin+(xmax-xmin)*i/4));
 for(const v of ticks)svg+=`<line x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${h-B}" stroke="#263246"/><text x="${x(v)}" y="${h-21}" text-anchor="middle">${v.toLocaleString()}</text>`;
 if(finite(reference))svg+=`<line x1="${L}" x2="${w-R}" y1="${y(reference)}" y2="${y(reference)}" stroke="#a5b2c5" stroke-dasharray="9 4 2 4"/>`;
 if(finite(target))svg+=`<line x1="${L}" x2="${w-R}" y1="${y(target)}" y2="${y(target)}" stroke="#f0edf1" stroke-dasharray="3 5"/><text x="${w-R}" y="${Math.max(T+11,y(target)-7)}" text-anchor="end">Original 28,000 BF16 target ${target.toFixed(6)}</text>`;
 for(const s of series){let path='',pen=false;for(const p of s.points){if(!finite(p.y)){pen=false;continue;}path+=`${pen?'L':'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`;pen=true;}
  svg+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="${s.raw?1:2.5}" opacity="${s.raw ? .22 : 1}"${s.dash?` stroke-dasharray="${s.dash}"`:''}/>`;
  if(endpoints)for(const p of s.points)if(finite(p.y))svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="4" fill="${s.color}"/>`;
 }
 if(endpoints)for(const step of [16000,19200]){const pending=series.filter(s=>!s.points.some(p=>p.x===step&&finite(p.y))).length;if(pending)svg+=`<text x="${x(step)}" y="${T+16}" text-anchor="middle">${pending}/4 observations pending</text>`;}
 svg+='<text x="'+((L+w-R)/2)+'" y="'+(h-3)+'" text-anchor="middle">Absolute editing update</text></svg>';
 el.innerHTML=svg+'<div class="tip"></div>';const node=el.querySelector('svg'),tip=el.querySelector('.tip');
 node.onmousemove=e=>{const rect=node.getBoundingClientRect(),v=xmin+(((e.clientX-rect.left)/rect.width*w-L)/(w-L-R))*(xmax-xmin);
  if(v<xmin||v>xmax){tip.style.display='none';return;}let lines=[];
  for(const s of series.filter(s=>!s.raw)){let p=null;for(const q of s.points)if(finite(q.y)&&(!p||Math.abs(q.x-v)<Math.abs(p.x-v)))p=q;if(p)lines.push(`${s.label}\n  ${p.x.toLocaleString()}: ${p.y.toFixed(7)}`);}
  if(finite(target))lines.push(`Fixed original 28,000 BF16: ${target.toFixed(7)}\nMSE/AdamW, mixed schedule; different budget`);
  if(finite(reference))lines.push(`Original 28,000 FP32 final: ${reference.toFixed(7)}\nDescriptive precision control; target unchanged`);
  tip.textContent=lines.join('\n');tip.style.display='block';tip.style.left=Math.max(0,Math.min(e.clientX-rect.left+12,rect.width-320))+'px';tip.style.top='35px';};
 node.onmouseleave=()=>tip.style.display='none';
}
function endpointSeries(data,key){return END_SERIES.map(s=>({...s,points:[12800,16000,19200].map(step=>({x:step,y:step===12800?data.initial?.[`neutral12800_${s.eval}`]?.[key]:data.endpoints?.[`${s.arm}@${step}:${s.eval}`]?.[key]}))}));}
function paint(data){saved=data;const st=data.status||{};document.getElementById('phase').textContent=data.phase;
 document.getElementById('clock').textContent=`Elapsed ${duration(st.seconds??data.report.seconds)} · saved data refreshed ${new Date(data.updated*1000).toLocaleTimeString()}`;
 document.getElementById('error').textContent=data.error?`${data.error.type??'Error'}: ${data.error.message??data.error}`:'';
 document.getElementById('notes').textContent=data.notes.join(' · ');
 let progress='',cards='';for(const arm of ['BF16','FP32_final']){const rows=data.traces[arm],last=rows.at(-1),roll=data.rolling[arm].at(-1),info=st.arms?.[arm]||{},step=last?.step??12800,done=step-12800;
  progress+=`<div class="small">${name(arm)}: ${step.toLocaleString()} / 19,200 · ${done.toLocaleString()} / 6,400 additional editing updates</div><progress max="6400" value="${done}"></progress>`;
  cards+=`<div class="card"><h2 style="color:${COLORS[arm]}">${name(arm)}</h2><div class="value">${step.toLocaleString()} updates</div><p class="muted small">Charged runtime ${duration(info.charged_seconds)} · preservation updates 0</p><table><tr><th>Last 100 native updates (${Math.min(rows.length,100)})</th><th>Mean</th></tr>${[['loss_g','G game'],['loss_d_game','D game'],['penalty','KA2 penalty'],['output_sigma','Output sigma']].map(([k,l])=>`<tr><td>${l}</td><td>${fmt(roll?.[k])}</td></tr>`).join('')}<tr><td>Latest penalty phase</td><td>${esc(last?.penalty_phase??'pending')}</td></tr><tr><td>Native bank-live updates</td><td>${info.coverage?.bank_live_updates??'pending'}</td></tr><tr><td>Accepted moves</td><td>${info.coverage?.accepted_moves??'pending'}</td></tr></table></div>`;
 }
 document.getElementById('progress').innerHTML=progress;document.getElementById('arms').innerHTML=cards;
 const observed=Math.max(12801,...Object.values(data.traces).map(rows=>rows.at(-1)?.step??12800));
 const xmin=document.getElementById('loss-window').value==='1000'?Math.max(12800,observed-1000):12800;
 for(const key of ['loss_g','loss_d_game','output_sigma']){const series=[];for(const arm of ['BF16','FP32_final']){
  if(document.getElementById('raw').checked)series.push({label:name(arm)+' individual',color:COLORS[arm],raw:true,points:data.traces[arm].filter(r=>r.step>=xmin).map(r=>({x:r.step,y:r[key]}))});
  series.push({label:name(arm)+' rolling 100',color:COLORS[arm],points:data.rolling[arm].filter(r=>r.step>=xmin).map(r=>({x:r.step,y:r[key]}))});}
  chart(key,series,{title:key==='loss_g'?'Native generator loss':key==='loss_d_game'?'Native critic game loss':'Native output sigma',xmin,xmax:observed});
 }
 document.getElementById('endpoint-legend').innerHTML=END_SERIES.map(s=>`<span><i class="swatch" style="border-color:${s.color};${s.dash?'border-top-style:dashed':''}"></i>${esc(s.label)}</span>`).join('')+'<span><i class="swatch" style="border-color:#f0edf1;border-top-style:dotted"></i>Fixed original 28,000 BF16 target</span><span><i class="swatch" style="border-color:#a5b2c5;border-top-style:dashed"></i>Original 28,000 FP32 final (descriptive)</span>';
 for(const key of METRICS)chart(key,endpointSeries(data,key),{title:METRIC_NAMES[key],target:data.fixed_original?.[key],reference:data.initial?.original28000_FP32_final?.[key],endpoints:true});
 const rows=[['Initial particles, BF16','neutral12800_BF16'],['Initial particles, FP32 final','neutral12800_FP32_final'],['Fixed original 28,000, BF16','original28000_BF16'],['Original 28,000, FP32 final (descriptive)','original28000_FP32_final']];
 document.getElementById('initial-table').innerHTML='<h2>Qualified initial/reference captures</h2><table><tr><th>Capture</th><th>RMSE</th><th>D1856</th><th>D6400</th></tr>'+rows.map(([label,k])=>`<tr><td>${label}</td>${METRICS.map(m=>`<td>${fmt(data.initial?.[k]?.[m])}</td>`).join('')}</tr>`).join('')+'</table>';
 let table='<h2>Common FP32 endpoint comparison</h2><table><tr><th>Step</th><th>Training arm → evaluation</th><th>RMSE</th><th>D1856</th><th>D6400</th></tr>';
 for(const step of [16000,19200])for(const arm of ['BF16','FP32_final']){const value=data.endpoints?.[`${arm}@${step}:FP32_final`];table+=`<tr><td>${step.toLocaleString()}</td><td>${name(arm)} → FP32 final</td>${METRICS.map(m=>`<td>${fmt(value?.[m])}</td>`).join('')}</tr>`;}
 document.getElementById('endpoint-table').innerHTML=table+'</table>';
}
async function poll(){try{const r=await fetch('/data',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);paint(await r.json());}catch(e){document.getElementById('notes').textContent=`Dashboard read failed: ${e.message}. Showing the last saved view.`;}}
document.getElementById('raw').onchange=()=>{if(saved)paint(saved);};poll();setInterval(poll,5000);
document.getElementById('loss-window').onchange=()=>{if(saved)paint(saved);};
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-score-report", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8786)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/data":
                body = json.dumps(snapshot(args.run_dir, args.initial_score_report),
                                  allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def log_message(self, format, *values):
            # Five-second artifact polling does not need to fill the server log.
            if not values or str(values[1] if len(values) > 1 else "").startswith("4"):
                super().log_message(format, *values)

    print(f"Read-only precision dashboard: http://pop-os:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
