#!/usr/bin/env python3
"""Read-only dashboard for fresh and continued particle-LoRA gating runs."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

try:
    from .serve_e22_supra_bank_game_trust import read_json, read_history, read_metrics
except ImportError:
    from serve_e22_supra_bank_game_trust import read_json, read_history, read_metrics


ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supra · original ordinary LoRA vs particle LoRA</title>
<style>
:root{font:16px system-ui,sans-serif;color:#eaf0f8;background:#101723}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0 0 9px}.muted,footer{color:#9babbe;font-size:13px;line-height:1.6}header{display:flex;justify-content:space-between;gap:20px;align-items:center}.pill{border:1px solid #3b516d;border-radius:24px;padding:8px 14px;white-space:nowrap}.cards,.charts{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:22px 0}.charts{grid-template-columns:1fr;margin-bottom:0}.card,.panel{background:#162131;border:1px solid #2b3b51;border-radius:14px;padding:20px}.panel{margin:18px 0}.label{font-size:13px;color:#9babbe}.value{font-size:28px;margin:8px 0}.delta{color:#a5ceee;font-size:13px;line-height:1.5}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend span:before{content:'';display:inline-block;width:22px;height:3px;background:var(--color);vertical-align:middle;margin-right:6px}label{font-size:13px;margin-right:16px}.chart svg{display:block;width:100%}table{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}th,td{padding:11px 7px;border-bottom:1px solid #2b3b51}th{font-size:12px;color:#9babbe}th:first-child,td:first-child{text-align:left}a{color:#9dd4ff}footer{margin-top:15px}.overflow{overflow-x:auto}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#c2d2e4;line-height:1.5;max-height:240px;overflow:auto}@media(max-width:700px){body{padding:16px}.cards,.charts{grid-template-columns:1fr}header{display:block}.pill{display:inline-block;margin-top:13px}th,td{font-size:12px;padding:8px 3px}}
</style>
<header><div><h1>Original ordinary LoRA vs particle LoRA</h1><div id="runDescription" class="muted">Particle LoRA V3 · fixed-budget experiment · native shared ParticleGAN profile</div></div><div id="status" class="pill">Preparing</div></header>
<div class="panel"><h2>Ordinary LoRA remains the overall reference to beat</h2><div id="completedOutcome" style="font-weight:600;line-height:1.6;margin-bottom:8px"></div><div id="benchmarkContext" class="muted">Editing is the primary comparison; preservation is reported as a secondary tradeoff. Original ordinary LoRA at 6,400 updates remains the editing target.</div><div id="archiveStatus" class="muted" style="margin-top:8px"></div><div id="references" class="muted" style="margin-top:8px">Awaiting compatible ordinary LoRA probes under the common step-1,856 critic.</div></div>
<div class="cards"><div class="card"><div class="label">Particle LoRA V3 · total completed updates</div><div id="updates" class="value">— / 400</div><div id="probe" class="muted">Waiting for the first probe</div><div id="continuation" class="muted"></div><div id="trainingBudget" class="muted"></div></div><div class="card"><div class="label">Editing · primary · clean G game</div><div id="edit" class="value">—</div><div id="editDelta" class="delta">Waiting for matched V2 probes</div><div id="editOrdinaryDelta" class="delta"></div></div><details class="card" style="grid-column:1/-1"><summary class="label">Preservation · secondary metrics · expand to inspect</summary><div id="preservation" class="value">—</div><div id="preservationDelta" class="delta">Waiting for matched V2 probes</div><div id="preservationOrdinaryDelta" class="delta"></div></details></div>
<div class="panel"><h2 id="graphTitle">Matched-budget ordinary LoRA and particle LoRA progress</h2><div id="comparisonProtocol" class="muted">Common frozen V2 step-1,856 critic, matching probe contexts and private CPU72 paired panels. Solid: clean serving. Dashed: private DV12 probes. Lower G game loss is better.</div><div id="controlProtocol" class="muted"></div><div id="scheduleProtocol" class="muted"></div><div class="legend"><span style="--color:#f2c474">Original ordinary LoRA · clean</span><span style="--color:#94d5b1" id="controlLegend">Particle LoRA · V2 · control</span><span style="--color:#62b6f0">Particle LoRA · V3 · gated particles</span></div><label>View <select style="background:#162131;color:#eaf0f8;padding:6px;border:1px solid #3b516d;border-radius:6px" id="view" onchange="render(current)"><option id="matchedView" value="matched">Matched training budget</option><option value="overall">Overall target · through 6,400</option></select></label><label>Horizontal axis <select style="background:#162131;color:#eaf0f8;padding:6px;border:1px solid #3b516d;border-radius:6px" id="axis" onchange="render(current)"><option value="total">Total training updates</option><option value="editing">Editing updates</option></select></label><label><input id="clean" type="checkbox" checked onchange="render(current)">Clean</label><label><input id="dv12" type="checkbox" checked onchange="render(current)">DV12</label><div class="charts"><div id="editChart" class="chart"></div><details><summary class="muted">Secondary preservation curve · expand to inspect</summary><div id="preservationChart" class="chart"></div></details></div><div id="historyStatus" class="muted"></div></div>
<div class="panel"><h2>Overall target · original ordinary LoRA trained for 6,400 updates</h2><div id="overallTarget" class="muted">Awaiting compatible historical target probes.</div><div class="muted">The target uses the original 6,400-update ordinary-LoRA checkpoint. The V3 curve stops at its last recorded probe; longer-budget targets are separate from shorter matched comparisons. <a href="http://pop-os:8765">See the qualified 6,400-update overall benchmark</a>.</div></div>
<div class="panel"><h2>Rolling V3 training game · changing critic</h2><div class="muted">Last 100 updates of each task. G/D games exclude the preservation task weight. Penalty and bank gradient retain native units.</div><table><thead><tr><th>Task</th><th>Updates</th><th>G game</th><th>D game</th><th>Penalty</th><th>Bank gradient</th></tr></thead><tbody id="rolling"></tbody></table><div id="plateau" class="muted" style="margin-top:12px">Waiting for four fixed probes to report the game plateau diagnostic.</div></div>
<div class="panel"><h2>Particle connectivity and native controls</h2><div id="audit" class="muted">Awaiting the read-only 71-site gated-basis audit.</div><div id="nativeEvents" class="muted"></div><pre id="activity">Native controller activity pending</pre><div class="muted">FP32 branch effects and gradients show connectivity. Some small per-site changes can quantize away at the frozen BF16 host boundary. Full-model ablations below assess game dependence separately. Output error is not an optimizer or structural criterion.</div></div>
<div class="panel" id="bothEndpoints" hidden><h2>Both predeclared endpoints · held-out editing</h2><div class="muted">All 240 held-out editing contexts under both common judges; lower G game is better. The 5,440-total endpoint matches the original 5,120-edit budget. The 6,400-total endpoint uses 6,080 edits. Both fixed results are reported.</div><div class="overflow"><table><thead><tr><th>Total / editing updates</th><th>Common judge</th><th>Original ordinary LoRA</th><th>Particle V2</th><th>Particle V3</th><th>V3 − V2</th><th>V3 − ordinary</th><th>Artifact review</th></tr></thead><tbody id="endpointSummary"></tbody></table></div></div><div class="panel"><h2 id="finalTitle">Final matched-budget evaluation</h2><label id="endpointControl" hidden>Predeclared endpoint <select id="endpoint" onchange="render(current)"><option value="5440">5,440 total · 5,120 editing updates</option><option value="6400">6,400 total · 6,080 editing updates</option></select></label><div id="finalStatus" class="muted">Awaiting the fixed horizon, final evaluation and runtime receipt.</div><div class="overflow"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Common judge</th><th id="finalOrdinaryLabel">Original ordinary LoRA</th><th id="finalControlLabel">Particle V2</th><th id="finalCandidateLabel">Particle V3</th><th>V3 − V2</th><th>V3 − ordinary</th></tr></thead><tbody id="finalMetrics"></tbody></table></div><div class="muted" style="margin-top:16px">Output RMSE · final diagnostic only</div><table><thead><tr><th>Pool</th><th>V2</th><th>V3</th><th>V3 − V2</th></tr></thead><tbody id="finalOutput"></tbody></table></div>
<div class="panel"><h2 id="contributionTitle">Final V3 particle contribution · paired game ablations</h2><div class="muted">All 570 contexts, with common critics and paired noise within each ablation. Positive ablation − clean means the intact particle computation has lower game loss for this fixture. These CPU-generated panels differ from the endpoint evaluator's CUDA panels; absolute values belong to this table.</div><div class="overflow"><table><thead><tr><th>Pool</th><th>Contexts</th><th>Common judge</th><th>Zero codes − clean</th><th>Mass-only routing − clean</th></tr></thead><tbody id="contribution"></tbody></table></div><div class="muted">Ablations are observational; they do not select, stop, or change training. Native source and random-stream checks are reported in the final receipt.</div></div>
<footer id="updated">Waiting for first refresh</footer><footer>One task and stream; the gate has not established general superiority. <a href="http://pop-os:8765">Overall LoRA vs best particle</a> · <a href="http://pop-os:8766">PR223 vs old V2 at 1,600</a> · <a href="http://pop-os:8767">Separate step-6,400 bank-trust continuation</a> · <a href="http://pop-os:8782">Qualified 1,600-update particle V3 comparison</a></footer>
<script>
let current=null,axisInitialized=false,endpointCompletionApplied=false;const num=(x,d=6)=>Number.isFinite(x)?x.toFixed(d):'—',signed=x=>Number.isFinite(x)?(x>0?'+':'')+num(x):'—',ns='http://www.w3.org/2000/svg';
function node(tag,attrs={},text=''){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function score(p,task,mode='clean'){return p?.probes?.[task]?.[mode]?.frozen_start_D}
function exposure(total,arm,s){return arm==='v3'&&s.training_budget?.editing_only&&total>1600?1280+total-1600:total-Math.floor(total/5)}
function row(values){const tr=document.createElement('tr');if(values[0]==='preservation'||values[0]==='holds')tr.style.opacity='.7';for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}return tr}
function plot(id,title,series,max,axis='total'){const host=document.getElementById(id),svg=node('svg',{viewBox:'0 0 530 300',role:'img','aria-label':title+' common step-1856 critic generator game loss'}),points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max);svg.append(node('text',{x:68,y:22,fill:'#eaf0f8','font-size':16},title));if(!points.length){svg.append(node('text',{x:68,y:135,fill:'#9babbe'},'Awaiting recorded probes'));host.replaceChildren(svg);return}let lo=Math.min(...points.map(p=>p.y)),hi=Math.max(...points.map(p=>p.y)),pad=Math.max(.001,(hi-lo)*.12);lo=Math.max(0,lo-pad);hi+=pad;const X=x=>68+x/max*444,Y=y=>242-(y-lo)/(hi-lo)*190;for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4,x=max*i/4;svg.append(node('line',{x1:68,x2:512,y1:Y(y),y2:Y(y),stroke:'#2b3b51'}),node('text',{x:60,y:Y(y)+4,fill:'#9babbe','text-anchor':'end','font-size':11},num(y,3)),node('text',{x:X(x),y:263,fill:'#9babbe','text-anchor':'middle','font-size':11},Math.round(x).toLocaleString()))}svg.append(node('text',{x:68,y:39,fill:'#9babbe','font-size':11},'G game loss · frozen V2 critic at 1,856'),node('text',{x:290,y:287,fill:'#9babbe','text-anchor':'middle','font-size':12},axis==='editing'?'Completed editing updates':'Total completed training updates'));for(const s of series){const p=s.points.filter(p=>Number.isFinite(p.y)&&p.x>=0&&p.x<=max).sort((a,b)=>a.x-b.x);if(!p.length)continue;svg.append(node('path',{d:p.map((v,i)=>(i?'L':'M')+X(v.x)+','+Y(v.y)).join(' '),fill:'none',stroke:s.color,'stroke-width':s.color==='#f2c474'?3:2,'stroke-dasharray':s.dashed?'6 4':'none'}));for(const v of p){const dot=node('circle',{cx:X(v.x),cy:Y(v.y),r:s.radius??3,fill:s.color});dot.append(node('title',{},s.label+' · '+(v.total??v.x).toLocaleString()+' total updates'+(v.editing!=null?' · '+v.editing.toLocaleString()+' editing updates':'')+' · '+(v.origin||'recorded checkpoint')+' · frozen-1856 G '+num(v.y)));svg.append(dot)}}host.replaceChildren(svg)}
function render(s){if(!s)return;const st=s.status||{},p=s.latest||s.history.at(-1),step=st.step??p?.step??0,horizon=s.horizon,budget=s.training_budget||{},editingOnly=budget.editing_only;if(!axisInitialized){document.getElementById('axis').value=editingOnly?'editing':'total';axisInitialized=true}const axis=document.getElementById('axis').value,totalMax=document.getElementById('view').value==='overall'?6400:horizon,max=axis==='editing'?Math.max(exposure(totalMax,'v3',s),exposure(totalMax,'ordinary',s)):totalMax;document.getElementById('status').textContent=st.phase==='preflight_failed'?'Preflight failed':st.phase==='failed'?'Validation failed':st.phase==='complete'?'Complete':st.phase==='evaluating_fixed_endpoint'?'Evaluating fixed endpoint':st.phase==='evaluating'||step>=horizon?'Evaluating':st.phase==='preparing_control'?'Qualifying V2 control':st.phase==='training'?'Training particle V3':'Preparing';document.getElementById('updates').textContent=st.phase==='preflight_failed'?'Not started':step.toLocaleString()+' / '+horizon.toLocaleString();document.getElementById('runDescription').textContent=s.continued?'Particle LoRA V3: qualified '+s.start_step.toLocaleString()+' → fixed '+horizon.toLocaleString()+' total updates · '+(editingOnly?'editing-only schedule after '+s.start_step:'same native game/configuration'):'Particle LoRA V3: fresh initialization → fixed '+horizon.toLocaleString()+' updates';document.getElementById('probe').textContent=p?'Latest fixed probe: '+p.step.toLocaleString()+' total updates':'Waiting for the first V3 probe';document.getElementById('continuation').textContent=s.continued?Math.max(0,step-s.start_step).toLocaleString()+' updates since the qualified '+s.start_step+' boundary':'';document.getElementById('references').textContent=s.gold_compatible?'Original ordinary LoRA uses matching D1856/data/context/panel provenance. Card differences use only actual checkpoints at '+(editingOnly?'equal editing exposure.':'the same total step.'):'Awaiting compatible original ordinary LoRA probes under the common step-1,856 critic.';document.getElementById('matchedView').textContent=editingOnly?'Declared fixed horizon':'Matched training budget';document.getElementById('graphTitle').textContent=editingOnly?'Editing-only continuation vs historical ordinary LoRA and particle LoRA':totalMax===6400?(horizon===6400?'6,400-total-update target · ordinary LoRA vs particle LoRA':'Overall 6,400-update ordinary-LoRA target vs shorter particle runs'):'Matched-budget ordinary LoRA vs particle LoRA · through '+horizon.toLocaleString();document.getElementById('trainingBudget').textContent=editingOnly?budget.editing_updates.toLocaleString()+' editing + '+budget.preservation_updates.toLocaleString()+' inherited preservation updates'+(budget.recorded_counts_match?'':' · recorded schedule counters differ'):'';document.getElementById('scheduleProtocol').textContent=editingOnly?'Schedule changed after the qualified 1,600-update boundary (1,280 editing + 320 preservation): all new updates are editing at task weight 1. V3 at 5,440 total has 5,120 editing updates, matching the original 6,400-total-update target. V3 at 6,400 total has 6,080 editing updates. Historical references retain their 4-edit/1-preservation schedule; this comparison changes training exposure as well as architecture.':'';document.getElementById('historyStatus').textContent=s.continued?(s.inherited_history_qualified?'V3 probes through '+s.start_step+' are inherited from its independently qualified initial run; later probes are continuation measurements. ':'Inherited probes withheld until their provenance is qualified. ')+(s.control_kind==='historical'?'V2 is the historical qualified clean trajectory. Its DV12 stream differs and is withheld. ':'V2 is its archived qualified shared-profile trajectory. ')+'No future V3 points are inferred.':'V3 is its fresh run; V2 is the qualified shared-profile trajectory. No future V3 points are inferred.';
document.getElementById('benchmarkContext').textContent=editingOnly?'Editing is primary. The independently qualified V3 1,600-update starting model beat original ordinary LoRA at the same budget. This new editing-only schedule has predeclared 5,120-edit and 6,080-edit endpoints; '+(s.final.independent_qualified&&s.final.runner_qualified?'both endpoint evaluations are recorded and independently qualified below.':'its endpoint results are pending qualification.'):horizon===1600&&s.final.independent_qualified?'At matched 1,600 updates, independently qualified V3 improves editing versus original ordinary LoRA under both common judges. Preservation is the secondary tradeoff. The original 6,400-update model remains the editing target.':'Editing is primary; preservation remains visible as a secondary tradeoff. The original ordinary-LoRA model trained for 6,400 updates is the editing target. Full endpoint qualification for this horizon is separate from small training probes.';document.getElementById('comparisonProtocol').textContent='Common frozen V2 step-1,856 critic · same probe contexts/private CPU72 paired panels. Solid: clean serving. Dashed: V3 private DV12 probes. Lower G game loss is better.';document.getElementById('controlLegend').textContent=s.control_kind==='historical'?'Particle LoRA · V2 · historical legacy · clean only':'Particle LoRA · V2 · shared profile';document.getElementById('controlProtocol').textContent=s.control_kind==='historical'?'Historical V2 control: '+(s.control_commit||'pending')+' · '+s.control_profile+'. V3 uses '+(s.plan.particlegan_commit||'pending')+' / '+(s.plan.profile||'pending')+'. Clean scores share D1856/panels; historical DV12 is withheld because its private stream differs.':'V2 and V3 use the same shared profile and common step-2 DV12 replay stream.';if(st.phase==='preflight_failed'){document.getElementById('runDescription').textContent='Planned '+horizon.toLocaleString()+'-update continuation · preflight failed before training';document.getElementById('continuation').textContent='Failed checks: '+s.failed_checks.join(', ')+' · repair pending';document.getElementById('probe').textContent='No new training probes recorded';}const goldPoints=s.gold.flatMap(g=>g.points),target=goldPoints.find(r=>r.step===6400);document.getElementById('overallTarget').textContent=target?'Common D1856 clean probes at 6,400: edit '+num(score(target,'edit'))+' · preservation '+num(score(target,'preservation'))+'. Original target: 5,120 editing updates. V3 has '+step.toLocaleString()+' total updates'+(editingOnly?' / '+budget.editing_updates.toLocaleString()+' editing updates; its equal-edit-budget endpoint is 5,440 total.':'; the matched endpoint has '+horizon.toLocaleString()+'.'):'Awaiting compatible historical target probes.';document.getElementById('archiveStatus').replaceChildren();if(s.archived400.qualified){const text=document.createElement('span');text.textContent=s.archived400.all_pooled_games_better?'Qualified 400-update result: V3 has lower pooled G game than V2 in all four pools under both common judges. This does not establish an ordinary-LoRA win. ':'Qualified initial 400-update results are archived separately. ';const link=document.createElement('a');link.setAttribute('href',s.archived400.url);link.textContent='View qualified 400-update results';document.getElementById('archiveStatus').append(text,link)}
for(const task of ['edit','preservation']){document.getElementById(task).textContent=num(score(p,task));for(const [id,points,label]of [[task+'Delta',s.control,'Particle LoRA V2'],[task+'OrdinaryDelta',goldPoints,'Original ordinary LoRA']]){const matchKey=(r,arm)=>editingOnly?exposure(r.step,arm,s):r.step,matched=[...s.history].reverse().find(r=>points.some(old=>matchKey(old,'reference')===matchKey(r,'v3'))),old=matched&&points.find(r=>matchKey(r,'reference')===matchKey(matched,'v3')),delta=score(matched,task)-score(old,task);document.getElementById(id).textContent=Number.isFinite(delta)?(editingOnly?'Equal editing budget '+exposure(matched.step,'v3',s).toLocaleString()+' (V3 '+matched.step.toLocaleString()+' total; reference '+old.step.toLocaleString()+' total)': 'Matched '+matched.step.toLocaleString())+': V3 − '+label+' '+signed(delta)+' · small training probes':'Awaiting an actual '+(editingOnly?'equal-edit-budget ':'same-step ')+label+' checkpoint'}const series=[];for(const [points,label,color]of [[s.control,s.control_kind==='historical'?'Particle LoRA · V2 · historical legacy':'Particle LoRA · V2 · shared profile','#94d5b1'],[s.history,'Particle LoRA · V3 · gated particles','#62b6f0']])for(const mode of ['clean','dv12'])if(document.getElementById(mode).checked&&!(points===s.control&&mode==='dv12'&&!s.control_dv12_compatible))series.push({label:label+' · '+mode,color,dashed:mode==='dv12',points:points.map(r=>({x:axis==='editing'?exposure(r.step,points===s.history?'v3':'reference',s):r.step,total:r.step,editing:exposure(r.step,points===s.history?'v3':'reference',s),y:score(r,task,mode),origin:r.probe_origin||'qualified archived control'}))});if(s.gold_compatible&&document.getElementById('clean').checked)for(const gold of s.gold)series.push({label:'Original ordinary LoRA · clean',color:'#f2c474',radius:4,points:gold.points.map(r=>({x:axis==='editing'?exposure(r.step,'ordinary',s):r.step,total:r.step,editing:exposure(r.step,'ordinary',s),y:score(r,task),origin:r.step===6400?'overall target: 6,400 total / 5,120 editing updates':'recorded original reference'}))});plot(task+'Chart',task==='edit'?'Editing · primary':'Preservation · secondary',series,max,axis)}
const rolling=st.rolling??p?.rolling??{};document.getElementById('rolling').replaceChildren(...['edit','preservation'].filter(task=>rolling[task]).map(task=>{const r=rolling[task];return row([task,r.updates,num(r.g_game),num(r.d_game),num(r.penalty),Number.isFinite(r.bank_grad_norm)?r.bank_grad_norm.toExponential(3):'—'])}));const window=s.history.slice(-4);document.getElementById('plateau').textContent=window.length===4?'Last four plotted frozen-critic clean probes: '+['edit','preservation'].map(task=>{const first=score(window[0],task),last=score(window.at(-1),task),change=(first-last)/Math.max(Math.abs(first),1e-12);return task+' '+num(100*change,3)+'% relative improvement over '+(window.at(-1).step-window[0].step)+' updates'+(Math.abs(change)<.002?' · possible plateau':change<-.002?' · regressed':'')}).join('; ')+'. Read-only diagnostic; fixed training horizon.':'Waiting for four fixed probes to report the game plateau diagnostic.';document.getElementById('activity').textContent=st.guard_activity?JSON.stringify(st.guard_activity,null,2):'Native controller activity pending';const audit=s.audit;document.getElementById('audit').textContent=audit?'Last audit at '+audit.step+' total updates: '+audit.branch_count+' gated-basis sites · clean/DV12 bank rows '+audit.clean_bank_rows+'/'+audit.dv12_bank_rows+' of 128 · '+audit.site_effects_nonzero+' nonzero final-output site interventions. State/RNG/gradients unchanged: '+audit.immutable:'Awaiting the read-only 71-site gated-basis audit.';
document.getElementById('endpointControl').hidden=!editingOnly;if(editingOnly&&st.phase==='complete'&&!endpointCompletionApplied){document.getElementById('endpoint').value='6400';endpointCompletionApplied=true}const endpointStep=editingOnly?Number(document.getElementById('endpoint').value):horizon,endpoint=s.final.endpoints?.[String(endpointStep)],gated=editingOnly?endpoint?.gated:s.final.gated,endpointQualified=editingOnly?endpoint?.independent_qualified:s.final.independent_qualified;
document.getElementById('finalOrdinaryLabel').textContent=editingOnly?'Original ordinary LoRA · 6,400 total / 5,120 editing':'Original ordinary LoRA';document.getElementById('finalControlLabel').textContent=editingOnly?'Particle V2 · 6,400 total / 5,120 editing':'Particle V2';document.getElementById('finalCandidateLabel').textContent=editingOnly?'Particle V3 · '+endpointStep.toLocaleString()+' total / '+exposure(endpointStep,'v3',s).toLocaleString()+' editing':'Particle V3';document.getElementById('finalTitle').textContent=editingOnly?'Predeclared '+endpointStep.toLocaleString()+'-total / '+exposure(endpointStep,'v3',s).toLocaleString()+'-editing endpoint · all 570 contexts':'Final matched '+horizon.toLocaleString()+'-update evaluation · all 570 contexts';document.getElementById('contributionTitle').textContent='Final V3 at '+horizon.toLocaleString()+' · paired particle game ablations';document.getElementById('finalStatus').textContent=endpointQualified&&s.final.runner_qualified?'Fixed-budget endpoint and independent artifact review qualified. All models use the same two common judges.':gated?s.final.runner_qualified?'Fixed-budget endpoint recorded; source/replay/stream checks passed. This endpoint’s independent artifact qualification is pending.':'Endpoint evaluations recorded; runtime receipt and independent qualification pending.':'Full 570-context endpoint at '+endpointStep.toLocaleString()+' and qualification pending; progress probes are not full held-out results.';const gameRows=[],outputRows=[],original=s.final.originals[horizon===6400?'converged_6400':'converged_1600'];for(const pool of ['fit','test','holds','preservation']){const a=s.final.control?.[pool],b=gated?.[pool],o=(horizon===1600||horizon===6400)?original?.[pool]:null;if(!a&&!b&&!o)continue;for(const judge of ['fixed_start_D','arm_final_D'])gameRows.push(row([pool,a?.count??b?.count??o?.count,judge==='fixed_start_D'?'Common frozen V2 D1,856':'Common control V2 D'+horizon.toLocaleString(),num(o?.[judge]?.g_game),num(a?.[judge]?.g_game),num(b?.[judge]?.g_game),signed(b?.[judge]?.g_game-a?.[judge]?.g_game),signed(b?.[judge]?.g_game-o?.[judge]?.g_game)]));outputRows.push(row([pool,num(a?.rmse),num(b?.rmse),signed(b?.rmse-a?.rmse)]))}document.getElementById('finalMetrics').replaceChildren(...gameRows);document.getElementById('finalOutput').replaceChildren(...outputRows);document.getElementById('contribution').replaceChildren(...['fit','test','holds','preservation'].filter(pool=>s.contribution?.[pool]).flatMap(pool=>['common_D1856',s.contribution_final_judge].map(judge=>{const r=s.contribution[pool];return row([pool,r.contexts,judge==='common_D1856'?'Common frozen V2 D1,856':'Common control V2 D'+horizon.toLocaleString(),signed(r.per_arm?.zero_particle_codes?.[judge]?.g_game_delta_from_clean),signed(r.per_arm?.mass_only_routing?.[judge]?.g_game_delta_from_clean)])})));document.getElementById('bothEndpoints').hidden=!editingOnly;const endpoints=s.final.endpoints||{},early=endpoints['5440'],late=endpoints['6400'],original6400=s.final.originals.converged_6400,judges=['fixed_start_D','arm_final_D'];document.getElementById('endpointSummary').replaceChildren(...(editingOnly?['5440','6400'].flatMap(total=>judges.map(judge=>{const a=s.final.control?.test,b=endpoints[total]?.gated?.test,o=original6400?.test;return row([Number(total).toLocaleString()+' / '+exposure(Number(total),'v3',s).toLocaleString(),judge==='fixed_start_D'?'Common frozen V2 D1,856':'Common control V2 D6,400',num(o?.[judge]?.g_game),num(a?.[judge]?.g_game),num(b?.[judge]?.g_game),signed(b?.[judge]?.g_game-a?.[judge]?.g_game),signed(b?.[judge]?.g_game-o?.[judge]?.g_game),endpoints[total]?.independent_qualified?'Qualified':'Pending'])})):[]));const bothQualified=editingOnly&&s.final.runner_qualified&&s.final.independent_qualified&&early?.independent_qualified&&late?.independent_qualified,earlyBeatsV2=bothQualified&&judges.every(j=>early.gated.test[j].g_game<s.final.control.test[j].g_game),earlyTrailsOrdinary=bothQualified&&judges.every(j=>early.gated.test[j].g_game>original6400.test[j].g_game),laterRegresses=bothQualified&&judges.every(j=>late.gated.test[j].g_game>early.gated.test[j].g_game),lateTrailsBoth=bothQualified&&judges.every(j=>late.gated.test[j].g_game>s.final.control.test[j].g_game&&late.gated.test[j].g_game>original6400.test[j].g_game);document.getElementById('completedOutcome').textContent=bothQualified?(earlyBeatsV2&&earlyTrailsOrdinary&&laterRegresses&&lateTrailsBoth?'Qualified result: at equal 5,120 editing updates, V3 beats historical particle V2 but trails original ordinary LoRA. At the final 6,080 edits, V3 regresses and trails both. Both common judges agree.':'Both predeclared endpoints are independently qualified; their held-out editing results are reported together below.') :'';const events=s.verified_native_events?.surprise_log||[],eventSteps=events.filter(e=>Array.isArray(e)&&Number.isInteger(e[0])).map(e=>e[0]);document.getElementById('nativeEvents').textContent=eventSteps.length?'Independently verified native optimizer surprise steps: '+eventSteps.map(x=>x.toLocaleString()).join(', ')+'. Final recorded surprise: '+eventSteps.at(-1).toLocaleString()+'. Timing alone does not establish the cause of an endpoint change.':'';document.getElementById('updated').textContent='Updated '+new Date().toLocaleTimeString()+' · read-only experiment files';}
async function refresh(){try{const r=await fetch('/api/state'+(new URLSearchParams(location.search).get('stage')==='400'?'?stage=400':''),{cache:'no-store'}),s=await r.json();if(!r.ok){document.getElementById('status').textContent='Preparing';document.getElementById('updated').textContent=s.message||'Experiment files not ready';return}current=s;render(s)}catch(e){document.getElementById('status').textContent='Refresh unavailable';document.getElementById('updated').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></html>'''


def compact_contribution(path):
    result = read_json(path) or {}
    return {name: dict(contexts=pool.get('contexts'), per_arm={arm:
                {key: value for key, value in metrics.items() if key not in ('records', 'per_subject')}
                for arm, metrics in pool.get('per_arm', {}).items()}) for name, pool in result.items()}


def file_sha(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def runtime_qualified(run, receipt, status):
    checks = receipt.get('checks') or {}
    sources = all(checks.get(name) is True for name in
                  ('application_sources_unchanged', 'native_sources_unchanged', 'immutable_inputs_unchanged'))
    return bool(receipt.get('qualified') is True and checks and all(value is True for value in checks.values())
                and sources and status and status.get('phase') == 'complete')


def independently_qualified(run, receipt):
    review = read_json(run / 'independent-review.json') or {}
    return bool(review.get('qualified') is True and review.get('final_native_digest') == receipt.get('final_native_digest')
                and review.get('artifact_sha256', {}).get('receipt.json') == file_sha(run / 'receipt.json'))


CONTINUATION_SCHEMAS = frozenset((
    'supra_particle_gated_v3_continuation_v1',
    'supra_particle_gated_v3_long_continuation_v1',
    'supra_particle_gated_v3_editing_only_long_v1',
))


def is_continued(plan):
    return plan.get('schema') in CONTINUATION_SCHEMAS


def common_dv12_initial(plan):
    key = 'common_initial' if is_continued(plan) else 'initial'
    return plan.get('input_sha256', {}).get(key)


def training_budget(plan, status):
    """Expose the declared schedule separately from the total-update horizon."""
    editing_only = bool(plan.get('schema') == 'supra_particle_gated_v3_editing_only_long_v1'
        and plan.get('training_schedule') == 'editing_only_v1'
        and plan.get('schedule_start_step') == 1600
        and plan.get('inherited_edit_updates') == 1280
        and plan.get('inherited_preservation_updates') == 320)
    step = (status or {}).get('step', plan.get('start_step', 0))
    if editing_only:
        edits = 1280 + max(0, step - 1600)
        preservation = 320
    else:
        edits, preservation = step - step // 5, step // 5
    return dict(editing_only=editing_only, law=plan.get('training_schedule', 'original_4_edit_1_preservation'),
        schedule_start_step=plan.get('schedule_start_step'), editing_updates=edits,
        preservation_updates=preservation,
        recorded_editing_updates=(status or {}).get('editing_updates'),
        recorded_preservation_updates=(status or {}).get('preservation_updates'),
        recorded_counts_match=all((status or {}).get(name, count) == count for name, count in
                                  (('editing_updates', edits), ('preservation_updates', preservation))),
        endpoint_steps=plan.get('endpoint_steps', [plan.get('fixed_updates', 400)]),
        endpoint_edit_updates=plan.get('endpoint_edit_updates', {}),
        historical_control_edit_updates=plan.get('historical_control_edit_updates'),
        historical_control_preservation_updates=plan.get('historical_control_preservation_updates'))


def qualified_history(run, ancestors=()):
    """Follow qualified continuation boundaries; never reinterpret a probe."""
    run = run.resolve()
    if run in ancestors or len(ancestors) >= 8:
        return [], False
    plan = read_json(run / 'plan.json') or {}
    horizon = plan.get('fixed_updates', 400)
    continued = is_continued(plan)
    valid = lambda row: bool(row.get('native_state_unchanged') is True and row.get('output_sigma') == .125
                             and row.get('output_metrics_used') is False)
    rows = []
    inherited_ok = False
    if continued:
        previous = Path(plan.get('input_paths', {}).get('initial', str(run / 'final.pt'))).parent.resolve()
        previous_plan = read_json(previous / 'plan.json') or {}
        receipt = read_json(previous / 'receipt.json') or {}
        review = read_json(previous / 'independent-review.json') or {}
        inputs = plan.get('input_sha256', {})
        monitor = 'scripts/monitor_e22_supra_particle_convergence.py'
        inherited_ok = bool(previous != run and independently_qualified(previous, receipt)
            and receipt.get('qualified') is True
            and inputs.get('initial') == receipt.get('final_checkpoint_sha256')
            and inputs.get('initial_receipt') == file_sha(previous / 'receipt.json')
            and inputs.get('initial_review') == file_sha(previous / 'independent-review.json')
            and review.get('artifact_sha256', {}).get('progress.jsonl') == file_sha(previous / 'progress.jsonl')
            and all(inputs.get(key) == previous_plan.get('input_sha256', {}).get(key) for key in ('judge', 'data'))
            and common_dv12_initial(plan) == common_dv12_initial(previous_plan)
            and plan.get('profile') == previous_plan.get('profile')
            and plan.get('critic_digests', {}).get('D1856') == previous_plan.get('critic_digests', {}).get('D1856')
            and plan.get('particlegan_source_sha256') == previous_plan.get('particlegan_source_sha256')
            and plan.get('application_source_sha256', {}).get(monitor)
                == previous_plan.get('application_source_sha256', {}).get(monitor)
            and all(isinstance(read_json(run / f'progress-{task}-indices.json'), list)
                and read_json(run / f'progress-{task}-indices.json') == read_json(previous / f'progress-{task}-indices.json')
                for task in ('edit', 'preservation')))
        if inherited_ok:
            prior, prior_ok = qualified_history(previous, (*ancestors, run))
            if is_continued(previous_plan) and not prior_ok:
                inherited_ok = False
            else:
                rows = [dict(row, probe_origin=f"qualified {previous_plan.get('fixed_updates', 400)} boundary; "
                             + row.get('probe_origin', 'recorded probe'))
                        for row in prior if row['step'] <= plan.get('start_step', 0)]
    rows += [dict(row, probe_origin=f"recorded {plan.get('start_step', 0)}→{horizon} continuation" if continued else 'fresh run')
             for row in read_history(run / 'progress.jsonl') if valid(row)]
    return sorted({row['step']: row for row in rows}.values(), key=lambda row: row['step']), inherited_ok


def snapshot(run, baselines, control_run=ROOT / 'outputs/e22-pr223-shared-1600', expected_horizon=None):
    plan = read_json(run / 'plan.json') or {}
    status = read_json(run / 'status.json')
    horizon = plan.get('fixed_updates', expected_horizon or 400)
    continued = is_continued(plan)
    initial_run = Path(plan.get('input_paths', {}).get('initial', str(run / 'final.pt'))).parent if continued else run
    indices = {task: read_json(run / f'progress-{task}-indices.json') for task in ('edit', 'preservation')}
    monitor = 'scripts/monitor_e22_supra_particle_convergence.py'
    input_sha = plan.get('input_sha256', {})
    common_initial = common_dv12_initial(plan)
    valid_point = lambda row: bool(row.get('native_state_unchanged') is True and row.get('output_sigma') == .125
                                  and row.get('output_metrics_used') is False)
    history, inherited_ok = qualified_history(run)
    latest = history[-1] if history else None
    failure = read_json(run / 'failed-checks.json') or {}
    failure_names = [name for name, value in failure.items() if value is False]
    if failure_names:
        status = dict(status or {}, phase='failed' if (run / 'train.jsonl').is_file() else 'preflight_failed',
                      failed_checks=failure_names)
    control_plan = read_json(control_run / 'plan.json') or {}
    control_review = read_json(control_run / 'independent-review.json') or read_json(control_run / 'qualification-review.json') or {}
    control_sha = control_plan.get('input_sha256', {})
    control_ok = bool(control_review.get('qualified') is True
        and control_review.get('artifact_sha256', {}).get('progress.jsonl') == file_sha(control_run / 'progress.jsonl')
        and all(input_sha.get(key) == control_sha.get(key) for key in ('judge', 'data'))
        and common_initial == control_sha.get('shared_initial')
        and plan.get('profile') == control_plan.get('profile')
        and plan.get('particlegan_commit') == control_plan.get('particlegan_commit')
        and plan.get('critic_digests', {}).get('D1856')
            == control_review.get('scoring_critic_digests', {}).get('common_frozen1856')
        and plan.get('application_source_sha256', {}).get(monitor)
            == control_plan.get('application_source_sha256', {}).get(monitor)
        and all(isinstance(selected, list) and selected == read_json(control_run / f'progress-{task}-indices.json')
                for task, selected in indices.items()))
    historical_ok = bool(horizon == 6400 and control_review.get('qualified') is True
        and control_review.get('artifact_sha256', {}).get('progress.jsonl') == file_sha(control_run / 'progress.jsonl')
        and input_sha.get('judge') == control_sha.get('final.pt')
        and input_sha.get('data') == control_sha.get('data.pt')
        and plan.get('critic_digests', {}).get('D1856') == control_review.get('common_start_critic_digest')
        and plan.get('application_source_sha256', {}).get(monitor)
        and plan.get('application_source_sha256', {}).get(monitor)
            == control_review.get('source', {}).get('manifest', {}).get('application', {}).get(monitor)
        and all(isinstance(selected, list) and selected == read_json(control_run / f'progress-{task}-indices.json')
                for task, selected in indices.items()))
    control = [row for row in read_history(control_run / 'progress.jsonl')
               if valid_point(row) and row.get('step', horizon + 1) <= horizon] if control_ok else []
    if historical_ok:
        control = [dict(row, probes={task: dict(contexts=probe.get('contexts'),
                    clean=dict(frozen_start_D=probe.get('clean', {}).get('frozen_start_D')))
                    for task, probe in row.get('probes', {}).items()})
                   for row in read_history(control_run / 'progress.jsonl')
                   if valid_point(row) and row.get('step', horizon + 1) <= horizon]
    if not continued:
        point = read_json(run / 'control-progress400.json')
        if not control and point and valid_point(point):
            control = [point]
    reference = read_json(baselines) or {}
    metadata = reference.get('reference', {})
    gold_compatible = bool(input_sha.get('judge') and input_sha.get('data')
        and metadata.get('critic_checkpoint_sha256') == input_sha.get('judge')
        and metadata.get('data_sha256') == input_sha.get('data') and metadata.get('output_sigma') == .125
        and metadata.get('noise_generator') == 'private CPU72' and metadata.get('gaussian_panels') == 4
        and metadata.get('probe_indices') == indices)
    receipt = read_json(run / 'receipt.json') or {}
    checks = receipt.get('checks') or {}
    review = read_json(run / 'independent-review.json') or {}
    endpoint_metrics = {}
    for endpoint in plan.get('endpoint_steps', []):
        record_name = f'endpoint-{endpoint:05d}.json'
        metrics_name = f'evaluation-v3-step-{endpoint:05d}.json'
        record = read_json(run / record_name)
        metrics = read_metrics(run / metrics_name)
        if record or metrics:
            endpoint_metrics[str(endpoint)] = dict(record=record, gated=metrics,
                independent_qualified=bool(independently_qualified(run, receipt)
                    and record is not None and metrics is not None
                    and all(file_sha(run / name)
                            and review.get('artifact_sha256', {}).get(name) == file_sha(run / name)
                            for name in (record_name, metrics_name))))
    control_metrics = read_metrics(run / 'evaluation-v2.json')
    gated_metrics = read_metrics(run / 'evaluation-v3.json')
    audit = read_json(run / 'audit-final.json') or read_json(run / 'audit-step-two.json')
    if not audit and inherited_ok:
        audit = read_json(initial_run / 'audit-final.json')
    audit_summary = None
    if audit:
        audit_summary = dict(step=audit.get('step'), branch_count=len(audit.get('branches', {})),
            clean_bank_rows=audit.get('clean_gradient', {}).get('bank_rows_nonzero'),
            dv12_bank_rows=audit.get('dv12_gradient', {}).get('bank_rows_nonzero'),
            site_effects_nonzero=audit.get('final_output_site_effects_nonzero_diagnostic'),
            immutable=audit.get('native_state_rng_and_gradients_unchanged'))
    archive_run = ROOT / 'outputs/e22-particle-gated-v3-400'
    archive_receipt = read_json(archive_run / 'receipt.json') or {}
    archive_qualified = independently_qualified(archive_run, archive_receipt)
    archive_comparison = archive_receipt.get('comparison') or {}
    archive_better = bool(archive_qualified and set(archive_comparison) == {'fit', 'test', 'holds', 'preservation'}
        and all(pool.get(judge, {}).get('new_minus_control', 0) < 0 for pool in archive_comparison.values()
                for judge in ('fixed_start_D', 'arm_final_D')))
    return bool(status or history), dict(run=str(run), updated_at=datetime.now(timezone.utc).isoformat(),
        plan=plan, status=status, horizon=horizon, start_step=plan.get('start_step', 0), continued=continued,
        training_budget=training_budget(plan, status),
        history=history, latest=latest, inherited_history_qualified=inherited_ok, failed_checks=failure_names,
        control=control, control_compatible=control_ok or historical_ok, control_dv12_compatible=control_ok,
        control_kind='historical' if historical_ok or (horizon == 6400 and control_sha.get('final.pt')) else 'shared',
        control_profile=plan.get('control_profile', control_plan.get('profile', 'pr155_routed')),
        control_commit=plan.get('control_particlegan_commit', control_plan.get('particlegan_commit')),
        gold_compatible=gold_compatible,
        gold=[dict(label=curve.get('label'), points=curve.get('points', []))
              for curve in reference.get('curves', [])] if gold_compatible else [],
        audit=audit_summary, contribution=compact_contribution(run / 'particle-contribution.json'),
        contribution_final_judge='common_control_final_D' if continued else 'common_D400',
        verified_native_events=dict(surprise_log=review.get('guard_activity', {}).get('surprise_log', []),
                                    source='independent-review.json')
            if independently_qualified(run, receipt) else None,
        archived400=dict(qualified=archive_qualified, all_pooled_games_better=archive_better,
                         url='http://pop-os:8782/?stage=400'),
        final=dict(available=control_metrics is not None and gated_metrics is not None,
            runner_qualified=runtime_qualified(run, receipt, status), independent_qualified=independently_qualified(run, receipt),
            check_count=len(checks), failed_checks=[name for name, value in checks.items() if value is not True],
            control=control_metrics, gated=gated_metrics, comparison=receipt.get('comparison'),
            endpoints=endpoint_metrics,
            originals={label: read_metrics(run / f'evaluation-original-{label}.json')
                       for label in ('converged_1600', 'published_1600', 'converged_6400')},
            original_comparison=receipt.get('original_comparison')),
        evaluation_only=True, output_metrics_used_for_optimization_or_selection=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/e22-particle-gated-v3-400')
    parser.add_argument('--baselines', type=Path,
                        default=ROOT / 'outputs/e22-dashboard-baselines/fixed-reference-baselines.json')
    parser.add_argument('--control-run', type=Path, default=ROOT / 'outputs/e22-pr223-shared-1600')
    parser.add_argument('--expected-horizon', type=int, choices=(400, 1600, 6400))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8782)
    args = parser.parse_args()
    run = args.run.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            path = request.path
            if path == '/':
                self.respond(200, HTML.encode(), 'text/html; charset=utf-8')
                return
            if path not in ('/api/state', '/api/status', '/api/history', '/api/references', '/api/final', '/api/contribution'):
                self.send_error(404)
                return
            archive = parse_qs(request.query).get('stage') == ['400']
            selected_run = ROOT / 'outputs/e22-particle-gated-v3-400' if archive else run
            selected_control = ROOT / 'outputs/e22-pr223-shared-1600' if archive else args.control_run
            ready, state = snapshot(selected_run, args.baselines, selected_control, 400 if archive else args.expected_horizon)
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
