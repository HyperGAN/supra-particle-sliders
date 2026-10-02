#!/usr/bin/env python3
"""Read-only progress across the qualified particle run and its continuation."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import serve_e22_supra_neutral_initialization as parent_view

ARMS = parent_view.ARMS
PARENT_ENDPOINTS = (5120, 6400)
NEW_ENDPOINTS = (9600, 12800)


def state(run, parent):
    """Combine observation files without writing to either run or loading models."""
    prior = parent_view.state(parent)
    current = parent_view.state(run)
    prior_probes = prior["historical_probes"]
    new_probes = current["historical_probes"] or parent_view.read_json(
        run / "historical-probes/references.json", {})
    probes = {**prior_probes, **new_probes}
    probes["scores"] = {**prior_probes.get("scores", {}), **new_probes.get("scores", {})}
    references = {**prior["historical_references"], **current["historical_references"]}
    selected_path = ROOT / "outputs/e22-supra-historical-selected-28000"
    selected = parent_view.read_json(selected_path / "report.json", {})
    completion = parent_view.read_json(selected_path / "completion.json", {})
    selected_verified = (completion.get("complete") is True and completion.get("qualified") is True
        and selected.get("qualified") is True and selected.get("training_updates") == 0
        and selected.get("card", {}).get("parent_run") == str(parent.resolve())
        and selected.get("probe_contract", {}).get("indices") == probes.get("indices"))
    if selected_verified:
        selected_verified = hashlib.sha256((selected_path / "report.json").read_bytes()).hexdigest() == completion.get("report_sha256")
    if selected_verified:
        probes["scores"]["ordinary_lora_selected_28000"] = selected["probes"]["ordinary_lora_selected_28000"]
        references["ordinary_lora_selected_28000"] = selected["results"]["ordinary_lora_selected_28000"]
    arms = {}
    for arm in ARMS:
        old_history = prior["arms"][arm]["history"]
        new_history = current["arms"][arm]["history"]
        # The boundary belongs to the parent; continuation history starts later.
        history = old_history + [row for row in new_history if row.get("step", 0) > 6400]
        arms[arm] = dict(history=history, evaluations={
            **prior["arms"][arm]["evaluations"],
            **{str(step): parent_view.read_json(run / arm / f"evaluation-{step:05d}.json")
               for step in NEW_ENDPOINTS},
        })
    status = current["status"] or dict(phase="preparing continuation", step=6400,
                                      editing_updates=6400, preservation_updates=0, steps=12800)
    return parent_view.finite_json(dict(
        updated=datetime.now(timezone.utc).isoformat(), status=status, arms=arms,
        historical_probes=probes,
        historical_references=references, selected_reference_verified=selected_verified,
        receipt=current["receipt"], parent_receipt=prior["receipt"], failure=current["failure"],
        parent_path=str(parent), continuation_path=str(run),
    ))


def replace_once(html, old, new):
    if html.count(old) != 1:
        raise ValueError("parent dashboard template changed: " + old[:70])
    return html.replace(old, new)


HTML = parent_view.HTML
for old, new in (
    ("<title>Supra · sampled particle initialization</title>",
     "<title>Supra · particle continuation</title>"),
    ("<h1>Supra · particle initialization comparison</h1>",
     "<h1>Supra · particle continuation to 12,800 edits</h1>"),
    ("71 gated particle sites · shared 128 × 4 bank · two fresh public sampled-initialization runs",
     "71 gated particle sites · shared 128 × 4 bank · exact native continuation from 6,400 edits"),
    ("Gold: the historical ordinary LoRA checkpoint trained for 6,400 total updates, including 5,120 editing updates. The fresh runs train on editing only through 6,400 updates. Historical references use a different initialization cohort and training schedule.",
     "Ordinary LoRA is the standard low-rank adapter without a particle bank or routing. The original selected 28,000-update model is the best-model target; it had 22,400 editing and 5,600 preservation updates under MSE/AdamW. The 6,400 and 12,800 ordinary checkpoints are separate fixed-horizon references. The particle runs use the native learned game and editing only through 12,800 updates. The horizontal levels use the small progress probe; full-task scores appear below."),
    ("<h2>Both fixed endpoints · full held-out editing set</h2>",
     "<h2>Fixed endpoints · full held-out editing set</h2>"),
    ("Both fresh runs have exactly the editing updates shown.",
     "Both particle runs have exactly the editing updates shown. The 5,120/6,400 rows are qualified parent results; 9,600/12,800 are the mandatory continuation endpoints."),
    ("maximum=6400", "maximum=12800"),
    ("+' / 6,400 edits'", "+' / 12,800 edits'"),
    ("const referenceLevels=[{name:'Ordinary LoRA target',y:gold,color:'#f2c474',target:true},",
     "const referenceLevels=[{name:'Original selected LoRA 28,000 · best-model target',y:s.historical_probes?.scores?.ordinary_lora_selected_28000,color:'#f2c474',target:true},{name:'Ordinary LoRA 6,400 · fixed horizon',y:gold,color:'#c9984e'},{name:'Ordinary LoRA 12,800 · budget context',y:s.historical_probes?.scores?.ordinary_lora_12800,color:'#b5a47d'},"),
    ("<span style=\"--color:#f2c474\">Ordinary LoRA · target</span>",
     "<span style=\"--color:#f2c474\">Original selected LoRA 28,000 · best-model target</span><span style=\"--color:#c9984e\">Ordinary LoRA 6,400 · fixed horizon</span><span style=\"--color:#b5a47d\">Ordinary LoRA 12,800 · budget context</span>"),
    ("difference from ordinary target ", "difference from ordinary 6,400 reference "),
    ("const ng=game(latest.sampled_hb_neutral),cg=game(latest.sampled_control);",
     "const ng=game(latest.sampled_hb_neutral),cg=game(latest.sampled_control);gold=s.historical_probes?.scores?.ordinary_lora_selected_28000;"),
    ("arms=s.arms||{},gold=", "arms=s.arms||{};let gold="),
    ("the ordinary target ", "the original selected 28,000 target "),
    ("<th>Ordinary target</th>", "<th>Ordinary 6,400 reference</th>"),
    ("<div id=\"endpointStatus\" class=\"muted\" style=\"margin-top:10px\"></div>",
     "<div id=\"endpointStatus\" class=\"muted\" style=\"margin-top:10px\"></div><div id=\"bestReference\" class=\"muted gold\" style=\"margin-top:10px\"></div>"),
    ("const receipt=s.receipt;",
     "const best=s.historical_references?.ordinary_lora_selected_28000?.test;document.getElementById('bestReference').textContent=best?'Original selected 28,000 · all '+best.count+' test contexts: D1856 '+num(best.D1856)+'; D6400 '+num(best.D6400)+'; RMSE diagnostic '+num(best.rmse_diagnostic)+'. This is the best-model target, with a larger historical training budget.':'Selected 28,000 full-task reference pending.';const receipt=s.receipt;"),
    ("for(const step of [5120,6400])", "for(const step of [5120,6400,9600,12800])"),
    ("complete===4?'Both declared endpoints and both judges are available.':'Waiting for '+(4-complete)+' endpoint/judge comparisons. The 5,120 and 6,400 results are both required.'",
     "complete===8?'All four fixed endpoints and both judges are available.':'Waiting for '+(8-complete)+' endpoint/judge comparisons. Both continuation endpoints, 9,600 and 12,800, are required.'"),
):
    HTML = replace_once(HTML, old, new)


def make_handler(run, parent):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                payload, content_type = HTML.encode(), "text/html; charset=utf-8"
            elif path in ("/api", "/api/state"):
                payload = json.dumps(state(run, parent), allow_nan=False).encode()
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
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-supra-neutral-continuation-12800")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8784)
    args = parser.parse_args()
    if args.run.resolve() == args.parent.resolve():
        parser.error("parent and continuation observations must have distinct directories")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.run.resolve(), args.parent.resolve()))
    print(f"Serving read-only Supra continuation on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
