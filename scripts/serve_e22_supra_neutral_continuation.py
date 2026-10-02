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
    review = parent_view.read_json(run / "independent-review.json", {})
    return parent_view.finite_json(dict(
        updated=datetime.now(timezone.utc).isoformat(), status=status, arms=arms,
        historical_probes=probes,
        historical_references=references, selected_reference_verified=selected_verified,
        receipt=current["receipt"], parent_receipt=prior["receipt"], failure=current["failure"],
        continuation_qualified=review.get("qualified") is True,
        parent_path=str(parent), continuation_path=str(run),
    ))


HTML = (Path(__file__).with_name("supra_continuation_dashboard.html")).read_text()


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
