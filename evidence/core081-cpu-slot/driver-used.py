"""One admitted offline Supra CPU validation worker; no global installs."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "supra"
HEAD = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
RUN = ROOT / "validation" / HEAD[:12]
RUN.mkdir(parents=True, exist_ok=True)
EVIDENCE = REPO / "evidence" / "core081-cpu-slot" / HEAD[:12]
EVIDENCE.mkdir(parents=True, exist_ok=True)
LIMIT, RESERVE = 2 * 1024**3, 16 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
BASE_PYTHON = "/ml2/codex-storage/future-workspaces/2026-10-21/task-2/.venv-core081/bin/python"
VENV = ROOT / ".venv-supra-validation"
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
           SUPRA_REFERENCE_CORE=str(ROOT / "core-milestone/packages/particle-sliders-core/src/particle_sliders"))
ENV.pop("PARTICLE_SLIDERS_ROOT", None)


def available():
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(rows["MemAvailable"].strip().split()[0]) * 1024


def run(label, command, cwd, timeout=120):
    before = available()
    if before < RESERVE + LIMIT:
        raise RuntimeError("Host reserve admission rejected before " + label)
    log = EVIDENCE / (label + ".log")
    start = time.monotonic()
    minimum = before
    stopped = None
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            free = available()
            minimum = min(minimum, free)
            if free < RESERVE or time.monotonic() - start > timeout:
                stopped = "host reserve" if free < RESERVE else "timeout"
                process.terminate()
                break
            time.sleep(.2)
        code = process.wait(timeout=10)
    receipt = dict(source_head=HEAD, command=command, cwd=str(cwd), exit_code=code,
        elapsed_seconds=time.monotonic()-start, address_space_limit_bytes=LIMIT,
        minimum_host_available_bytes=minimum, required_host_reserve_bytes=RESERVE,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        operator_threads=1, interop_threads=1, cuda_visible_devices="", offline=True,
        stopped=stopped)
    (EVIDENCE / (label + "-resources.json")).write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)
    print(log.read_text()[-12000:], flush=True)
    if code or stopped:
        raise SystemExit(code or 2)


if not VENV.exists():
    run("venv", [BASE_PYTHON, "-I", "-c", "import resource,venv,sys; resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648)); venv.create(sys.argv[1],with_pip=False)", str(VENV)], "/tmp")
SITE = VENV / "lib/python3.12/site-packages"
(SITE / "borrowed-cpu-dependencies.pth").write_text("/ml2/iris-particle-sliders-task-20261008/.venv/lib/python3.12/site-packages\n")
core_wheel = Path("/ml2/codex-storage/future-workspaces/2026-10-21/task-2/evidence/core081-build/particle_sliders_core-0.8.1-py3-none-any.whl")
pg_wheel = Path("/ml2/codex-storage/future-workspaces/2026-10-21/task-2/dependencies/particlegan-0.9.1-py3-none-any.whl")
assert hashlib.sha256(core_wheel.read_bytes()).hexdigest() == "279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d"
for wheel in (core_wheel, pg_wheel):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(SITE)
SOURCE = RUN / "source"
SOURCE.mkdir(exist_ok=True)
data = subprocess.check_output(["git", "-C", str(REPO), "archive", HEAD])
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    archive.extractall(SOURCE, filter="data")
DIST = RUN / "dist"
DIST.mkdir(exist_ok=True)
PYTHON = str(VENV / "bin/python")
if not any(DIST.glob("*.whl")):
    run("build", [PYTHON, "-I", "-c", "import resource,sys; resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648)); import setuptools.build_meta as b; print(b.build_wheel(sys.argv[1]))", str(DIST)], SOURCE)
product_wheel = next(DIST.glob("*.whl"))
with zipfile.ZipFile(product_wheel) as archive:
    archive.extractall(SITE)

worker = '''import sys,resource,json,hashlib,importlib.util
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
import supra,particle_sliders,particlegan
import importlib.metadata as metadata
assert metadata.version('particle-sliders-core')=='0.8.1'
assert metadata.version('particlegan')=='0.9.1'
assert 'source' not in str(Path(supra.__file__).resolve()).split('/')
for name in ('supra_model','pg09_remaining_campaigns','pg09_moss_campaign'):
    assert importlib.util.find_spec('particle_sliders.'+name) is None
import supra.lifecycle,supra.native_inference
from particle_sliders.lifecycle import backend
from particle_sliders.inference import inference_backend
assert backend('supra').name == inference_backend('supra').name == 'supra'
receipt=json.loads(Path(supra.__file__).with_name('core081-receipt.json').read_text())
root=Path(particle_sliders.__file__).parent.parent
for key,digest in receipt['source_sha256'].items():
    assert hashlib.sha256((root/key).read_bytes()).hexdigest()==digest,key
origins=dict(supra=supra.__file__,core=particle_sliders.__file__,pg=particlegan.__file__,
            torch=torch.__version__,core_version=metadata.version('particle-sliders-core'),
            pg_version=metadata.version('particlegan'),cuda_initialized=torch.cuda.is_initialized(),canonical_hashes=len(receipt['source_sha256']))
Path(sys.argv[1]).write_text(json.dumps(origins,indent=2))
print(json.dumps(origins),flush=True)
import pytest
code=pytest.main(sys.argv[2:])
assert not torch.cuda.is_initialized()
raise SystemExit(code)
'''
artifacts = dict(source_head=HEAD, core_head="32f6830f6402e85aa77b6631db64fcbbb2131f6b",
    artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (core_wheel,pg_wheel,product_wheel)},
    evidence=str(EVIDENCE), native_reference_head="fd89e1c780ab5210cd1be98ec3b70977e6349379")
(EVIDENCE / "artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
run("tests", [PYTHON, "-I", "-c", worker, str(EVIDENCE / "installed-origins.json"),
    "-q", str(SOURCE / "tests/core081"), "--import-mode=importlib", "-o", "pythonpath=",
    "--tb=short", "-p", "no:cacheprovider", "--junitxml="+str(EVIDENCE / "tests.xml")], "/tmp")
print(json.dumps(artifacts),flush=True)
