"""Stable collection must not import excluded candidate modules or HNDL."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(version("supra-particle-sliders") != "0.3.0", reason="stable collection qualification")
def test_stable_collection_does_not_import_candidate_or_hndl(tmp_path):
    tests = Path(__file__).resolve().parent
    program = '''import json,sys,pytest
result=pytest.main(["--collect-only","--import-mode=importlib","-p","no:cacheprovider","-o","pythonpath="+sys.argv[1],sys.argv[1],"-q"])
blocked=[name for name in sys.modules if name in ("hndl","hypergan","supra.hndl","supra.hndl_lifecycle","supra.hndl_inference") or name.startswith(("hndl.","hypergan."))]
print(json.dumps(dict(result=int(result),forbidden_imports=blocked)))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(tests)], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == dict(result=0, forbidden_imports=[]), result.stdout
