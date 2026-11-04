"""Reuse the existing native fixture as data, without adding product source paths."""
import importlib.util
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / "core081/test_thin_relocation.py"
spec = importlib.util.spec_from_file_location("supra_hndl_native_fixture", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
TinyRuntime, assets, equal = module.TinyRuntime, module.assets, module.equal
