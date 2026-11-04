"""Keep the opt-in HNDL candidate out of the supported wheel."""
from setuptools import setup
from setuptools.command.build_py import build_py
from pathlib import Path


class StableBuildPy(build_py):
    def run(self):
        super().run()
        # A previous build may have left candidate files in build/lib.
        for name in ("hndl.py", "hndl_lifecycle.py", "hndl_inference.py",
                     "supra_lora.hndl", "core012-receipt.json", "core012-a794e3-receipt.json",
                     "core012-ddb4142-hndl-receipt.json"):
            (Path(self.build_lib) / "supra" / name).unlink(missing_ok=True)

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [item for item in modules if not (
            item[0] == "supra" and item[1] in {"hndl", "hndl_lifecycle", "hndl_inference"})]


setup(cmdclass={"build_py": StableBuildPy})
