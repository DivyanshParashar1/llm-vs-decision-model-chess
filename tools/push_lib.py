"""Publish src/chessdecide as a private Kaggle dataset `<user>/chessdecide-lib`
so notebooks can `sys.path.append` it. Creates it the first time, adds a version after.

Usage: python tools/push_lib.py "message"
"""
import json
import pathlib
import shutil
import subprocess
import sys

from kpush import ROOT, kaggle_user

SLUG = "chessdecide-lib"


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "update"
    user = kaggle_user()
    build = ROOT / "build" / SLUG
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(ROOT / "src" / "chessdecide", build / "chessdecide",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (build / "dataset-metadata.json").write_text(json.dumps(
        {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"],
                            capture_output=True, text=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", str(build), "-m", msg, "--dir-mode", "zip"] if exists
           else ["kaggle", "datasets", "create", "-p", str(build), "--dir-mode", "zip"])
    subprocess.run(cmd, check=True)
    print(f"dataset: {user}/{SLUG}")


if __name__ == "__main__":
    main()
