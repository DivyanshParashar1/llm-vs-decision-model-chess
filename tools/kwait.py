"""Wait for Kaggle notebooks without being fooled by login/network problems.

Kaggle notebooks run server-side and keep going regardless of this machine; this only watches them.

  python tools/kwait.py chess-02-llm0 chess-03-llm-star [--fetch]

Exits as soon as ANY listed notebook finishes (so the next step can start), printing all statuses.
Exit codes: 0 = one finished OK, 1 = one errored/cancelled, 3 = login expired (run `kaggle auth login --force`).
--fetch downloads only small outputs (json/log/csv) of finished notebooks into build/<slug>/out.
"""
import argparse
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUTH_HINTS = ("Authentication required", "kernels.get' was denied", "401", "Unauthorized", "invalid_grant")


def run(args):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def user():
    _, out = run(["config", "view"])
    m = re.search(r"username: (\S+)", out)
    return m.group(1) if m else None


def status(ref):
    code, out = run(["kernels", "status", ref])
    m = re.search(r'status "KernelWorkerStatus\.(\w+)"', out)
    if m:
        return m.group(1)
    if any(h in out for h in AUTH_HINTS):
        return "AUTH"
    return "UNKNOWN"


def refresh_login():
    """Ask the CLI for a fresh access token (uses the cached refresh token)."""
    code, _ = run(["auth", "print-access-token"])
    return code == 0


def fetch(ref, slug):
    out = ROOT / "build" / slug / "out"
    out.mkdir(parents=True, exist_ok=True)
    run(["kernels", "output", ref, "-p", str(out), "-o", "--file-pattern", r".*\.(json|log|csv)$"])
    print(f"fetched small outputs -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--every", type=int, default=90, help="seconds between checks")
    ap.add_argument("--auth-grace-min", type=int, default=30, help="keep retrying login this long before giving up")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()

    u = user() or "divyanshparashar1"
    refs = {s: s if "/" in s else f"{u}/{s}" for s in a.slugs}
    auth_fail_since = None
    while True:
        st = {s: status(r) for s, r in refs.items()}
        stamp = time.strftime("%H:%M:%S")
        if "AUTH" in st.values():
            auth_fail_since = auth_fail_since or time.time()
            mins = (time.time() - auth_fail_since) / 60
            ok = refresh_login()
            print(f"[{stamp}] login problem ({mins:.0f} min); token refresh {'ok' if ok else 'failed'}; "
                  f"notebooks keep running on Kaggle", flush=True)
            if mins >= a.auth_grace_min:
                print("LOGIN EXPIRED: run `kaggle auth login --force` in a terminal, then re-run kwait.", flush=True)
                sys.exit(3)
            time.sleep(60)
            continue
        auth_fail_since = None
        done = {s: v for s, v in st.items() if v in ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED")}
        if done:
            print(f"[{stamp}] " + ", ".join(f"{s}: {v}" for s, v in st.items()), flush=True)
            if a.fetch:
                for s in done:
                    fetch(refs[s], s.split("/")[-1])
            sys.exit(0 if all(v == "COMPLETE" for v in done.values()) else 1)
        time.sleep(a.every)  # RUNNING / QUEUED / transient UNKNOWN (network): keep waiting


if __name__ == "__main__":
    main()
