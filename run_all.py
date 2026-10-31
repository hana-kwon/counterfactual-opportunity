# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : demo ~4 min, full ~3 min on a laptop CPU
# DESCRIPTION  : Runs the whole pipeline into a new, empty output folder and writes a receipt.
#                --demo (default): hypothetical firm pairs (cached real model answers) + two synthetic
#                randomized experiments with known effects, evaluated by the same code as the full path.
#                --full: needs the two replication packages in data/raw/ (see data/README.md) and
#                reproduces the proposal's YouWiN! and Egypt numbers from the cached model answers.
# INPUT FILES  : data/hypothetical_firms/*, data/model_outputs/* (full), data/raw/** (full)
# OUTPUT FILES : outputs/<run>/... and outputs/<run>/receipt.json
# ============================================================
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    "demo": ["01_hypothetical_firms", "02_synthetic_experiments", "05_grant_policy_value", "06_grant_heterogeneity",
             "07_lending_evaluation", "08_simulation_check", "09_figures", "10_compare_reference"],
    "full": ["01_hypothetical_firms", "03_youwin_build", "04_egypt_build", "05_grant_policy_value",
             "06_grant_heterogeneity", "07_lending_evaluation", "09_figures", "10_compare_reference"],
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(mode):
    files = sorted((ROOT / "data" / "hypothetical_firms").glob("*.jsonl")) + [ROOT / "config" / "params.json"]
    if mode == "full":
        files += sorted((ROOT / "data" / "model_outputs").glob("*.csv")) + [ROOT / "config" / "proposal_numbers.json"]
        needed = {"BaselineandFirstFollowup.dta", "ThirdFollowup.dta", "16-ABA-All-Merged.dta",
                  "20-ABA-ML-group.dta", "10-ABA-baseline-clean.dta"}
        for dp, _, fs in os.walk(ROOT / "data" / "raw", followlinks=True):
            files += [Path(dp) / f for f in fs if f in needed]
    return files


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="synthetic experiments (default)")
    g.add_argument("--full", action="store_true", help="replication packages in data/raw/")
    ap.add_argument("--out", type=Path, default=None, help="output folder (must be new or empty)")
    a = ap.parse_args()
    mode = "full" if a.full else "demo"
    out = (a.out or ROOT / "outputs" / f"{mode}_{datetime.now().strftime('%y%m%d_%H%M%S')}").resolve()
    # ★ IMPORTANT: never write into a folder that already holds results.
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Output folder {out} is not empty; choose a new one.")
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"mode": mode, "started": datetime.now().isoformat(timespec="seconds"), "environment": versions(),
               "steps": [], "not_done": ["no language model was called; cached answers are scored",
                                         "no replication data or row-level file is written outside outputs/<run>/work"]}
    t_all = time.time()
    for step in STEPS[mode]:
        t0 = time.time()
        print(f"== {step}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "src" / f"{step}.py"), "--run-dir", str(out), "--mode", mode],
                           cwd=str(ROOT))
        receipt["steps"].append({"step": step, "seconds": round(time.time() - t0, 1), "exit_code": r.returncode})
        if r.returncode != 0:
            receipt["failed_at"] = step
            break
    receipt["seconds_total"] = round(time.time() - t_all, 1)
    receipt["inputs"] = {os.path.relpath(p, ROOT): sha256(p) for p in input_files(mode)}
    receipt["outputs"] = {os.path.relpath(p, out): sha256(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name != "receipt.json"}
    with open(out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"done in {receipt['seconds_total']} s; receipt: {out / 'receipt.json'}")
    if "failed_at" in receipt:
        sys.exit(1)


if __name__ == "__main__":
    main()
