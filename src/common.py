# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : (library, imported by the numbered steps)
# DESCRIPTION  : Paths, command-line arguments and small I/O helpers shared by the numbered steps.
#                All paths are computed from this file's location; nothing is absolute.
# INPUT FILES  : config/params.json
# OUTPUT FILES : none (imported)
# ============================================================
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CONFIG = ROOT / "config"


def params():
    return json.loads((CONFIG / "params.json").read_text())


def args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run-dir", type=Path, required=True, help="outputs/<run> folder created by run_all.py")
    ap.add_argument("--mode", choices=["demo", "full"], required=True)
    a = ap.parse_args()
    a.run_dir = a.run_dir.resolve()
    a.work = a.run_dir / "work"          # row-level intermediate files (never copied to reference/)
    a.work.mkdir(parents=True, exist_ok=True)
    return a


def step_dir(a, name):
    d = a.run_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_raw(filename):
    """Locate a replication-package file anywhere under data/raw/ (users unzip packages as they like)."""
    hits = sorted(Path(dp) / filename for dp, _, files in os.walk(RAW, followlinks=True) if filename in files)
    if not hits:
        raise SystemExit(f"Missing data/raw/**/{filename}. See data/README.md for where to obtain it.")
    if len(hits) > 1:
        raise SystemExit(f"Found {len(hits)} copies of {filename} under data/raw/; keep one.")
    return hits[0]


def to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=to_jsonable)


def rel(p):
    return os.path.relpath(Path(p).resolve(), ROOT)
