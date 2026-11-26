#!/usr/bin/env python3
"""Launch / poll / fetch Kaggle script kernels that run this repo at an explicit pushed commit.

  launch : python scripts/kaggle_run.py launch NAME --cmd "python scripts/x.py" [--gpu|--cpu]
             [--accelerator NvidiaTeslaT4] [--source OWNER/SLUG ...] [--output PATH ...]
             [--timeout SEC] [--purpose TEXT] [--wait]
  status : python scripts/kaggle_run.py status NAME
  fetch  : python scripts/kaggle_run.py fetch NAME     (download outputs, sync small results, append ledger row)

Kernel slug: <kaggle user>/atml4-<NAME>. Launch info is kept in kaggle_runs/<NAME>/launch.json.
Accelerator: this CLI (kaggle 2.2.5) sends `machine_shape` = --accelerator if given, else kernel-metadata
`machine_shape`; `enable_gpu` is the boolean GPU switch. What actually ran is read from nvidia-smi in
run_meta.json, never assumed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "kaggle_runs"
LEDGER = ROOT / "docs" / "gpu_ledger.md"
TEMPLATE = ROOT / "kaggle" / "bootstrap_template.py"
REPO_URL = "https://github.com/FaaizKhan1999/atml-assignment4.git"
SYNC_EXT = {".json", ".jsonl", ".csv", ".gz", ".npz"}
SYNC_MAX_BYTES = 5 * 1024 * 1024
TERMINAL = ("complete", "error", "cancel")


def run(cmd: list[str], check=True, cwd=ROOT) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return (p.stdout + p.stderr).strip()


def kaggle_user() -> str:
    out = run(["kaggle", "config", "view"])
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        raise SystemExit("cannot determine Kaggle username from `kaggle config view`")
    return m.group(1)


def clean_pushed_commit() -> str:
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"])
    sub_dirty = run(["git", "submodule", "status"])
    if dirty:
        raise SystemExit(f"refusing to launch: working tree has uncommitted changes:\n{dirty}")
    if sub_dirty.startswith(("+", "-", "U")):
        raise SystemExit(f"refusing to launch: submodule not at recorded commit:\n{sub_dirty}")
    sha = run(["git", "rev-parse", "HEAD"])
    run(["git", "fetch", "--quiet", "origin"])
    remote = run(["git", "branch", "-r", "--contains", sha], check=False)
    if not remote.strip():
        raise SystemExit(f"refusing to launch: commit {sha} is not on any pushed remote branch")
    return sha


def slug_for(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", name):
        raise SystemExit("NAME must be lowercase letters/digits/hyphens")
    return f"atml4-{name}"


def launch(a) -> None:
    sha = clean_pushed_commit()
    user = kaggle_user()
    slug = slug_for(a.name)
    kernel_id = f"{user}/{slug}"
    kdir = RUNS / a.name / "kernel"
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    cfg = {"repo_url": REPO_URL, "commit": sha, "command": a.cmd, "outputs": a.output,
           "kernel_slug": kernel_id, "accelerator": a.accelerator if a.gpu else None}
    body = TEMPLATE.read_text().replace("__CONFIG__", json.dumps(cfg))
    (kdir / "bootstrap.py").write_text(body)
    meta = {"id": kernel_id, "title": slug, "code_file": "bootstrap.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": bool(a.gpu), "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": a.source, "model_sources": []}
    if a.gpu and a.accelerator:
        meta["machine_shape"] = a.accelerator
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    cmd = ["kaggle", "kernels", "push", "-p", str(kdir)]
    if a.timeout:
        cmd += ["-t", str(a.timeout)]
    if a.gpu and a.accelerator:
        cmd += ["--accelerator", a.accelerator]
    out = run(cmd)
    print(out)
    info = {"name": a.name, "kernel_id": kernel_id, "commit": sha, "command": a.cmd, "gpu": bool(a.gpu),
            "accelerator_requested": a.accelerator if a.gpu else None, "sources": a.source,
            "outputs": a.output, "purpose": a.purpose,
            "pushed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "push_output": out}
    (RUNS / a.name / "launch.json").write_text(json.dumps(info, indent=1))
    print(f"launched {kernel_id} at commit {sha[:10]}")
    if a.wait:
        wait(a.name, a.poll)
        fetch(a.name)


def status(name: str) -> str:
    info = json.loads((RUNS / name / "launch.json").read_text())
    out = run(["kaggle", "kernels", "status", info["kernel_id"]], check=False)
    return out


def wait(name: str, poll: int = 30) -> str:
    t0 = time.time()
    while True:
        s = status(name)
        low = s.lower()
        print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)
        if any(f"status.{t}" in low or f'"{t}' in low or low.endswith(t) for t in TERMINAL):
            return s
        time.sleep(poll)


def _git_blob_sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _tree_blobs(commit: str, prefix: str = "results") -> dict[str, str]:
    """{repo-relative path: blob sha} of every file under prefix at the kernel's checkout commit."""
    out = run(["git", "ls-tree", "-r", commit, "--", prefix], check=False)
    blobs = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    return blobs


def _sync_results(dl: Path, commit: str, root: Path = ROOT, blobs: dict | None = None) -> dict:
    """Copy ONLY result files the kernel created or changed relative to its checkout commit. A file whose bytes
    equal the committed blob at `commit` is a checked-out input, not an output, and is never copied (it could
    overwrite newer local files)."""
    copied, unchanged, too_big = [], [], []
    src = dl / "results"
    if not src.exists():
        return {"copied": copied, "skipped_unchanged_from_checkout": unchanged, "skipped_size": too_big}
    blobs = _tree_blobs(commit) if blobs is None else blobs
    for f in sorted(src.rglob("*")):
        if not (f.is_file() and f.suffix in SYNC_EXT):
            continue
        rel = str(Path("results") / f.relative_to(src))
        if f.stat().st_size > SYNC_MAX_BYTES:
            too_big.append(rel)
            continue
        if blobs.get(rel) == _git_blob_sha(f.read_bytes()):
            unchanged.append(rel)
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        copied.append(rel)
    return {"copied": copied, "skipped_unchanged_from_checkout": unchanged, "skipped_size": too_big}


def _ledger_append(row: dict) -> float:
    header = ("| date (UTC) | kernel | accelerator (nvidia-smi) | wall time (h) | GPU-hours | status | commit | purpose |\n"
              "|---|---|---|---|---|---|---|---|\n")
    text = LEDGER.read_text() if LEDGER.exists() else ""
    if header not in text:
        text = ("# GPU ledger\n\nOne row per Kaggle kernel run (appended by `scripts/kaggle_run.py fetch`). "
                "Wall time is the bootstrap's own wall clock (excludes Kaggle queue/startup). "
                "GPU-hours = wall hours for GPU kernels, 0 for CPU kernels.\n\n"
                "RUNNING_TOTAL\n\n" + header)
    line = (f"| {row['date']} | {row['kernel']} | {row['acc']} | {row['wall_h']:.3f} | {row['gpu_h']:.3f} | "
            f"{row['status']} | {row['commit'][:10]} | {row['purpose']} |\n")
    text = re.sub(r"\*\*Running total.*\*\*\n?", "RUNNING_TOTAL\n", text)
    text = text.rstrip("\n") + "\n" + line
    total = sum(float(m.group(1)) for m in re.finditer(r"^\|[^|]*\|[^|]*\|[^|]*\|[^|]*\| ([0-9.]+) \|", text, re.M))
    text = text.replace("RUNNING_TOTAL", f"**Running total GPU-hours: {total:.3f}**")
    LEDGER.write_text(text)
    return total


def fetch(name: str) -> None:
    info = json.loads((RUNS / name / "launch.json").read_text())
    dl = RUNS / name / "output"
    dl.mkdir(parents=True, exist_ok=True)
    print(run(["kaggle", "kernels", "output", info["kernel_id"], "-p", str(dl), "-o"], check=False)[-2000:])
    st = status(name)
    rm_path = dl / "run_meta.json"
    rm = json.loads(rm_path.read_text()) if rm_path.exists() else {}
    gpus = rm.get("gpu_names") or []
    acc = f"{len(gpus)}x {gpus[0]}" if gpus else ("none (CPU)" if not info["gpu"] else "unknown")
    wall_h = rm.get("wall_seconds", 0) / 3600
    if not rm:
        # download failed (e.g. network) or the kernel produced no run_meta: never write a ledger row or sync
        raise SystemExit(f"fetch incomplete for {info['kernel_id']}: no run_meta.json in {dl} (status: {st[:200]}); "
                         "nothing synced, ledger unchanged - re-run fetch")
    ok = rm.get("exit_code") == 0 and not rm.get("error")
    state = ("ok" if ok else f"FAILED(rc={rm.get('exit_code')},err={rm.get('error')})") if rm else f"no run_meta: {st}"
    sync = _sync_results(dl, info["commit"])
    copied = sync["copied"]
    total = _ledger_append({"date": rm.get("start_utc", info["pushed_utc"]), "kernel": info["kernel_id"],
                            "acc": acc, "wall_h": wall_h, "gpu_h": wall_h if gpus else 0.0, "status": state,
                            "commit": info["commit"], "purpose": info.get("purpose") or ""})
    info.update({"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "final_status": st,
                 "accelerator_observed": acc, "wall_seconds": rm.get("wall_seconds"), "synced": copied,
                 "skipped_unchanged_from_checkout": len(sync["skipped_unchanged_from_checkout"]),
                 "skipped_size": sync["skipped_size"]})
    (RUNS / name / "launch.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"kernel": info["kernel_id"], "status": st, "state": state, "accelerator": acc,
                      "wall_s": rm.get("wall_seconds"), "synced": copied,
                      "skipped_unchanged_from_checkout": len(sync["skipped_unchanged_from_checkout"]),
                      "skipped_size": sync["skipped_size"], "ledger_total_gpu_h": round(total, 3)},
                     indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="action", required=True)
    l = sp.add_parser("launch")
    l.add_argument("name")
    l.add_argument("--cmd", required=True, help="shell command run from the repo root on Kaggle")
    g = l.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", default=False)
    g.add_argument("--cpu", dest="gpu", action="store_false")
    l.add_argument("--accelerator", default="NvidiaTeslaT4",
                   help="machine_shape sent with --gpu ('' = server default); verified via nvidia-smi")
    l.add_argument("--source", action="append", default=[], help="kernel_sources OWNER/SLUG (repeatable)")
    l.add_argument("--output", action="append", default=[], help="extra repo-relative path copied to output")
    l.add_argument("--timeout", type=int)
    l.add_argument("--purpose", default="")
    l.add_argument("--wait", action="store_true")
    l.add_argument("--poll", type=int, default=30)
    for n in ("status", "fetch", "wait"):
        s = sp.add_parser(n)
        s.add_argument("name")
    a = p.parse_args()
    if a.action == "launch":
        if a.accelerator == "":
            a.accelerator = None
        launch(a)
    elif a.action == "status":
        print(status(a.name))
    elif a.action == "wait":
        wait(a.name)
        fetch(a.name)
    else:
        fetch(a.name)


if __name__ == "__main__":
    sys.exit(main())
