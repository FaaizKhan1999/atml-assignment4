import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts/kaggle_run.py")
kr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kr)


def test_ledger_running_total(tmp_path, monkeypatch):
    led = tmp_path / "gpu_ledger.md"
    # fresh ledger: runner creates header + running total itself
    monkeypatch.setattr(kr, "LEDGER", led)
    row = dict(date="d", kernel="u/k", acc="2x Tesla T4", wall_h=0.5, gpu_h=0.5, status="ok", commit="a" * 40,
               purpose="p")
    kr._ledger_append(row)
    total = kr._ledger_append({**row, "gpu_h": 0.25, "acc": "none (CPU)"})
    assert abs(total - 0.75) < 1e-9
    text = led.read_text()
    assert "**Running total GPU-hours: 0.750**" in text and text.count("Running total") == 1


def test_bootstrap_template_renders():
    body = (ROOT / "kaggle/bootstrap_template.py").read_text().replace(
        "__CONFIG__", json.dumps({"repo_url": "x", "commit": "y", "command": "echo 'hi'", "outputs": [],
                                  "kernel_slug": "u/s", "accelerator": None}))
    compile(body, "bootstrap.py", "exec")


def test_sync_copies_only_kernel_outputs(tmp_path):
    dl = tmp_path / "dl"
    root = tmp_path / "repo"
    (dl / "results/task1/old").mkdir(parents=True)
    (dl / "results/task2/new").mkdir(parents=True)
    checked_out = b'{"v": "old committed"}'
    (dl / "results/task1/old/summary.json").write_bytes(checked_out)      # unchanged checkout copy
    (dl / "results/task1/old/changed.json").write_bytes(b'{"v": "kernel"}')  # kernel rewrote a tracked file
    (dl / "results/task2/new/out.jsonl").write_bytes(b"{}\n")              # new output
    (dl / "results/task2/new/notes.txt").write_bytes(b"x")                  # extension not synced
    local = root / "results/task1/old/summary.json"
    local.parent.mkdir(parents=True)
    local.write_bytes(b'{"v": "NEWER local"}')
    blobs = {"results/task1/old/summary.json": kr._git_blob_sha(checked_out),
             "results/task1/old/changed.json": kr._git_blob_sha(b'{"v": "committed"}')}
    res = kr._sync_results(dl, "deadbeef", root=root, blobs=blobs)
    assert sorted(res["copied"]) == ["results/task1/old/changed.json", "results/task2/new/out.jsonl"]
    assert res["skipped_unchanged_from_checkout"] == ["results/task1/old/summary.json"]
    assert local.read_bytes() == b'{"v": "NEWER local"}'  # the newer local file is never overwritten


def test_git_blob_sha_matches_git(tmp_path):
    import subprocess

    f = tmp_path / "a.json"
    f.write_bytes(b'{"a": 1}\n')
    ref = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True).stdout.strip()
    assert kr._git_blob_sha(f.read_bytes()) == ref
