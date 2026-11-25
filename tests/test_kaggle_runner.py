import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts/kaggle_run.py")
kr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kr)


def test_ledger_running_total(tmp_path, monkeypatch):
    led = tmp_path / "gpu_ledger.md"
    led.write_text((ROOT / "docs/gpu_ledger.md").read_text())
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
