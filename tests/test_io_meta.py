import json

from atml4.common.io import load_jsonl_result, load_result, save_result
from atml4.common.meta import collect_meta, model_revision


def test_json_roundtrip(tmp_path):
    meta = collect_meta(seed=6304, models=["base"], command="python x.py --a 1")
    payload = {"ppl": 12.5, "n": 50000, "nan": float("nan")}
    p = save_result(tmp_path / "r.json", payload, meta)
    m, pl = load_result(p)
    assert m == json.loads(json.dumps(meta))
    assert pl["ppl"] == 12.5 and pl["n"] == 50000
    assert m["command"] == "python x.py --a 1" and m["seed"] == 6304
    assert m["models"][0]["revision"] == model_revision("Qwen/Qwen2.5-1.5B")[1]
    for k in ("git", "versions", "cuda", "sdpa", "timestamp_utc", "hostname"):
        assert k in m
    assert not list(tmp_path.glob(".*.tmp"))


def test_jsonl_roundtrip_gz(tmp_path):
    meta = collect_meta(seed=1, command=["python", "y.py"])
    rows = [{"id": i, "score": i / 3} for i in range(5)]
    p = save_result(tmp_path / "r.jsonl.gz", iter(rows), meta)
    m, back = load_jsonl_result(p)
    assert back == rows and m["command"] == "python y.py"


def test_unpinned_model_rejected():
    import pytest
    with pytest.raises(KeyError):
        model_revision("gpt2")
    with pytest.raises(ValueError):
        save_result("/tmp/never.json", {}, {})
