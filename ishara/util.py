"""Shared helpers: config loading, hashing, jsonl io, manifests."""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    root = Path(cfg.get("project_dir") or ".")
    if not root.is_absolute():
        root = (Path(path).resolve().parent.parent / root).resolve()
    cfg["_root"] = str(root)
    return cfg


def rpath(cfg, p):
    """Resolve a path from the config against the project folder."""
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(*parts):
    """Deterministic integer from any parts, independent of PYTHONHASHSEED."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest(cfg, extra=None):
    """Hashes of every input that can change a result."""
    m = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "config_sha256": sha256_file(cfg["_config_path"]),
    }
    for key, p in [
        ("prompts_sha256", cfg["data"]["prompts_file"]),
        ("split_sha256", cfg["data"]["split_file"]),
        ("rubric_sha256", cfg["rubric"]["file"]),
    ]:
        fp = rpath(cfg, p)
        m[key] = sha256_file(fp) if fp.exists() else None
    try:
        import torch
        import transformers
        m["torch"] = torch.__version__
        m["transformers"] = transformers.__version__
        m["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import peft
        m["peft"] = peft.__version__
    except ImportError:
        pass
    m["model"] = {k: cfg["model"].get(k) for k in ("hf_id", "revision", "dtype", "load_in_4bit")}
    m["generation"] = cfg["generation"]
    if extra:
        m.update(extra)
    return m


def out_dir(cfg):
    d = rpath(cfg, cfg["out_dir"]) / cfg["model"]["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes")
