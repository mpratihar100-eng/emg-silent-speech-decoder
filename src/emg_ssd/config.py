from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherit = cfg.pop("inherit", None)
    if inherit:
        inherit_path = Path(inherit)
        if inherit_path.is_absolute():
            parent = inherit_path
        else:
            local = (path.parent / inherit_path).resolve()
            cwd_rel = (Path.cwd() / inherit_path).resolve()
            parent = local if local.exists() else cwd_rel
        base = load_config(parent)
        cfg = _deep_update(base, cfg)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ["data_root", "internal_root", "checkpoints", "preprocessed_root", "manifests_root"]:
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            p.mkdir(parents=True, exist_ok=True)
