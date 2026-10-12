"""Где лежат результаты и кэши. Всё локально, в results/.

    results/
      index.csv                     одна строка на прогон — сводка по всем экспериментам
      runs/<run_id>/                всё про один прогон
        config.yaml                 полный конфиг
        manifest.json               git, окружение, хеши протокола и шаблона
        decisions/<split>__<method>.jsonl
        probe.csv  sweep.csv  selection.json  test.json  report.md
      cache/<model_slug>/           переиспользуется между прогонами одной модели
        decisions.jsonl             ключ = модель + метод + промпт + режим решения
        acts/<prompt_hash>.npz      пулинг активаций по (слой, окно)
      frozen/<freeze_id>.json       замороженный метод перед test
      test_lock.jsonl               журнал открытий test (одно на freeze_id)
      protocol.json                 хеши HYPOTHESES.md / METRICS.md при первом прогоне

Кэш — главное, что экономит время на Mac: повторный запуск ноутбука с мелкой
правкой пересчитывает только то, что реально изменилось.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    """Корень репозитория: ближайшая папка вверх с pyproject.toml и docs/."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "docs").is_dir():
            return cand
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "pyproject.toml").exists() and (cand / "docs").is_dir():
            return cand
    raise FileNotFoundError("repository root not found (no pyproject.toml + docs/)")


def sha(obj: Any, n: int = 16) -> str:
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
    elif isinstance(obj, str):
        b = obj.encode()
    else:
        b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:n]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80] or "x"


def file_sha(path: Path) -> str | None:
    return sha(path.read_bytes()) if path.exists() else None


# --------------------------------------------------------------------------
# Git и окружение
# --------------------------------------------------------------------------


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    status = run("status", "--porcelain", "--", "src", "docs", "notebooks", "configs")
    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:20],
    }


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "torch", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


# --------------------------------------------------------------------------
# Хранилище
# --------------------------------------------------------------------------


class Store:
    def __init__(self, root: Path | None = None, results_dir: Path | None = None):
        self.root = Path(root) if root else find_repo_root()
        env = os.environ.get("STF_RESULTS_DIR")
        self.results = Path(results_dir) if results_dir else (Path(env) if env else self.root / "results")
        self.results.mkdir(parents=True, exist_ok=True)

    # --- прогон --------------------------------------------------------
    def new_run_dir(self, name: str, cfg_hash: str) -> tuple[str, Path]:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{slug(name)}_{cfg_hash[:6]}"
        d = self.results / "runs" / run_id
        d.mkdir(parents=True, exist_ok=False)
        (d / "decisions").mkdir()
        return run_id, d

    # --- кэши модели ---------------------------------------------------
    def cache_dir(self, model_key: str) -> Path:
        d = self.results / "cache" / slug(model_key)
        (d / "acts").mkdir(parents=True, exist_ok=True)
        return d

    # --- протокол ------------------------------------------------------
    def protocol_hashes(self) -> dict[str, Any]:
        """Хеши зафиксированных документов сейчас и при первом прогоне."""
        cur = {
            "HYPOTHESES.md": file_sha(self.root / "docs" / "HYPOTHESES.md"),
            "METRICS.md": file_sha(self.root / "docs" / "METRICS.md"),
        }
        path = self.results / "protocol.json"
        if not path.exists():
            path.write_text(json.dumps({"first_seen": time.strftime("%Y-%m-%d %H:%M"),
                                        "hashes": cur}, indent=2))
        first = json.loads(path.read_text())["hashes"]
        changed = [k for k in cur if first.get(k) and cur[k] != first[k]]
        return {"current": cur, "first": first, "changed_since_first_run": changed}

    # --- сводная таблица ----------------------------------------------
    INDEX_FIELDS = ("run_id", "time", "name", "hypothesis", "model", "mode", "stages",
                    "base_success", "steer_success", "paired_diff", "ci", "spec_drop",
                    "H1", "H2", "H3", "git", "dirty", "notebook")

    def update_index(self, row: dict[str, Any]) -> None:
        path = self.results / "index.csv"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open() as f:
                rows = [r for r in csv.DictReader(f) if r.get("run_id") != row.get("run_id")]
        rows.append({k: row.get(k, "") for k in self.INDEX_FIELDS})
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.INDEX_FIELDS)
            w.writeheader()
            w.writerows(rows)

    def read_index(self):
        import pandas as pd

        path = self.results / "index.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=self.INDEX_FIELDS)

    # --- заморозка и test-lock ----------------------------------------
    def save_frozen(self, payload: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
        """freeze_id = хеш payload (только то, что определяет метод). meta — справочно."""
        fid = sha(payload, 12)
        d = self.results / "frozen"
        d.mkdir(exist_ok=True)
        (d / f"{fid}.json").write_text(json.dumps({**payload, "freeze_id": fid, "meta": meta or {}},
                                                  indent=2, ensure_ascii=False, default=str))
        return fid

    def test_openings(self, freeze_id: str) -> list[dict[str, Any]]:
        path = self.results / "test_lock.jsonl"
        if not path.exists():
            return []
        return [r for r in map(json.loads, path.read_text().splitlines())
                if r.get("freeze_id") == freeze_id]

    def record_test_opening(self, freeze_id: str, run_id: str, reason: str | None) -> None:
        with (self.results / "test_lock.jsonl").open("a") as f:
            f.write(json.dumps({"freeze_id": freeze_id, "run_id": run_id,
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "reopen_reason": reason}, ensure_ascii=False) + "\n")


class DecisionCache:
    """JSONL-кэш решений: ключ -> запись. Дописывается, читается целиком."""

    def __init__(self, path: Path):
        self.path = path
        self._d: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._d[r["key"]] = r
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        r = self._d.get(key)
        if r is None:
            self.misses += 1
        else:
            self.hits += 1
        return r

    def put(self, rec: dict[str, Any]) -> None:
        self._d[rec["key"]] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


class ActivationCache:
    """npz на промпт: массивы f"L{layer}|{window}" -> [hidden]."""

    def __init__(self, d: Path):
        self.d = d

    @staticmethod
    def key(layer: int, window: str) -> str:
        return f"L{layer}|{window}"

    def load(self, prompt_hash: str) -> dict[str, np.ndarray]:
        p = self.d / f"{prompt_hash}.npz"
        if not p.exists():
            return {}
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(self, prompt_hash: str, arrays: dict[str, np.ndarray]) -> None:
        merged = {**self.load(prompt_hash), **arrays}
        tmp = self.d / f"{prompt_hash}.tmp.npz"
        np.savez(tmp, **merged)
        os.replace(tmp, self.d / f"{prompt_hash}.npz")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
