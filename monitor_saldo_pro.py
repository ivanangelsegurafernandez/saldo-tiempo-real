#!/usr/bin/env python3
"""
Monitor de saldo REAL DERIV (PySide6 + PyQtGraph).

Dependencias:
  pip install pandas numpy pyqtgraph PySide6

Lectura de datos (prioridad REAL):
1) saldo_real_live_history.jsonl (ruta compartida)
2) saldo_real_live.json (snapshot maestro)
3) saldo_real_series.csv (fallback real)
4) LOG_SALDOS / *.log / *.txt (observado)
5) registro_enriquecido_fulll*.csv (auxiliar)
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    import numpy as np
except Exception as e:
    print(f"[MONITOR][ERROR] Dependencia faltante o inválida: numpy ({e})")
    raise
try:
    import pandas as pd
except Exception as e:
    print(f"[MONITOR][ERROR] Dependencia faltante o inválida: pandas ({e})")
    raise
try:
    import pyqtgraph as pg
except Exception as e:
    print(f"[MONITOR][ERROR] Dependencia faltante o inválida: pyqtgraph ({e})")
    raise
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:
    print(f"[MONITOR][ERROR] Dependencia faltante o inválida: PySide6 ({e})")
    raise

# ------------------------ Config ------------------------
CUENTA_OBJETIVO = "REAL"  # REAL | DEMO | ALL
REFRESH_SEGUNDOS = 5
REFRESH_SEGUNDOS_MINIMIZADO = 15
VENTANA_HORAS = 9
VENTANA_MINUTOS = 60
VENTANA_DIAS = 14
FULLSCREEN_INICIAL = False
LOG_SALDOS = "LOG_SALDOS"
CSV_PATTERN = "registro_enriquecido_fulll*.csv"
SALDO_LIVE_FILE = "saldo_real_live.json"
SALDO_LIVE_HISTORY_FILE = "saldo_real_live_history.jsonl"
SALDO_SERIES_CSV_FILE = "saldo_real_series.csv"
DISPLAY_TIMEZONE = "America/Lima"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SALDO_LIVE_SHARED_PATH = os.path.abspath(
    os.getenv("SALDO_LIVE_SHARED_PATH", os.path.join(os.path.expanduser("~"), SALDO_LIVE_FILE))
)
SALDO_LIVE_HISTORY_SHARED_PATH = os.path.abspath(
    os.getenv(
        "SALDO_LIVE_HISTORY_SHARED_PATH",
        os.path.join(os.path.dirname(SALDO_LIVE_SHARED_PATH), SALDO_LIVE_HISTORY_FILE),
    )
)
def resolver_ruta_saldo_series() -> str:
    custom = os.getenv("SALDO_SERIES_CSV_PATH", "").strip()
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.abspath(os.path.join(SCRIPT_DIR, SALDO_SERIES_CSV_FILE))

SALDO_SERIES_CSV_PATH = resolver_ruta_saldo_series()
SALDO_LIVE_PATH = os.getenv("SALDO_LIVE_PATH", "").strip()

MONITOR_VERSION = "v2026.03.31-r1"
MONITOR_BUILD_ID = "MONITOR_SALDO_PRO_REAL_SERIES_GUARD"
MIN_POINTS_FOR_LINE = 2
SHOW_LAST_MARKER = True
SHOW_EXTREME_MARKERS = False
GLOW_ENABLED = False
PLOT_DOWNSAMPLE_THRESHOLD = 1800
RANGE_EPSILON = 1e-4
Y_SCALE_MODE = os.getenv("Y_SCALE_MODE", "auto").strip().lower()  # capital | manual | auto
Y_AXIS_MIN_USD = float(os.getenv("Y_AXIS_MIN_USD", "0"))
Y_AXIS_MAX_USD = float(os.getenv("Y_AXIS_MAX_USD", "300"))
Y_AUTO_SPAN_USD = float(os.getenv("Y_AUTO_SPAN_USD", "120"))
CAPITAL_BASE_USD = float(os.getenv("CAPITAL_BASE_USD", "0") or "0")
MIN_X_SPAN_SECONDS = 20.0
WORKER_WARN_SECONDS = 2.5
DEGRADED_ERRORS_THRESHOLD = 4
DEGRADED_RENDER_MS = 220.0
DEGRADED_WORKER_MS = 1500.0
RULE_FLATLINE_S = 90.0
RULE_DROP_FAST_PCT = 8.0
RULE_REBOUND_PCT = 4.0
RULE_STALE_S = 90.0
RULE_VOL_HIGH_THR = 1.25
RULE_NARROW_RANGE_USD = 1.0
RULE_ALERT_COOLDOWN_S = 25.0
OBSERVED_MAX_BYTES = 180_000
ESTIMATED_REFRESH_S = 60.0
AUX_REFRESH_EVERY_N = 12
def _safe_display_tz():
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(DISPLAY_TIMEZONE)
    except Exception:
        return timezone.utc


DISPLAY_TZ = _safe_display_tz()



def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone(DISPLAY_TZ)


def _fmt_money(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:,.2f} USD"


def _fmt_local_ts(ts_obj) -> str:
    if ts_obj is None:
        return "--"
    try:
        if isinstance(ts_obj, pd.Timestamp):
            if ts_obj.tzinfo is None:
                ts_obj = ts_obj.tz_localize("UTC")
            return ts_obj.to_pydatetime().astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        if isinstance(ts_obj, datetime):
            if ts_obj.tzinfo is None:
                ts_obj = ts_obj.replace(tzinfo=timezone.utc)
            return ts_obj.astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "--"
    return "--"


def _sanitize_series_for_plot(s: pd.DataFrame) -> pd.DataFrame:
    if s is None or s.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
    d = s.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d["equity"] = pd.to_numeric(d["equity"], errors="coerce")
    d = d.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    d = d.drop_duplicates(subset=["timestamp"], keep="last")
    return d.reset_index(drop=True)


def _sample_window_series(
    series: pd.DataFrame,
    cutoff: datetime,
    target_points: int,
) -> pd.DataFrame:
    base = _sanitize_series_for_plot(series[series["timestamp"] >= cutoff].copy()) if not series.empty else pd.DataFrame(columns=["timestamp", "equity"])
    if base.empty or len(base) <= max(4, target_points):
        return base
    try:
        t_sec = (base["timestamp"].astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)
        span = int(max(1, t_sec[-1] - t_sec[0]))
        bucket_s = max(1, span // max(1, int(target_points)))
        bucket_id = (t_sec - t_sec[0]) // bucket_s
        df = base.reset_index(drop=True).copy()
        df["bucket_id"] = bucket_id
        keep_idx = set()
        for _, g in df.groupby("bucket_id", sort=True):
            if g.empty:
                continue
            keep_idx.add(int(g.index[0]))   # first
            keep_idx.add(int(g.index[-1]))  # last
            keep_idx.add(int(g["equity"].idxmin()))  # min
            keep_idx.add(int(g["equity"].idxmax()))  # max
        sampled = df.loc[sorted(keep_idx), ["timestamp", "equity"]]
        sampled = sampled.drop_duplicates(subset=["timestamp", "equity"], keep="last").sort_values("timestamp")
        return sampled.reset_index(drop=True)
    except Exception:
        return base


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace("USD", "").replace("$", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return default


class SmartDateAxis(pg.DateAxisItem):
    def tickStrings(self, values, scale, spacing):
        if not values:
            return []
        finite_vals = [v for v in values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
        if not finite_vals:
            return ["" for _ in values]
        span = max(finite_vals) - min(finite_vals)
        out = []
        last_label = None
        for v in values:
            try:
                if not isinstance(v, (int, float, np.floating)) or not np.isfinite(v):
                    out.append("")
                    continue
                # Rango seguro para datetime (aprox. 0001..9999 en segundos epoch)
                if v < -62135596800 or v > 253402300799:
                    out.append("")
                    continue
                dt_utc = datetime.fromtimestamp(float(v), tz=timezone.utc)
                dt_local = dt_utc.astimezone(DISPLAY_TZ)
                if span <= 15 * 60:
                    label = dt_local.strftime("%H:%M:%S")
                elif span <= 6 * 3600:
                    label = dt_local.strftime("%H:%M")
                elif span <= 48 * 3600:
                    label = dt_local.strftime("%d-%m %H:%M")
                else:
                    label = dt_local.strftime("%d-%m")
                if label == last_label:
                    out.append("")
                else:
                    out.append(label)
                    last_label = label
            except Exception:
                out.append("")
        return out


class MoneyAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{v:,.2f}" for v in values]


@dataclass
class Snapshot:
    source: str
    saldo_actual: Optional[float]
    last_update: Optional[datetime]
    now: datetime
    series_real: pd.DataFrame
    series_main: pd.DataFrame
    series_minutes: pd.DataFrame
    series_hours: pd.DataFrame
    series_days: pd.DataFrame
    series_est: pd.DataFrame
    warnings: List[str]
    view: str


class SnapshotWorker(QtCore.QObject):
    finished = QtCore.Signal(object, float, float, str)
    failed = QtCore.Signal(str)

    def __init__(self, engine: "DataEngine", view: str):
        super().__init__()
        self.engine = engine
        self.view = view

    @QtCore.Slot()
    def run(self):
        t0 = time.perf_counter()
        try:
            snap = self.engine.build_snapshot(self.view)
            t1 = time.perf_counter()
            self.finished.emit(snap, (t1 - t0) * 1000.0, time.time(), self.view)
        except Exception:
            self.failed.emit(traceback.format_exc())


class DataEngine:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._cache: Dict[str, Tuple[Tuple, object]] = {}
        self._history_last_seen: Optional[Tuple[str, int, int]] = None
        self._agg_cache_state: Dict[str, object] = {}
        self._history_inc: Dict[str, object] = {"path": None, "offset": 0, "df": pd.DataFrame(columns=["timestamp", "equity"])}
        self._series_inc: Dict[str, object] = {"path": None, "offset": 0, "df": pd.DataFrame(columns=["timestamp", "equity"])}
        self._estimated_last_ts: float = 0.0
        self._estimated_last_view: str = ""
        self._tick_counter: int = 0

    @staticmethod
    def _sig(paths: List[Path]) -> Tuple:
        sig = []
        for p in paths:
            try:
                st = p.stat()
                sig.append((str(p), st.st_mtime_ns, st.st_size))
            except Exception:
                sig.append((str(p), None, None))
        return tuple(sig)

    def _master_live_candidates(self) -> List[Path]:
        cands: List[Path] = [Path(SALDO_LIVE_SHARED_PATH).expanduser()]
        if SALDO_LIVE_PATH:
            custom = Path(SALDO_LIVE_PATH).expanduser()
            cands.append(custom / SALDO_LIVE_FILE if custom.is_dir() else custom)
        out: List[Path] = []
        seen = set()
        for p in cands:
            k = str(p)
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    def _master_history_candidates(self) -> List[Path]:
        cands: List[Path] = [Path(SALDO_LIVE_HISTORY_SHARED_PATH).expanduser()]
        for p in self._master_live_candidates():
            cands.append(p.parent / SALDO_LIVE_HISTORY_FILE)
        out: List[Path] = []
        seen = set()
        for p in cands:
            k = str(p)
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    def _master_series_candidates(self) -> List[Path]:
        cands: List[Path] = [Path(SALDO_SERIES_CSV_PATH).expanduser()]
        for p in self._master_live_candidates():
            cands.append(p.parent / SALDO_SERIES_CSV_FILE)
        out: List[Path] = []
        seen = set()
        for p in cands:
            k = str(p)
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    def _cached(self, key: str, sig: Tuple):
        old = self._cache.get(key)
        if old and old[0] == sig:
            return old[1]
        return None

    def _store_cache(self, key: str, sig: Tuple, value):
        self._cache[key] = (sig, value)
        return value

    def _read_master_live(self) -> Tuple[Optional[Tuple[float, datetime]], Optional[str], Optional[Path]]:
        paths = self._master_live_candidates()
        sig = self._sig(paths)
        cached = self._cached("live", sig)
        if cached is not None:
            return cached

        found_any = False
        for p in paths:
            if not p.exists():
                continue
            found_any = True
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                v = _safe_float(obj.get("saldo_real"), default=np.nan)
                if not np.isfinite(v):
                    msg = f"{SALDO_LIVE_FILE} inválido en {'ruta compartida' if str(p)==SALDO_LIVE_SHARED_PATH else p}"
                    return self._store_cache("live", sig, (None, msg, p))
                ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", utc=True)
                if pd.isna(ts):
                    ts = pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)
                return self._store_cache("live", sig, ((float(v), ts.to_pydatetime()), None, p))
            except Exception:
                msg = f"{SALDO_LIVE_FILE} inválido en {'ruta compartida' if str(p)==SALDO_LIVE_SHARED_PATH else p}"
                return self._store_cache("live", sig, (None, msg, p))

        msg = f"{SALDO_LIVE_FILE} no encontrado en ruta compartida: {SALDO_LIVE_SHARED_PATH}"
        if found_any:
            msg = f"saldo real del maestro no disponible ({SALDO_LIVE_FILE})"
        return self._store_cache("live", sig, (None, msg, None))

    def _read_master_history(self) -> Tuple[pd.DataFrame, Optional[str], Optional[Path], Optional[str]]:
        paths = self._master_history_candidates()
        sig = self._sig(paths)
        cached = self._cached("hist", sig)
        if cached is not None:
            return cached

        for p in paths:
            if not p.exists():
                continue
            try:
                rows = []
                st = p.stat()
                inc = self._history_inc
                use_inc = str(inc.get("path")) == str(p) and int(inc.get("offset", 0) or 0) <= int(st.st_size)
                if use_inc and int(inc.get("offset", 0) or 0) > 0:
                    with p.open("r", encoding="utf-8", errors="ignore") as fh:
                        fh.seek(int(inc.get("offset", 0) or 0))
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            v = _safe_float(obj.get("saldo_real"), default=np.nan)
                            if not np.isfinite(v):
                                continue
                            ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", utc=True)
                            if pd.isna(ts):
                                continue
                            rows.append((ts.to_pydatetime(), float(v)))
                    base_df = inc.get("df", pd.DataFrame(columns=["timestamp", "equity"]))
                    add_df = pd.DataFrame(rows, columns=["timestamp", "equity"]) if rows else pd.DataFrame(columns=["timestamp", "equity"])
                    d = pd.concat([base_df, add_df], ignore_index=True)
                else:
                    with p.open("r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            v = _safe_float(obj.get("saldo_real"), default=np.nan)
                            if not np.isfinite(v):
                                continue
                            ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", utc=True)
                            if pd.isna(ts):
                                continue
                            rows.append((ts.to_pydatetime(), float(v)))
                    d = pd.DataFrame(rows, columns=["timestamp", "equity"]) if rows else pd.DataFrame(columns=["timestamp", "equity"])
                if not d.empty:
                    d = d.sort_values("timestamp").drop_duplicates(subset=["timestamp", "equity"], keep="last")
                    self._history_inc = {"path": str(p), "offset": int(st.st_size), "df": d}
                    growth_msg = self._check_history_growth(p, len(d))
                    return self._store_cache("hist", sig, (d, None, p, growth_msg))
            except Exception:
                msg = f"{SALDO_LIVE_HISTORY_FILE} inválido en {'ruta compartida' if str(p)==SALDO_LIVE_HISTORY_SHARED_PATH else p}"
                return self._store_cache("hist", sig, (pd.DataFrame(columns=["timestamp", "equity"]), msg, p, None))

        return self._store_cache(
            "hist",
            sig,
            (pd.DataFrame(columns=["timestamp", "equity"]), f"Sin histórico real: no se encontró {SALDO_LIVE_HISTORY_FILE} en ruta compartida: {SALDO_LIVE_HISTORY_SHARED_PATH}", None, None),
        )

    def _read_saldo_series_csv(self) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
        paths = self._master_series_candidates()
        sig = self._sig(paths)
        cached = self._cached("series_csv", sig)
        if cached is not None:
            return cached
        for p in paths:
            if not p.exists():
                continue
            try:
                st = p.stat()
                inc = self._series_inc
                use_inc = str(inc.get("path")) == str(p) and int(inc.get("offset", 0) or 0) <= int(st.st_size)
                if use_inc and int(inc.get("offset", 0) or 0) > 0:
                    with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                        fh.seek(int(inc.get("offset", 0) or 0))
                        chunk = fh.read()
                    if not chunk.strip():
                        d = inc.get("df", pd.DataFrame(columns=["timestamp", "equity"]))
                        return self._store_cache("series_csv", sig, (d, None, p))
                    header = "ts_utc,ts_lima,epoch,saldo_real,status,source,event_type\n"
                    reader = pd.read_csv(io.StringIO(header + chunk))
                    base_df = inc.get("df", pd.DataFrame(columns=["timestamp", "equity"]))
                else:
                    with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                        reader = pd.read_csv(fh)
                    base_df = pd.DataFrame(columns=["timestamp", "equity"])
                if "saldo_real" not in reader.columns:
                    continue
                ts = pd.to_datetime(reader.get("ts_utc"), errors="coerce", utc=True)
                if ts.isna().all() and "epoch" in reader.columns:
                    ts = pd.to_datetime(pd.to_numeric(reader["epoch"], errors="coerce"), unit="s", errors="coerce", utc=True)
                vals = pd.to_numeric(reader["saldo_real"], errors="coerce")
                d = pd.DataFrame({"timestamp": ts, "equity": vals}).dropna(subset=["timestamp", "equity"])
                d = pd.concat([base_df, d], ignore_index=True)
                if d.empty:
                    continue
                d = d.sort_values("timestamp").drop_duplicates(subset=["timestamp", "equity"], keep="last")
                self._series_inc = {"path": str(p), "offset": int(st.st_size), "df": d}
                return self._store_cache("series_csv", sig, (d, None, p))
            except Exception:
                return self._store_cache("series_csv", sig, (pd.DataFrame(columns=["timestamp", "equity"]), f"{SALDO_SERIES_CSV_FILE} inválido", p))
        return self._store_cache("series_csv", sig, (pd.DataFrame(columns=["timestamp", "equity"]), None, None))

    def _check_history_growth(self, path: Path, valid_rows: int) -> Optional[str]:
        try:
            st = path.stat()
            current = (str(path), st.st_size, valid_rows)
            previous = self._history_last_seen
            self._history_last_seen = current
            if previous is None or previous[0] != current[0]:
                return None
            if previous[1] == current[1] and previous[2] == current[2]:
                return (
                    f"Histórico sin crecimiento: {SALDO_LIVE_HISTORY_FILE} no cambió "
                    f"(size={current[1]} bytes, muestras={current[2]})"
                )
        except Exception:
            return None
        return None

    def _parse_observed(self, view: str) -> pd.DataFrame:
        if view == "ALL":
            real = self._parse_observed("REAL")
            demo = self._parse_observed("DEMO")
            if real.empty:
                return demo
            if demo.empty:
                return real
            rr = real.rename(columns={"equity": "real"}).sort_values("timestamp")
            dd = demo.rename(columns={"equity": "demo"}).sort_values("timestamp")
            z = pd.merge_asof(rr, dd, on="timestamp", direction="nearest", tolerance=pd.Timedelta("12h"))
            z["real"] = z["real"].ffill().fillna(0.0)
            z["demo"] = z["demo"].ffill().fillna(0.0)
            z["equity"] = z["real"] + z["demo"]
            return z[["timestamp", "equity"]]

        files = [self.base_dir / LOG_SALDOS]
        files.extend(sorted(self.base_dir.glob("*.log")))
        files.extend(sorted(self.base_dir.glob("*.txt")))
        sig = self._sig(files)
        key = f"obs:{view}"
        cached = self._cached(key, sig)
        if cached is not None:
            return cached

        patterns = [
            re.compile(r"SALDO\s+EN\s+CUENTA\s+REAL\s+DERIV\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE),
            re.compile(r"Saldo\s+cuenta\s+REAL(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE),
        ] if view == "REAL" else [
            re.compile(r"Saldo\s+cuenta\s+DEMO(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE)
        ]

        rows: List[Tuple[datetime, float]] = []
        for p in files:
            if not p.exists() or not p.is_file():
                continue
            try:
                with p.open("rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    sz = fh.tell()
                    fh.seek(max(0, sz - OBSERVED_MAX_BYTES), os.SEEK_SET)
                    raw = fh.read().decode("utf-8", errors="ignore")
                lines = raw.splitlines()[-3000:]
            except Exception:
                continue
            base = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) - timedelta(seconds=len(lines))
            for i, line in enumerate(lines):
                match = None
                for pat in patterns:
                    match = pat.search(line)
                    if match:
                        break
                if not match:
                    continue
                v = _safe_float(match.group(1), default=np.nan)
                if not np.isfinite(v):
                    continue
                ts = pd.to_datetime(line, errors="coerce", utc=True)
                ts = (base + timedelta(seconds=i)) if pd.isna(ts) else ts.to_pydatetime()
                rows.append((ts, float(v)))

        if not rows:
            return self._store_cache(key, sig, pd.DataFrame(columns=["timestamp", "equity"]))
        d = pd.DataFrame(rows, columns=["timestamp", "equity"]).sort_values("timestamp")
        d = d.drop_duplicates(subset=["timestamp", "equity"], keep="last")
        return self._store_cache(key, sig, d)

    def _build_estimated(self, view: str) -> pd.DataFrame:
        files = [Path(p) for p in sorted(glob.glob(str(self.base_dir / CSV_PATTERN)))]
        sig = self._sig(files)
        key = f"est:{view}"
        now = time.time()
        if (
            self._estimated_last_view == view
            and (now - float(self._estimated_last_ts or 0.0)) < ESTIMATED_REFRESH_S
            and key in self._cache
        ):
            old = self._cache.get(key)
            if old is not None:
                return old[1]
        cached = self._cached(key, sig)
        if cached is not None:
            return cached
        if not files:
            return self._store_cache(key, sig, pd.DataFrame(columns=["timestamp", "equity"]))

        dfs = []
        for p in files:
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    dfs.append(pd.read_csv(p, encoding=enc, low_memory=False))
                    break
                except Exception:
                    pass
        if not dfs:
            return self._store_cache(key, sig, pd.DataFrame(columns=["timestamp", "equity"]))

        d = pd.concat(dfs, ignore_index=True)
        if "ganancia_perdida" not in d.columns:
            return self._store_cache(key, sig, pd.DataFrame(columns=["timestamp", "equity"]))
        ts = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
        if "fecha" in d.columns:
            ts = ts.fillna(pd.to_datetime(d["fecha"], errors="coerce", utc=True))
        if "ts" in d.columns:
            ts = ts.fillna(pd.to_datetime(d["ts"], errors="coerce", utc=True))
        if "epoch" in d.columns:
            ts = ts.fillna(pd.to_datetime(pd.to_numeric(d["epoch"], errors="coerce"), unit="s", errors="coerce", utc=True))
        d["timestamp"] = ts
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
        if "trade_status" in d.columns:
            d = d[d["trade_status"].astype(str).str.upper().str.strip() == "CERRADO"]
        if "cuenta" in d.columns and view in ("REAL", "DEMO"):
            d = d[d["cuenta"].astype(str).str.upper().str.contains(view, na=False)]
        pnl = pd.to_numeric(d["ganancia_perdida"], errors="coerce").fillna(0.0)
        base = 1000.0 if view == "REAL" else (10000.0 if view == "DEMO" else 11000.0)
        d["equity"] = base + pnl.cumsum()
        self._estimated_last_ts = now
        self._estimated_last_view = view
        return self._store_cache(key, sig, d[["timestamp", "equity"]])

    def build_snapshot(self, view: str) -> Snapshot:
        now = _now()
        self._tick_counter += 1
        warnings: List[str] = []

        master, master_msg, live_path_used = self._read_master_live() if view == "REAL" else (None, None, None)
        hist, hist_msg, hist_path_used, hist_growth_msg = self._read_master_history() if view == "REAL" else (pd.DataFrame(columns=["timestamp", "equity"]), None, None, None)
        series_csv, series_msg, series_path_used = self._read_saldo_series_csv() if view == "REAL" else (pd.DataFrame(columns=["timestamp", "equity"]), None, None)
        master_ok = bool(view == "REAL" and (not hist.empty or master is not None or not series_csv.empty))
        aux_tick = (self._tick_counter % max(1, AUX_REFRESH_EVERY_N)) == 0
        observed = pd.DataFrame(columns=["timestamp", "equity"])
        estimated = pd.DataFrame(columns=["timestamp", "equity"])
        if (not master_ok) or aux_tick or view in ("DEMO", "ALL"):
            observed = self._parse_observed(view)
        if (not master_ok) or view in ("DEMO", "ALL") or (aux_tick and view == "REAL"):
            estimated = self._build_estimated(view)

        if master_msg and view == "REAL":
            warnings.append(master_msg)
        if hist_msg and view == "REAL":
            warnings.append(hist_msg)

        if view == "REAL":
            warnings.append(f"Monitor {MONITOR_VERSION} · id={MONITOR_BUILD_ID}")
            warnings.append(f"Ruta snapshot real: {live_path_used if live_path_used else SALDO_LIVE_SHARED_PATH}")
            warnings.append(f"Ruta histórico real: {hist_path_used if hist_path_used else SALDO_LIVE_HISTORY_SHARED_PATH}")
            if series_msg:
                warnings.append(series_msg)
            if series_path_used:
                warnings.append(f"Ruta serie CSV real: {series_path_used}")
            warnings.append(
                f"Estado snapshot: {'OK' if live_path_used and Path(live_path_used).exists() else 'NO ENCONTRADO'} | "
                f"Estado histórico: {'OK' if hist_path_used and Path(hist_path_used).exists() else 'NO ENCONTRADO'}"
            )
            valid_rows = int(len(hist))
            last_hist_ts = _fmt_local_ts(hist["timestamp"].iloc[-1]) if valid_rows else "--"
            warnings.append(
                f"Histórico válido: {valid_rows} muestra(s) | última marca válida: {last_hist_ts}"
            )
            if hist_growth_msg:
                warnings.append(hist_growth_msg)

        source = "SIN DATOS REALES"
        saldo_actual: Optional[float] = None
        last_update: Optional[datetime] = None
        real_series = pd.DataFrame(columns=["timestamp", "equity"])

        if view == "REAL" and not hist.empty:
            source = "MAESTRO"
            real_series = hist.copy()
            if master is not None:
                mv, mts = master
                saldo_actual = mv
                last_update = mts
                real_series = pd.concat([real_series, pd.DataFrame([{"timestamp": mts, "equity": mv}])], ignore_index=True)
                real_series = real_series.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            else:
                saldo_actual = float(real_series["equity"].iloc[-1])
                last_update = real_series["timestamp"].iloc[-1]
        elif master is not None:
            mv, mts = master
            source = "MAESTRO"
            saldo_actual = mv
            last_update = mts
            real_series = pd.DataFrame([{"timestamp": mts, "equity": mv}])
        elif view == "REAL" and not series_csv.empty:
            source = "SERIE_CSV"
            saldo_actual = float(series_csv["equity"].iloc[-1])
            last_update = series_csv["timestamp"].iloc[-1]
            real_series = series_csv
        elif not observed.empty:
            source = "OBSERVADO"
            saldo_actual = float(observed["equity"].iloc[-1])
            last_update = observed["timestamp"].iloc[-1]
            real_series = observed
        else:
            live_real = self._parse_observed("REAL") if view == "REAL" else self._parse_observed(view)
            if not live_real.empty:
                source = "LIVE"
                saldo_actual = float(live_real["equity"].iloc[-1])
                last_update = live_real["timestamp"].iloc[-1]
                real_series = pd.DataFrame([live_real.iloc[-1]])
            else:
                warnings.append("saldo real del maestro no disponible")

        real_points = int(len(real_series))
        if real_points == 0:
            warnings.append("Sin datos para graficar: 0 muestras reales válidas")
            warnings.append(f"Sin histórico real: no se encontró {SALDO_LIVE_HISTORY_FILE}")
        elif real_points == 1:
            warnings.append("Histórico insuficiente: solo 1 muestra real válida")
            warnings.append("No se puede trazar línea: se requieren al menos 2 puntos")
        if source == "SIN DATOS REALES" and not estimated.empty:
            warnings.append("Estimado CSV disponible solo como auxiliar")

        mcut = now - timedelta(minutes=VENTANA_MINUTOS)
        hcut = now - timedelta(hours=VENTANA_HORAS)
        dcut = now - timedelta(days=VENTANA_DIAS)

        smain = _sample_window_series(real_series, dcut, target_points=900)
        smin = _sample_window_series(real_series, mcut, target_points=420)
        shrs = _sample_window_series(real_series, hcut, target_points=520)
        sday = _sample_window_series(real_series, dcut, target_points=360)
        if not sday.empty and len(sday) < MIN_POINTS_FOR_LINE:
            sday = sday.tail(min(120, len(sday))).copy()
            warnings.append("Panel DÍAS en fallback crudo: histórico diario aún insuficiente")

        return Snapshot(
            source=source,
            saldo_actual=saldo_actual,
            last_update=last_update,
            now=now,
            series_real=real_series,
            series_main=smain,
            series_minutes=smin,
            series_hours=shrs,
            series_days=sday,
            series_est=estimated,
            warnings=warnings,
            view=view,
        )


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: DataEngine):
        super().__init__()
        self.engine = engine
        self.view = CUENTA_OBJETIVO if CUENTA_OBJETIVO in ("REAL", "DEMO", "ALL") else "REAL"
        self.paused = False
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker_busy = False
        self.refresh_in_progress = False
        self.pending_refresh_skip_count = 0
        self._render_signature = None
        self._last_good_snapshot: Optional[Snapshot] = None
        self._last_metrics: Dict[str, object] = {}
        self._rule_last_emit: Dict[str, float] = {}
        self.snapshot_build_ms = 0.0
        self.render_ms = 0.0
        self.last_worker_started_ts = 0.0
        self.last_worker_finished_ts = 0.0
        self.skipped_refresh_count = 0
        self.last_error_ts = 0.0
        self.consecutive_errors = 0
        self.degraded_mode_active = False
        self.degraded_activations = 0
        self.setWindowTitle(f"Monitor Saldo Real Deriv {MONITOR_VERSION}")
        self.resize(1600, 900)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        header = QtWidgets.QFrame(); header.setObjectName("HeaderCard")
        hl = QtWidgets.QVBoxLayout(header); hl.setContentsMargins(16, 14, 16, 14); hl.setSpacing(12)

        top = QtWidgets.QHBoxLayout(); top.setSpacing(10)
        self.lbl_title = QtWidgets.QLabel(f"SALDO REAL DERIV ACTUAL · {MONITOR_VERSION}"); self.lbl_title.setObjectName("Title")
        self.lbl_source = QtWidgets.QLabel("FUENTE: --"); self.lbl_source.setObjectName("BadgeWarn")
        top.addWidget(self.lbl_title, 1); top.addWidget(self.lbl_source, 0)
        hl.addLayout(top)

        self.lbl_big = QtWidgets.QLabel("--"); self.lbl_big.setObjectName("Big")
        self.lbl_big.setAlignment(QtCore.Qt.AlignCenter); self.lbl_big.setMinimumHeight(118)
        self.lbl_big.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        hl.addWidget(self.lbl_big)

        meta = QtWidgets.QHBoxLayout(); meta.setSpacing(10)
        self.lbl_refresh = QtWidgets.QLabel("REFRESCO: ACTIVO"); self.lbl_refresh.setObjectName("MetaBox")
        self.lbl_scale = QtWidgets.QLabel("ESCALA Y: --"); self.lbl_scale.setObjectName("MetaBox")
        self.lbl_delta = QtWidgets.QLabel("Δ VIS: --"); self.lbl_delta.setObjectName("MetaBox")
        self.lbl_samples = QtWidgets.QLabel("MUESTRAS: --"); self.lbl_samples.setObjectName("MetaBox")
        self.lbl_tz = QtWidgets.QLabel(f"TZ: {DISPLAY_TIMEZONE}"); self.lbl_tz.setObjectName("MetaBox")
        self.lbl_now = QtWidgets.QLabel("HORA LOCAL: --"); self.lbl_now.setObjectName("MetaNow")
        self.lbl_last = QtWidgets.QLabel("ÚLTIMA ACT: --"); self.lbl_last.setObjectName("MetaLast")
        self.lbl_state = QtWidgets.QLabel("STATE: IDLE"); self.lbl_state.setObjectName("MetaBox")
        self.lbl_csv_health = QtWidgets.QLabel("CSV REAL: --"); self.lbl_csv_health.setObjectName("MetaBox")
        self.lbl_rule = QtWidgets.QLabel("RULE: --"); self.lbl_rule.setObjectName("MetaBox")
        meta.addWidget(self.lbl_refresh)
        meta.addWidget(self.lbl_scale)
        meta.addWidget(self.lbl_delta)
        meta.addWidget(self.lbl_samples)
        meta.addWidget(self.lbl_tz)
        meta.addWidget(self.lbl_state)
        meta.addWidget(self.lbl_csv_health)
        meta.addWidget(self.lbl_rule)
        meta.addWidget(self.lbl_now, 1)
        meta.addWidget(self.lbl_last)
        hl.addLayout(meta)

        self.lbl_live_metrics = QtWidgets.QLabel("LIVE: --"); self.lbl_live_metrics.setObjectName("Help")
        hl.addWidget(self.lbl_live_metrics)
        tools = QtWidgets.QHBoxLayout(); tools.setSpacing(8)
        self.btn_real = QtWidgets.QPushButton("REAL")
        self.btn_demo = QtWidgets.QPushButton("DEMO")
        self.btn_all = QtWidgets.QPushButton("ALL")
        self.btn_pause = QtWidgets.QPushButton("PAUSA")
        self.btn_reset = QtWidgets.QPushButton("RESET VISTA")
        self.btn_export = QtWidgets.QPushButton("EXPORTAR CSV")
        self.btn_measure = QtWidgets.QPushButton("REGLA")
        self.btn_markers = QtWidgets.QPushButton("MARCADORES")
        self.btn_scale = QtWidgets.QPushButton(f"ESCALA: {Y_SCALE_MODE.upper()}")
        for b in (self.btn_real, self.btn_demo, self.btn_all, self.btn_pause, self.btn_reset, self.btn_export, self.btn_measure, self.btn_markers, self.btn_scale):
            tools.addWidget(b)
        hl.addLayout(tools)
        root.addWidget(header)

        self.graphics = pg.GraphicsLayoutWidget(); root.addWidget(self.graphics, 1)
        self.p_main = self.graphics.addPlot(row=0, col=0, colspan=2, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})
        self.p_min = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})
        self.p_hour = self.graphics.addPlot(row=1, col=1, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})
        self.p_day = self.graphics.addPlot(row=2, col=0, colspan=2, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})

        self._style_plot(self.p_main, "EQUITY CURVE PRINCIPAL · dinero vs tiempo")
        self._style_plot(self.p_min, "MINUTOS · detalle")
        self._style_plot(self.p_hour, "HORAS · comportamiento")
        self._style_plot(self.p_day, "DÍAS · tendencia")
        self._measurement_points: List[Tuple[float, float]] = []
        self._measurement_active = False
        self._cross_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#587ca8aa", width=1))
        self._cross_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#587ca8aa", width=1))
        self.p_main.addItem(self._cross_v, ignoreBounds=True)
        self.p_main.addItem(self._cross_h, ignoreBounds=True)
        self._cross_proxy = pg.SignalProxy(self.p_main.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved)
        self.p_main.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.plot_states = {
            "main": self._init_plot_state(self.p_main, "#5df2ff", "#d9fbff", VENTANA_DIAS * 86400),
            "min": self._init_plot_state(self.p_min, "#3fe9ff", "#c6f7ff", VENTANA_MINUTOS * 60),
            "hour": self._init_plot_state(self.p_hour, "#7aa6ff", "#dae3ff", VENTANA_HORAS * 3600),
            "day": self._init_plot_state(self.p_day, "#7ff0b9", "#dcffe9", VENTANA_DIAS * 86400),
        }

        self.lbl_warn = QtWidgets.QLabel(""); self.lbl_warn.setObjectName("Warn"); root.addWidget(self.lbl_warn)
        self.lbl_help = QtWidgets.QLabel(f"Teclas: [1]REAL [2]DEMO [3]ALL [F]Fullscreen [P]Pausa [R]Reset [Q]Salir · {MONITOR_VERSION} · {MONITOR_BUILD_ID}")
        self.lbl_help.setObjectName("Help"); root.addWidget(self.lbl_help)
        self.btn_real.clicked.connect(lambda: self._set_view("REAL"))
        self.btn_demo.clicked.connect(lambda: self._set_view("DEMO"))
        self.btn_all.clicked.connect(lambda: self._set_view("ALL"))
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_reset.clicked.connect(self._reset_view)
        self.btn_export.clicked.connect(self._export_visible_csv)
        self.btn_measure.clicked.connect(self._toggle_measurement)
        self.btn_markers.clicked.connect(self._toggle_markers)
        self.btn_scale.clicked.connect(self._cycle_scale_mode)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b0f14; color: #d9e2f2; }
            #HeaderCard { background: #0f1622; border: 1px solid #203047; border-radius: 14px; }
            #Title { font-size: 18px; color: #b9d3ff; font-weight: 780; }
            #Big { font-size: 86px; color: #ecfff3; font-weight: 900; padding: 8px 0 10px 0; }
            #MetaBox { font-size: 14px; color: #bdd6ff; background: #101e31; border: 1px solid #263e5f; border-radius: 10px; padding: 8px 12px; }
            #MetaNow { font-size: 16px; color: #ecf6ff; background: #153153; border: 1px solid #2f5e92; border-radius: 10px; padding: 8px 12px; font-weight: 780; }
            #MetaLast { font-size: 14px; color: #d9e8ff; background: #12263d; border: 1px solid #2c4b72; border-radius: 10px; padding: 8px 12px; }
            #BadgeMaster { font-size: 13px; color: #041d13; background: #72f8b1; border: 1px solid #9dffd0; border-radius: 13px; padding: 4px 11px; font-weight: 900; }
            #BadgeObserved { font-size: 13px; color: #02222b; background: #67efff; border: 1px solid #8ff6ff; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeLive { font-size: 13px; color: #0b2a1f; background: #9ef7d8; border: 1px solid #c0ffe8; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeNeutral { font-size: 13px; color: #d8e7ff; background: #23364f; border: 1px solid #3d5c81; border-radius: 13px; padding: 4px 11px; font-weight: 800; }
            #BadgeWarn { font-size: 13px; color: #3d2a00; background: #ffd67f; border: 1px solid #ffe09e; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeBad { font-size: 13px; color: #390000; background: #ff9c9c; border: 1px solid #ffb8b8; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #Warn { font-size: 11px; color: #ffc374; font-weight: 520; }
            #Help { font-size: 9px; color: #6b84a6; }
            """
        )
        pg.setConfigOptions(antialias=False, background="#0b0f14", foreground="#d9e2f2")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(REFRESH_SEGUNDOS * 1000))

        if FULLSCREEN_INICIAL:
            self.showMaximized()
        self.refresh(force=True)

    def _style_plot(self, plot: pg.PlotItem, title: str):
        plot.setTitle(f"<span style='color:#cfe2ff;font-size:12pt;font-weight:680'>{title}</span>")
        plot.setLabel("left", "USD")
        plot.showGrid(x=True, y=True, alpha=0.05)
        for ax in (plot.getAxis("left"), plot.getAxis("bottom")):
            ax.setTextPen(pg.mkPen("#b9d0ee")); ax.setPen(pg.mkPen("#35506f"))
        plot.addLegend(offset=(5, 5), labelTextSize="7pt")
        y0, y1, _ = self._resolve_y_range(None)
        plot.setYRange(y0, y1, padding=0.0)
        plot.setLimits(yMin=y0, yMax=y1)

    def _snapshot_signature(self, snap: Snapshot) -> Tuple:
        s = _sanitize_series_for_plot(snap.series_main)
        last_ts = s["timestamp"].iloc[-1] if not s.empty else None
        last_eq = float(s["equity"].iloc[-1]) if not s.empty else None
        return (
            str(snap.source),
            None if snap.saldo_actual is None else round(float(snap.saldo_actual), 8),
            snap.last_update.isoformat() if isinstance(snap.last_update, datetime) else None,
            int(len(s)),
            last_ts.isoformat() if last_ts is not None else None,
            None if last_eq is None else round(last_eq, 8),
            int(len(snap.series_minutes)),
            int(len(snap.series_hours)),
            int(len(snap.series_days)),
        )

    def _compute_live_metrics(self, visible: pd.DataFrame, snap: Snapshot) -> Dict[str, object]:
        m: Dict[str, object] = {"text": "LIVE: --", "rule": "--"}
        if visible.empty:
            return m
        y = visible["equity"].to_numpy(dtype=float)
        x = (visible["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        peak = float(np.nanmax(y)); floor = float(np.nanmin(y))
        last_v = float(y[-1]); range_v = peak - floor
        dd_usd = last_v - peak
        dd_pct = (dd_usd / peak * 100.0) if abs(peak) > 1e-12 else 0.0
        speed = np.nan
        slope = np.nan
        vol = np.nan
        age_change = np.nan
        if len(y) >= 2:
            dtm = max(1e-9, (x[-1] - x[-2]) / 60.0)
            speed = (y[-1] - y[-2]) / dtm
            last_change_idx = np.where(np.abs(np.diff(y)) > 1e-9)[0]
            if len(last_change_idx) > 0:
                age_change = max(0.0, x[-1] - x[last_change_idx[-1] + 1])
            dx = (x - x[0]) / 60.0
            if len(y) >= 3 and float(dx[-1]) > 0:
                slope = float(np.polyfit(dx, y, 1)[0])
            diffs = np.diff(y)
            if len(diffs) >= 2:
                vol = float(np.nanstd(diffs))
        dist_peak_usd = peak - last_v
        dist_peak_pct = (dist_peak_usd / peak * 100.0) if abs(peak) > 1e-12 else 0.0
        m.update(
            {
                "peak": peak, "floor": floor, "range": range_v,
                "drawdown_usd": dd_usd, "drawdown_pct": dd_pct,
                "speed": speed, "slope": slope, "vol": vol,
                "dist_peak_usd": dist_peak_usd, "dist_peak_pct": dist_peak_pct,
                "age_last_change_s": age_change,
                "text": (
                    f"LIVE | peak={peak:,.2f} floor={floor:,.2f} dd={dd_usd:,.2f} ({dd_pct:+.2f}%) "
                    f"rng={range_v:,.2f} spd={speed if np.isfinite(speed) else float('nan'):+.3f}/min "
                    f"slope={slope if np.isfinite(slope) else float('nan'):+.3f} vol={vol if np.isfinite(vol) else float('nan'):.3f}"
                ),
            }
        )
        return m

    def _evaluate_live_rules(self, metrics: Dict[str, object], snap: Snapshot) -> str:
        now = time.time()
        alerts: List[str] = []
        age_change = metrics.get("age_last_change_s", np.nan)
        drawdown_pct = abs(float(metrics.get("drawdown_pct", 0.0) or 0.0))
        speed = float(metrics.get("speed", np.nan))
        vol = float(metrics.get("vol", np.nan))
        range_v = float(metrics.get("range", 0.0) or 0.0)
        if np.isfinite(age_change) and age_change >= RULE_FLATLINE_S:
            alerts.append("FLATLINE")
        if np.isfinite(speed) and speed <= -(RULE_DROP_FAST_PCT / 100.0):
            alerts.append("DROP_FAST")
        if np.isfinite(speed) and speed >= (RULE_REBOUND_PCT / 100.0):
            alerts.append("REBOUND")
        if drawdown_pct <= 0.05 and range_v > RULE_NARROW_RANGE_USD:
            alerts.append("NEW_PEAK")
        if snap.last_update and (snap.now - snap.last_update).total_seconds() > RULE_STALE_S:
            alerts.append("STALE_SOURCE")
        if np.isfinite(vol) and vol >= RULE_VOL_HIGH_THR:
            alerts.append("HIGH_VOL")
        if range_v <= RULE_NARROW_RANGE_USD:
            alerts.append("NARROW_RANGE")
        visible = []
        for a in alerts:
            last_emit = float(self._rule_last_emit.get(a, 0.0) or 0.0)
            if (now - last_emit) >= RULE_ALERT_COOLDOWN_S:
                self._rule_last_emit[a] = now
                visible.append(a)
        if not visible and alerts:
            visible = alerts[:1]
        return ", ".join(visible[:3]) if visible else "--"

    def _start_snapshot_worker(self):
        if self._worker_busy:
            self.skipped_refresh_count += 1
            self.pending_refresh_skip_count += 1
            return
        self._worker_busy = True
        self.refresh_in_progress = True
        self.last_worker_started_ts = time.time()
        thread = QtCore.QThread(self)
        worker = SnapshotWorker(self.engine, self.view)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_snapshot_ready)
        worker.failed.connect(self._on_snapshot_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker_thread = thread
        thread.start()

    def _resolve_y_range(self, y: Optional[np.ndarray]) -> Tuple[float, float, str]:
        if Y_SCALE_MODE == "manual":
            ymin = float(min(Y_AXIS_MIN_USD, Y_AXIS_MAX_USD))
            ymax = float(max(Y_AXIS_MIN_USD, Y_AXIS_MAX_USD))
            if ymax - ymin < 0.01:
                ymax = ymin + 1.0
            return ymin, ymax, f"manual · min={ymin:,.2f} max={ymax:,.2f}"
        if Y_SCALE_MODE == "capital":
            if CAPITAL_BASE_USD > 0:
                base = float(CAPITAL_BASE_USD)
            elif y is not None and len(y) > 0:
                base = float(max(0.01, y[-1]))
            else:
                base = 10.0
            data_span = 0.0
            if y is not None and len(y) > 1:
                data_span = float(max(0.0, np.nanmax(y) - np.nanmin(y)))
            band = max(2.0, base * 0.35, data_span * 2.5)
            y0 = max(0.0, base - band * 0.5)
            y1 = y0 + band
            return y0, y1, f"capital · base={base:,.2f} span={band:,.2f}"
        if y is None or len(y) == 0:
            span = float(max(10.0, Y_AUTO_SPAN_USD))
            return 0.0, span, f"auto · span={span:,.2f}"
        ymin_data = float(np.nanmin(y))
        ymax_data = float(np.nanmax(y))
        span = float(max(0.0, ymax_data - ymin_data))
        if span < 1e-9:
            pad = max(0.5, abs(ymin_data) * 0.10)
        else:
            pad = max(0.25, span * 0.08)
        y0 = max(0.0, ymin_data - pad)
        y1 = ymax_data + pad
        if y1 - y0 < 0.5:
            y1 = y0 + 0.5
        return y0, y1, f"auto · data=[{ymin_data:,.2f},{ymax_data:,.2f}]"

    def _init_plot_state(self, plot: pg.PlotItem, color: str, endpoint: str, canonical_window_s: int) -> Dict[str, object]:
        glow_width = 2.0 if GLOW_ENABLED else 0.1
        glow = plot.plot([], [], pen=pg.mkPen(color + "44", width=glow_width), name=None)
        line = plot.plot([], [], pen=pg.mkPen(color, width=2.0), name="Equity")
        try:
            line.setClipToView(True)
            line.setDownsampling(auto=True, method="peak")
            line.setSkipFiniteCheck(True)
        except Exception:
            pass
        last = plot.plot([], [], pen=None, symbol="o", symbolSize=5, symbolBrush=endpoint, name=None)
        vmax = plot.plot([], [], pen=None, symbol="o", symbolSize=4, symbolBrush="#ffd36b99", name=None)
        vmin = plot.plot([], [], pen=None, symbol="o", symbolSize=4, symbolBrush="#ff8f8f99", name=None)
        txt = pg.TextItem(text="", color="#9ec2ff", anchor=(0, 1))
        plot.addItem(txt)
        return {"plot": plot, "glow": glow, "line": line, "last": last, "max": vmax, "min": vmin, "text": txt, "canonical_window_s": int(canonical_window_s), "last_range": (None, None), "last_point": None}

    def _set_x_range_visible(self, plot: pg.PlotItem, x: np.ndarray, canonical_window_s: int):
        if len(x) == 0:
            return
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        span = max(0.0, xmax - xmin)
        if len(x) == 1:
            pad = max(MIN_X_SPAN_SECONDS, canonical_window_s * 0.02)
            plot.setXRange(xmin - pad, xmax + pad, padding=0.0)
            return
        if span < float(canonical_window_s):
            pad = max(MIN_X_SPAN_SECONDS, span * 0.08, canonical_window_s * 0.01)
            plot.setXRange(xmin - pad, xmax + pad, padding=0.0)
        else:
            plot.setXRange(xmax - float(canonical_window_s), xmax, padding=0.0)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k == QtCore.Qt.Key_1:
            self.view = "REAL"; self.refresh(force=True)
        elif k == QtCore.Qt.Key_2:
            self.view = "DEMO"; self.refresh(force=True)
        elif k == QtCore.Qt.Key_3:
            self.view = "ALL"; self.refresh(force=True)
        elif k == QtCore.Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == QtCore.Qt.Key_P:
            self.paused = not self.paused; self.refresh(force=True)
        elif k == QtCore.Qt.Key_R:
            self.p_main.enableAutoRange(); self.p_min.enableAutoRange(); self.p_hour.enableAutoRange(); self.p_day.enableAutoRange()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(ev)

    def _set_view(self, view: str):
        self.view = view
        self.refresh(force=True)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.refresh(force=True)

    def _reset_view(self):
        self.p_main.enableAutoRange(); self.p_min.enableAutoRange(); self.p_hour.enableAutoRange(); self.p_day.enableAutoRange()

    def _on_mouse_moved(self, evt):
        try:
            pos = evt[0]
            if self.p_main.sceneBoundingRect().contains(pos):
                mouse_point = self.p_main.vb.mapSceneToView(pos)
                self._cross_v.setPos(mouse_point.x())
                self._cross_h.setPos(mouse_point.y())
                self.lbl_help.setText(
                    f"Cursor · t={datetime.fromtimestamp(float(mouse_point.x()), tz=DISPLAY_TZ).strftime('%H:%M:%S')} | saldo={float(mouse_point.y()):,.2f}"
                )
        except Exception:
            pass

    def _on_plot_clicked(self, evt):
        if not self._measurement_active:
            return
        try:
            pos = evt.scenePos()
            if not self.p_main.sceneBoundingRect().contains(pos):
                return
            p = self.p_main.vb.mapSceneToView(pos)
            self._measurement_points.append((float(p.x()), float(p.y())))
            self._measurement_points = self._measurement_points[-2:]
            if len(self._measurement_points) == 2:
                (x1, y1), (x2, y2) = self._measurement_points
                dt_min = max(1e-9, abs(x2 - x1) / 60.0)
                d_usd = y2 - y1
                d_pct = (d_usd / y1 * 100.0) if abs(y1) > 1e-12 else 0.0
                slope = d_usd / dt_min
                self.lbl_rule.setText(f"RULE: ΔUSD={d_usd:+,.2f} Δ%={d_pct:+.2f} Δt={dt_min:,.2f}m slope={slope:+,.3f}/m")
        except Exception:
            pass

    def _toggle_measurement(self):
        self._measurement_active = not self._measurement_active
        if not self._measurement_active:
            self._measurement_points.clear()
            self.lbl_rule.setText("RULE: --")
        self.btn_measure.setText("REGLA ON" if self._measurement_active else "REGLA")

    def _toggle_markers(self):
        global SHOW_EXTREME_MARKERS
        SHOW_EXTREME_MARKERS = not SHOW_EXTREME_MARKERS
        self.btn_markers.setText("MARCADORES ON" if SHOW_EXTREME_MARKERS else "MARCADORES")

    def _cycle_scale_mode(self):
        global Y_SCALE_MODE
        modes = ["auto", "capital", "manual"]
        try:
            idx = modes.index(Y_SCALE_MODE)
        except Exception:
            idx = 0
        Y_SCALE_MODE = modes[(idx + 1) % len(modes)]
        self.btn_scale.setText(f"ESCALA: {Y_SCALE_MODE.upper()}")
        self.refresh(force=True)

    def _export_visible_csv(self):
        try:
            if not self._last_good_snapshot:
                self.lbl_warn.setText("⚠ Export: sin snapshot visible")
                return
            vis = _sanitize_series_for_plot(self._last_good_snapshot.series_main)
            if vis.empty:
                self.lbl_warn.setText("⚠ Export: sin datos visibles")
                return
            out = vis.copy()
            out["source"] = str(self._last_good_snapshot.source)
            out["window_tag"] = "main_visible"
            out.to_csv("monitor_visible_export.csv", index=False)
            self.lbl_warn.setText(f"✅ Exportado monitor_visible_export.csv ({len(out)} filas)")
        except Exception as e:
            self.lbl_warn.setText(f"⚠ Export error: {e}")

    def changeEvent(self, ev: QtCore.QEvent):
        if ev.type() == QtCore.QEvent.WindowStateChange:
            interval = REFRESH_SEGUNDOS_MINIMIZADO if self.isMinimized() else REFRESH_SEGUNDOS
            self.timer.setInterval(int(interval * 1000))
        super().changeEvent(ev)

    def _update_plot_state(self, state: Dict[str, object], s: pd.DataFrame) -> Tuple[str, float, float]:
        plot = state["plot"]
        glow = state["glow"]; line = state["line"]; last = state["last"]; vmax = state["max"]; vmin = state["min"]; txt = state["text"]
        s = _sanitize_series_for_plot(s)
        if s.empty:
            glow.setData([], [])
            line.setData([], [])
            last.setData([], [])
            vmax.setData([], [])
            vmin.setData([], [])
            txt.setText("Sin puntos")
            y0, y1, scale_info = self._resolve_y_range(None)
            lr0, lr1 = state.get("last_range", (None, None))
            if lr0 is None or abs(float(lr0) - y0) > RANGE_EPSILON or abs(float(lr1) - y1) > RANGE_EPSILON:
                plot.setYRange(y0, y1, padding=0.0)
                state["last_range"] = (y0, y1)
            return "sin datos", y0, y1

        x = (s["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = s["equity"].to_numpy(dtype=float)

        if len(x) >= MIN_POINTS_FOR_LINE:
            if GLOW_ENABLED:
                glow.setData(x, y)
            else:
                glow.setData([], [])
            line.setData(x, y)
            txt.setText("")
        else:
            glow.setData([], [])
            line.setData([], [])
            txt.setText("1 punto: esperando más histórico")

        marker_size = 8 if len(x) == 1 else 4
        if SHOW_LAST_MARKER:
            last.setData([x[-1]], [y[-1]], symbolSize=marker_size)
        else:
            last.setData([], [])
        if SHOW_EXTREME_MARKERS and len(x) >= 8:
            imax = int(np.argmax(y)); imin = int(np.argmin(y))
            vmax.setData([x[imax]], [y[imax]])
            vmin.setData([x[imin]], [y[imin]])
        else:
            vmax.setData([], [])
            vmin.setData([], [])
        y0, y1, scale_info = self._resolve_y_range(y)
        lr0, lr1 = state.get("last_range", (None, None))
        if lr0 is None or abs(float(lr0) - y0) > RANGE_EPSILON or abs(float(lr1) - y1) > RANGE_EPSILON:
            plot.setYRange(y0, y1, padding=0.0)
            state["last_range"] = (y0, y1)
        self._set_x_range_visible(plot, x, int(state["canonical_window_s"]))
        return scale_info, y0, y1

    def refresh(self, force: bool = False):
        self.lbl_now.setText(f"HORA LOCAL: {_now().strftime('%H:%M:%S %Z')}")
        if self.paused and not force:
            self.lbl_state.setText("STATE: PAUSED")
            return
        if self.isMinimized() and not force:
            self.lbl_state.setText("STATE: IDLE")
            return
        if self._worker_busy:
            self.skipped_refresh_count += 1
            self.lbl_state.setText("STATE: LIVE (busy)")
            return
        target_interval = REFRESH_SEGUNDOS_MINIMIZADO if self.degraded_mode_active else REFRESH_SEGUNDOS
        self.timer.setInterval(int(target_interval * 1000))
        self._start_snapshot_worker()

    @QtCore.Slot(object, float, float, str)
    def _on_snapshot_ready(self, snap: Snapshot, build_ms: float, finished_ts: float, _view: str):
        self._worker_busy = False
        self.refresh_in_progress = False
        self.snapshot_build_ms = float(build_ms)
        self.last_worker_finished_ts = float(finished_ts)
        self._last_good_snapshot = snap
        t0 = time.perf_counter()
        signature = None
        refresh_state = "PAUSADO" if self.paused else "ACTIVO"
        self.lbl_refresh.setText(f"REFRESCO: {refresh_state}")
        try:
            signature = self._snapshot_signature(snap)
            last = snap.last_update.astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z") if snap.last_update else "--"
            self.lbl_last.setText(f"ÚLTIMA ACT: {last}")
            if signature == self._render_signature:
                self.lbl_state.setText("STATE: IDLE")
                return
            self._render_signature = signature
        except Exception:
            snap.warnings.append("render prep error")
            traceback.print_exc()

        main_scale = "--"
        main_y0, main_y1 = 0.0, 0.0
        visible = pd.DataFrame(columns=["timestamp", "equity"])

        # FASE 2 — Header crítico
        try:
            self.lbl_big.setText(_fmt_money(snap.saldo_actual))
            src = snap.source.upper().strip()
            self.lbl_source.setText(f"FUENTE: {src}")
            if src == "MAESTRO":
                self.lbl_source.setObjectName("BadgeMaster"); self.lbl_big.setStyleSheet("color:#72f8b1;")
            elif src in ("OBSERVADO", "MAESTRO_HIST", "SERIE_CSV"):
                self.lbl_source.setObjectName("BadgeObserved"); self.lbl_big.setStyleSheet("color:#67efff;")
            elif src == "LIVE":
                self.lbl_source.setObjectName("BadgeLive"); self.lbl_big.setStyleSheet("color:#9ef7d8;")
            elif src == "STALE":
                self.lbl_source.setObjectName("BadgeWarn"); self.lbl_big.setStyleSheet("color:#ffe9b8;")
            elif src in ("ESTIMADO", "SIN DATOS REALES"):
                self.lbl_source.setObjectName("BadgeBad"); self.lbl_big.setStyleSheet("color:#ffb9b9;")
            else:
                self.lbl_source.setObjectName("BadgeNeutral"); self.lbl_big.setStyleSheet("color:#d8e7ff;")
            visible = _sanitize_series_for_plot(snap.series_main)
            n_visible = int(len(visible))
            if n_visible >= 1:
                first = float(visible["equity"].iloc[0]); last_v = float(visible["equity"].iloc[-1])
                delta = last_v - first
                pct = (delta / first * 100.0) if abs(first) > 1e-12 else 0.0
                self.lbl_delta.setText(f"Δ VIS: {delta:+,.2f} USD ({pct:+.2f}%)")
            else:
                self.lbl_delta.setText("Δ VIS: --")
            self.lbl_samples.setText(f"MUESTRAS: {n_visible}")
            self.lbl_state.setText("STATE: DEGRADED" if self.degraded_mode_active else "STATE: LIVE")
            age_s = (snap.now - snap.last_update).total_seconds() if snap.last_update else float("inf")
            if snap.source == "SERIE_CSV":
                health = "OK" if age_s <= RULE_STALE_S else "STALE"
            else:
                health = "MISSING" if n_visible <= 0 else "OK"
            self.lbl_csv_health.setText(f"CSV REAL: {health}")
        except Exception:
            snap.warnings.append("header render error")
            traceback.print_exc()

        # FASE 3 — Plots
        try:
            panel_items = [("main", snap.series_main)]
            if not self.degraded_mode_active:
                panel_items.extend([("min", snap.series_minutes), ("hour", snap.series_hours), ("day", snap.series_days)])
            for key, series in panel_items:
                try:
                    scale_info, py0, py1 = self._update_plot_state(self.plot_states[key], series)
                    if key == "main":
                        main_scale = scale_info
                        main_y0, main_y1 = py0, py1
                except Exception as plot_err:
                    snap.warnings.append(f"plot {key} con error: {plot_err}")
            self.lbl_scale.setText(f"ESCALA Y: {main_scale}")
        except Exception:
            snap.warnings.append("plot phase error")
            traceback.print_exc()

        # FASE 4 — Métricas live (secundarias)
        try:
            metrics = self._compute_live_metrics(visible, snap)
            rule = self._evaluate_live_rules(metrics, snap)
            self.lbl_live_metrics.setText(str(metrics.get("text", "LIVE: --")))
            self.lbl_rule.setText(f"RULE: {rule}")
        except Exception:
            self.lbl_live_metrics.setText("LIVE: --")
            self.lbl_rule.setText("RULE: --")
            snap.warnings.append("live metrics error")
            traceback.print_exc()

        # FASE 5 — Warnings / tooltip / perf
        try:
            self.render_ms = (time.perf_counter() - t0) * 1000.0
            if self.snapshot_build_ms >= DEGRADED_WORKER_MS or self.render_ms >= DEGRADED_RENDER_MS:
                if not self.degraded_mode_active:
                    self.degraded_activations += 1
                self.degraded_mode_active = True
            elif self.consecutive_errors <= 0:
                self.degraded_mode_active = False
            perf_diag = (
                f"worker_busy={self._worker_busy} skipped={self.skipped_refresh_count} "
                f"build_ms={self.snapshot_build_ms:,.1f} render_ms={self.render_ms:,.1f} "
                f"degraded={self.degraded_mode_active}"
            )
            if snap.warnings:
                compact = [w.strip()[:100] + ("…" if len(w.strip()) > 100 else "") for w in snap.warnings[:3]]
                self.lbl_warn.setText("⚠ " + " · ".join(compact))
                self.lbl_warn.setToolTip("\n".join(snap.warnings + [perf_diag, f"Escala main: {Y_SCALE_MODE} | y=[{main_y0:,.2f}, {main_y1:,.2f}]"]))
            else:
                self.lbl_warn.setText("")
                self.lbl_warn.setToolTip(perf_diag + f"\nEscala main: {Y_SCALE_MODE} | y=[{main_y0:,.2f}, {main_y1:,.2f}]")
            self.consecutive_errors = 0
        except Exception:
            traceback.print_exc()

    @QtCore.Slot(str)
    def _on_snapshot_error(self, tb_text: str):
        self._worker_busy = False
        self.refresh_in_progress = False
        self.last_error_ts = time.time()
        self.consecutive_errors += 1
        if self.consecutive_errors >= DEGRADED_ERRORS_THRESHOLD:
            self.degraded_mode_active = True
        self.lbl_state.setText("STATE: ERROR")
        self.lbl_warn.setText("⚠ Error en worker snapshot (se conserva último snapshot)")
        self.lbl_warn.setToolTip(tb_text[-2000:])
        print(tb_text)


def main():
    print(f"[MONITOR] Monitor Saldo Real Deriv {MONITOR_VERSION} · build={MONITOR_BUILD_ID}")
    try:
        app = QtWidgets.QApplication(sys.argv)
        w = DashboardWindow(DataEngine(Path(__file__).resolve().parent))
        w.show()
        return int(app.exec())
    except Exception:
        print("[MONITOR][ERROR] Falló el arranque del monitor.")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
