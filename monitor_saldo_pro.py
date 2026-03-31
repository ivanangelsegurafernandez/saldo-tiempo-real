#!/usr/bin/env python3
"""
Monitor de saldo REAL DERIV (PySide6 + PyQtGraph).

Dependencias:
  pip install pandas numpy pyqtgraph PySide6

Lectura de datos (prioridad REAL):
1) saldo_real_live_history.jsonl (ruta compartida)
2) saldo_real_live.json (snapshot maestro)
3) LOG_SALDOS / *.log / *.txt (observado)
4) registro_enriquecido_fulll*.csv (auxiliar)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

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
SALDO_LIVE_SHARED_PATH = os.path.abspath(
    os.getenv("SALDO_LIVE_SHARED_PATH", os.path.join(os.path.expanduser("~"), SALDO_LIVE_FILE))
)
SALDO_LIVE_HISTORY_SHARED_PATH = os.path.abspath(
    os.getenv(
        "SALDO_LIVE_HISTORY_SHARED_PATH",
        os.path.join(os.path.dirname(SALDO_LIVE_SHARED_PATH), SALDO_LIVE_HISTORY_FILE),
    )
)
SALDO_LIVE_PATH = os.getenv("SALDO_LIVE_PATH", "").strip()

MONITOR_VERSION = "v2026.03.31-r1"
MONITOR_BUILD_ID = "MONITOR_SALDO_PRO_REAL_SERIES_GUARD"
MIN_POINTS_FOR_LINE = 2
SHOW_LAST_MARKER = True
SHOW_EXTREME_MARKERS = False



def _now() -> datetime:
    return datetime.now().astimezone()


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
            return ts_obj.to_pydatetime().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if isinstance(ts_obj, datetime):
            if ts_obj.tzinfo is None:
                ts_obj = ts_obj.replace(tzinfo=timezone.utc)
            return ts_obj.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
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
                dt_local = dt_utc.astimezone()
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
    series_minutes: pd.DataFrame
    series_hours: pd.DataFrame
    series_days: pd.DataFrame
    series_est: pd.DataFrame
    warnings: List[str]
    view: str


class DataEngine:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._cache: Dict[str, Tuple[Tuple, object]] = {}
        self._history_last_seen: Optional[Tuple[str, int, int]] = None

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
        cands.append(self.base_dir / SALDO_LIVE_FILE)
        cands.append(Path.cwd() / SALDO_LIVE_FILE)
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
                if rows:
                    d = pd.DataFrame(rows, columns=["timestamp", "equity"]).sort_values("timestamp")
                    d = d.drop_duplicates(subset=["timestamp", "equity"], keep="last")
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
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
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
        return self._store_cache(key, sig, d[["timestamp", "equity"]])

    def build_snapshot(self, view: str) -> Snapshot:
        now = _now()
        warnings: List[str] = []

        master, master_msg, live_path_used = self._read_master_live() if view == "REAL" else (None, None, None)
        hist, hist_msg, hist_path_used, hist_growth_msg = self._read_master_history() if view == "REAL" else (pd.DataFrame(columns=["timestamp", "equity"]), None, None, None)
        observed = self._parse_observed(view)
        estimated = self._build_estimated(view)

        if master_msg and view == "REAL":
            warnings.append(master_msg)
        if hist_msg and view == "REAL":
            warnings.append(hist_msg)

        if view == "REAL":
            warnings.append(f"Monitor {MONITOR_VERSION} · id={MONITOR_BUILD_ID}")
            if not hasattr(self, "_check_history_growth"):
                warnings.append("Versión desactualizada detectada: faltan validaciones de crecimiento de histórico")
            warnings.append(f"Ruta snapshot real: {live_path_used if live_path_used else SALDO_LIVE_SHARED_PATH}")
            warnings.append(f"Ruta histórico real: {hist_path_used if hist_path_used else SALDO_LIVE_HISTORY_SHARED_PATH}")
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

        smin = real_series[real_series["timestamp"] >= mcut].copy() if not real_series.empty else pd.DataFrame(columns=["timestamp", "equity"])
        shrs = real_series[real_series["timestamp"] >= hcut].copy() if not real_series.empty else pd.DataFrame(columns=["timestamp", "equity"])
        sday = real_series[real_series["timestamp"] >= dcut].copy() if not real_series.empty else pd.DataFrame(columns=["timestamp", "equity"])
        if not sday.empty:
            try:
                sday_daily = sday.set_index("timestamp").resample("1D").last().dropna().reset_index()
                if len(sday_daily) >= MIN_POINTS_FOR_LINE:
                    sday = sday_daily
                else:
                    try:
                        sday_fallback = sday.set_index("timestamp").resample("6h").last().dropna().reset_index()
                        if len(sday_fallback) >= MIN_POINTS_FOR_LINE:
                            sday = sday_fallback
                            warnings.append("Panel DÍAS en fallback 6h: histórico diario aún insuficiente")
                        else:
                            sday = sday.tail(min(120, len(sday))).copy()
                            warnings.append("Panel DÍAS en fallback crudo: histórico diario aún insuficiente")
                    except Exception as e_fallback:
                        sday = sday.tail(min(120, len(sday))).copy()
                        warnings.append(f"Fallback 6h DÍAS no disponible: {e_fallback}")
            except Exception as e_daily:
                sday = sday.tail(min(120, len(sday))).copy()
                warnings.append(f"Resample diario DÍAS no disponible: {e_daily}")

        return Snapshot(
            source=source,
            saldo_actual=saldo_actual,
            last_update=last_update,
            now=now,
            series_real=real_series,
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
        self.lbl_now = QtWidgets.QLabel("HORA LOCAL: --"); self.lbl_now.setObjectName("MetaNow")
        self.lbl_last = QtWidgets.QLabel("ÚLTIMA ACT: --"); self.lbl_last.setObjectName("MetaLast")
        self.lbl_build = QtWidgets.QLabel(f"BUILD: {MONITOR_BUILD_ID}"); self.lbl_build.setObjectName("MetaBox")
        meta.addWidget(self.lbl_refresh); meta.addWidget(self.lbl_now, 1); meta.addWidget(self.lbl_last); meta.addWidget(self.lbl_build)
        hl.addLayout(meta)
        root.addWidget(header)

        self.graphics = pg.GraphicsLayoutWidget(); root.addWidget(self.graphics, 1)
        self.p_min = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})
        self.p_hour = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})
        self.p_day = self.graphics.addPlot(row=2, col=0, axisItems={"bottom": SmartDateAxis("bottom"), "left": MoneyAxis("left")})

        self._style_plot(self.p_min, "MINUTOS · lectura rápida")
        self._style_plot(self.p_hour, "HORAS · comportamiento reciente")
        self._style_plot(self.p_day, "DÍAS · tendencia general")

        self.plot_states = {
            "min": self._init_plot_state(self.p_min, "#3fe9ff", "#c6f7ff"),
            "hour": self._init_plot_state(self.p_hour, "#7aa6ff", "#dae3ff"),
            "day": self._init_plot_state(self.p_day, "#7ff0b9", "#dcffe9"),
        }

        self.lbl_warn = QtWidgets.QLabel(""); self.lbl_warn.setObjectName("Warn"); root.addWidget(self.lbl_warn)
        self.lbl_help = QtWidgets.QLabel(f"Teclas: [1]REAL [2]DEMO [3]ALL [F]Fullscreen [P]Pausa [R]Reset [Q]Salir · {MONITOR_VERSION} · {MONITOR_BUILD_ID}")
        self.lbl_help.setObjectName("Help"); root.addWidget(self.lbl_help)

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
        pg.setConfigOptions(antialias=True, background="#0b0f14", foreground="#d9e2f2")

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

    def _init_plot_state(self, plot: pg.PlotItem, color: str, endpoint: str) -> Dict[str, object]:
        glow = plot.plot([], [], pen=pg.mkPen(color + "55", width=8.0), name=None)
        line = plot.plot([], [], pen=pg.mkPen(color, width=4.8), name="Equity")
        last = plot.plot([], [], pen=None, symbol="o", symbolSize=6, symbolBrush=endpoint, name=None)
        vmax = plot.plot([], [], pen=None, symbol="o", symbolSize=4, symbolBrush="#ffd36b99", name=None)
        vmin = plot.plot([], [], pen=None, symbol="o", symbolSize=4, symbolBrush="#ff8f8f99", name=None)
        txt = pg.TextItem(text="", color="#9ec2ff", anchor=(0, 1))
        plot.addItem(txt)
        return {"plot": plot, "glow": glow, "line": line, "last": last, "max": vmax, "min": vmin, "text": txt}

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
            self.p_min.enableAutoRange(); self.p_hour.enableAutoRange(); self.p_day.enableAutoRange()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(ev)

    def changeEvent(self, ev: QtCore.QEvent):
        if ev.type() == QtCore.QEvent.WindowStateChange:
            interval = REFRESH_SEGUNDOS_MINIMIZADO if self.isMinimized() else REFRESH_SEGUNDOS
            self.timer.setInterval(int(interval * 1000))
        super().changeEvent(ev)

    def _update_plot_state(self, state: Dict[str, object], s: pd.DataFrame):
        plot = state["plot"]
        glow = state["glow"]; line = state["line"]; last = state["last"]; vmax = state["max"]; vmin = state["min"]; txt = state["text"]
        s = _sanitize_series_for_plot(s)
        if s.empty:
            glow.setData([], [])
            line.setData([], [])
            last.setData([], [])
            vmax.setData([], [])
            vmin.setData([], [])
            txt.setText("")
            return

        x = (s["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = s["equity"].to_numpy(dtype=float)

        if len(x) >= MIN_POINTS_FOR_LINE:
            glow.setData(x, y)
            line.setData(x, y)
            txt.setText("")
        else:
            glow.setData([], [])
            line.setData([], [])
            txt.setText("")

        marker_size = 8 if len(x) == 1 else 6
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
        y_recent = y[-min(len(y), 60):]
        q_low = float(np.nanpercentile(y_recent, 10))
        q_high = float(np.nanpercentile(y_recent, 90))
        y_min = float(np.min(y_recent))
        y_max = float(np.max(y_recent))
        lower = min(y_min, q_low)
        upper = max(y_max, q_high)
        span = upper - lower
        if span <= 0.0:
            pad = max(0.05, abs(upper) * 0.005)
        else:
            pad = max(0.05, span * 0.28)
        plot.setYRange(round(lower - pad, 2), round(upper + pad, 2), padding=0.0)

    def refresh(self, force: bool = False):
        if self.paused and not force:
            return
        if self.isMinimized() and not force:
            return
        try:
            if not MONITOR_BUILD_ID or MIN_POINTS_FOR_LINE != 2:
                self.lbl_warn.setText("⚠ Versión desactualizada o incompleta del monitor detectada")
            snap = self.engine.build_snapshot(self.view)
            self.lbl_big.setText(_fmt_money(snap.saldo_actual))
            refresh_state = "PAUSADO" if self.paused else "ACTIVO"
            last = snap.last_update.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if snap.last_update else "--"
            self.lbl_refresh.setText(f"REFRESCO: {refresh_state}")
            self.lbl_now.setText(f"HORA LOCAL: {snap.now.strftime('%H:%M:%S %Z')}")
            self.lbl_last.setText(f"ÚLTIMA ACT: {last}")

            src = snap.source.upper().strip()
            self.lbl_source.setText(f"FUENTE: {src}")
            if src == "MAESTRO":
                self.lbl_source.setObjectName("BadgeMaster"); self.lbl_big.setStyleSheet("color:#72f8b1;")
            elif src in ("OBSERVADO", "MAESTRO_HIST"):
                self.lbl_source.setObjectName("BadgeObserved"); self.lbl_big.setStyleSheet("color:#67efff;")
            elif src == "LIVE":
                self.lbl_source.setObjectName("BadgeLive"); self.lbl_big.setStyleSheet("color:#9ef7d8;")
            elif src == "STALE":
                self.lbl_source.setObjectName("BadgeWarn"); self.lbl_big.setStyleSheet("color:#ffe9b8;")
            elif src in ("ESTIMADO", "SIN DATOS REALES"):
                self.lbl_source.setObjectName("BadgeBad"); self.lbl_big.setStyleSheet("color:#ffb9b9;")
            else:
                self.lbl_source.setObjectName("BadgeNeutral"); self.lbl_big.setStyleSheet("color:#d8e7ff;")
            self.lbl_source.style().unpolish(self.lbl_source); self.lbl_source.style().polish(self.lbl_source)

            self._update_plot_state(self.plot_states["min"], snap.series_minutes)
            self._update_plot_state(self.plot_states["hour"], snap.series_hours)
            self._update_plot_state(self.plot_states["day"], snap.series_days)

            if snap.warnings:
                compact = [w.strip()[:110] + ("…" if len(w.strip()) > 110 else "") for w in snap.warnings[:3]]
                self.lbl_warn.setText("⚠ " + " · ".join(compact))
                self.lbl_warn.setToolTip("\n".join(snap.warnings))
            else:
                self.lbl_warn.setText("")
                self.lbl_warn.setToolTip("")
        except Exception as e:
            self.lbl_warn.setText(f"⚠ Error monitor: {e}")
            traceback.print_exc()


def main():
    print(f"[MONITOR] Monitor Saldo Real Deriv {MONITOR_VERSION} · build={MONITOR_BUILD_ID}")
    app = QtWidgets.QApplication(sys.argv)
    w = DashboardWindow(DataEngine(Path(__file__).resolve().parent))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
