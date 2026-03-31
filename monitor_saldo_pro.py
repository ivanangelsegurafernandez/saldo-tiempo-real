#!/usr/bin/env python3
"""
Monitor de saldo REAL DERIV (PySide6 + PyQtGraph).

Dependencias:
  pip install pandas numpy pyqtgraph PySide6

Archivos leídos (solo lectura):
- saldo_real_live.json (prioridad 1 para REAL)
- LOG_SALDOS, *.log, *.txt (prioridad 2/3)
- registro_enriquecido_fulll*.csv (solo auxiliar/estimado)

Teclas:
- 1 REAL, 2 DEMO, 3 ALL
- F fullscreen toggle
- P pausar/reanudar
- R reset zoom
- Q salir
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

# Config
CUENTA_OBJETIVO = "REAL"  # REAL | DEMO | ALL
REFRESH_SEGUNDOS = 5
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


def _now() -> datetime:
    return datetime.now().astimezone()


def _fmt_money(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:,.2f} USD"


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
        span = max(values) - min(values)
        out = []
        for v in values:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
            if span <= 6 * 3600:
                out.append(dt.strftime("%H:%M"))
            elif span <= 48 * 3600:
                out.append(dt.strftime("%d-%m %H:%M"))
            else:
                out.append(dt.strftime("%d-%m"))
        return out


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

    def _master_live_candidates(self) -> List[Path]:
        candidates: List[Path] = [Path(SALDO_LIVE_SHARED_PATH).expanduser()]
        if SALDO_LIVE_PATH:
            custom = Path(SALDO_LIVE_PATH).expanduser()
            candidates.append(custom / SALDO_LIVE_FILE if custom.is_dir() else custom)
        candidates.append(self.base_dir / SALDO_LIVE_FILE)
        candidates.append(Path.cwd() / SALDO_LIVE_FILE)

        unique: List[Path] = []
        seen = set()
        for p in candidates:
            k = str(p.resolve()) if p.exists() else str(p)
            if k in seen:
                continue
            seen.add(k)
            unique.append(p)
        return unique

    def _master_history_candidates(self) -> List[Path]:
        candidates: List[Path] = [Path(SALDO_LIVE_HISTORY_SHARED_PATH).expanduser()]
        for live_path in self._master_live_candidates():
            candidates.append(live_path.parent / SALDO_LIVE_HISTORY_FILE)
        unique: List[Path] = []
        seen = set()
        for p in candidates:
            k = str(p.resolve()) if p.exists() else str(p)
            if k in seen:
                continue
            seen.add(k)
            unique.append(p)
        return unique

    def _read_master_live(self) -> Tuple[Optional[Tuple[float, datetime]], Optional[str]]:
        candidates = self._master_live_candidates()
        found_any = False
        for p in candidates:
            if not p.exists():
                continue
            found_any = True
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                v = _safe_float(obj.get("saldo_real"), default=np.nan)
                if not np.isfinite(v):
                    if str(p) == str(Path(SALDO_LIVE_SHARED_PATH)):
                        return None, f"{SALDO_LIVE_FILE} inválido en ruta compartida: {p}"
                    return None, f"{SALDO_LIVE_FILE} inválido en {p}"
                ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", utc=True)
                if pd.isna(ts):
                    ts = pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)
                return (float(v), ts.to_pydatetime()), None
            except Exception:
                if str(p) == str(Path(SALDO_LIVE_SHARED_PATH)):
                    return None, f"{SALDO_LIVE_FILE} inválido en ruta compartida: {p}"
                return None, f"{SALDO_LIVE_FILE} inválido en {p}"

        configured = SALDO_LIVE_PATH if SALDO_LIVE_PATH else "(no configurada; usando ruta local/cwd)"
        if not found_any:
            return None, f"{SALDO_LIVE_FILE} no encontrado en ruta compartida: {SALDO_LIVE_SHARED_PATH}"
        if not Path(SALDO_LIVE_SHARED_PATH).exists():
            return None, f"{SALDO_LIVE_FILE} no encontrado en ruta compartida: {SALDO_LIVE_SHARED_PATH} (fallback: {configured})"
        return None, f"saldo real del maestro no disponible ({SALDO_LIVE_FILE})"

    def _read_master_history(self) -> Tuple[pd.DataFrame, Optional[str]]:
        for p in self._master_history_candidates():
            if not p.exists():
                continue
            try:
                rows: List[Dict[str, object]] = []
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
                        rows.append(
                            {
                                "timestamp": ts.to_pydatetime(),
                                "equity": float(v),
                                "status": str(obj.get("status", "")),
                                "source": str(obj.get("source", "MAESTRO_HIST")),
                            }
                        )
                if not rows:
                    continue
                d = pd.DataFrame(rows).sort_values("timestamp")
                d = d.drop_duplicates(subset=["timestamp", "equity"], keep="last")
                return d[["timestamp", "equity"]], None
            except Exception:
                if str(p) == str(Path(SALDO_LIVE_HISTORY_SHARED_PATH)):
                    return pd.DataFrame(columns=["timestamp", "equity"]), f"{SALDO_LIVE_HISTORY_FILE} inválido en ruta compartida: {p}"
                return pd.DataFrame(columns=["timestamp", "equity"]), f"{SALDO_LIVE_HISTORY_FILE} inválido en {p}"

        return pd.DataFrame(columns=["timestamp", "equity"]), f"{SALDO_LIVE_HISTORY_FILE} no encontrado en ruta compartida: {SALDO_LIVE_HISTORY_SHARED_PATH}"

    def _parse_observed(self, view: str) -> pd.DataFrame:
        patterns: List[re.Pattern]
        if view == "REAL":
            patterns = [
                re.compile(r"SALDO\s+EN\s+CUENTA\s+REAL\s+DERIV\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE),
                re.compile(r"Saldo\s+cuenta\s+REAL(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE),
            ]
        elif view == "DEMO":
            patterns = [
                re.compile(r"Saldo\s+cuenta\s+DEMO(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", re.IGNORECASE),
            ]
        else:
            real = self._parse_observed("REAL")
            demo = self._parse_observed("DEMO")
            if real.empty and demo.empty:
                return pd.DataFrame(columns=["timestamp", "equity"])
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
                m = None
                for pat in patterns:
                    m = pat.search(line)
                    if m:
                        break
                if not m:
                    continue
                v = _safe_float(m.group(1), default=np.nan)
                if not np.isfinite(v):
                    continue
                ts = pd.to_datetime(line, errors="coerce", utc=True)
                if pd.isna(ts):
                    ts = base + timedelta(seconds=i)
                else:
                    ts = ts.to_pydatetime()
                rows.append((ts, float(v)))

        if not rows:
            return pd.DataFrame(columns=["timestamp", "equity"])
        d = pd.DataFrame(rows, columns=["timestamp", "equity"]).sort_values("timestamp")
        d = d.drop_duplicates(subset=["timestamp", "equity"], keep="last")
        return d

    def _build_estimated(self, view: str) -> pd.DataFrame:
        files = sorted(glob.glob(str(self.base_dir / CSV_PATTERN)))
        if not files:
            return pd.DataFrame(columns=["timestamp", "equity"])
        dfs = []
        for p in files:
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    dfs.append(pd.read_csv(p, encoding=enc, low_memory=False))
                    break
                except Exception:
                    pass
        if not dfs:
            return pd.DataFrame(columns=["timestamp", "equity"])
        d = pd.concat(dfs, ignore_index=True)
        if "ganancia_perdida" not in d.columns:
            return pd.DataFrame(columns=["timestamp", "equity"])
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
        if "cuenta" in d.columns:
            c = d["cuenta"].astype(str).str.upper()
            if view in ("REAL", "DEMO"):
                d = d[c.str.contains(view, na=False)]
        pnl = pd.to_numeric(d["ganancia_perdida"], errors="coerce").fillna(0.0)
        base = 1000.0 if view == "REAL" else (10000.0 if view == "DEMO" else 11000.0)
        d["equity"] = base + pnl.cumsum()
        return d[["timestamp", "equity"]]

    def build_snapshot(self, view: str) -> Snapshot:
        now = _now()
        warnings: List[str] = []

        master, master_msg = self._read_master_live() if view == "REAL" else (None, None)
        history, history_msg = self._read_master_history() if view == "REAL" else (pd.DataFrame(columns=["timestamp", "equity"]), None)
        observed = self._parse_observed(view)
        estimated = self._build_estimated(view)
        if master_msg and view == "REAL":
            warnings.append(master_msg)
        if history_msg and view == "REAL":
            warnings.append(history_msg)

        source = "SIN DATOS REALES"
        saldo_actual: Optional[float] = None
        last_update: Optional[datetime] = None
        real_series = pd.DataFrame(columns=["timestamp", "equity"])

        if view == "REAL" and not history.empty:
            real_series = history.copy()
            if master is not None:
                mv, mts = master
                source = "MAESTRO"
                saldo_actual = mv
                last_update = mts
                real_series = pd.concat([real_series, pd.DataFrame([{"timestamp": mts, "equity": mv}])], ignore_index=True)
                real_series = real_series.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            else:
                source = "MAESTRO_HIST"
                saldo_actual = float(real_series["equity"].iloc[-1])
                last_update = real_series["timestamp"].iloc[-1]
            if len(real_series) < 3:
                warnings.append("historial real en construcción (aún pocas muestras)")
        elif master is not None:
            mv, mts = master
            source = "MAESTRO"
            saldo_actual = mv
            last_update = mts
            if not observed.empty:
                real_series = observed.copy()
                real_series = pd.concat([real_series, pd.DataFrame([{"timestamp": mts, "equity": mv}])], ignore_index=True)
                real_series = real_series.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            else:
                real_series = pd.DataFrame([{"timestamp": mts, "equity": mv}])
        elif not observed.empty:
            source = "OBSERVADO"
            saldo_actual = float(observed["equity"].iloc[-1])
            last_update = observed["timestamp"].iloc[-1]
            real_series = observed
        else:
            # intento live por regex (último valor en logs)
            live_real = self._parse_observed("REAL") if view == "REAL" else self._parse_observed(view)
            if not live_real.empty:
                source = "LIVE"
                saldo_actual = float(live_real["equity"].iloc[-1])
                last_update = live_real["timestamp"].iloc[-1]
                real_series = pd.DataFrame([live_real.iloc[-1]])
            else:
                source = "SIN DATOS REALES"
                saldo_actual = None
                last_update = None
                warnings.append("saldo real del maestro no disponible")

        if source == "SIN DATOS REALES" and not estimated.empty:
            warnings.append("Estimado CSV disponible solo como auxiliar (no saldo principal).")

        if not real_series.empty:
            mcut = now - timedelta(minutes=VENTANA_MINUTOS)
            hcut = now - timedelta(hours=VENTANA_HORAS)
            dcut = now - timedelta(days=VENTANA_DIAS)
            smin = real_series[real_series["timestamp"] >= mcut].copy()
            shrs = real_series[real_series["timestamp"] >= hcut].copy()
            sday_raw = real_series[real_series["timestamp"] >= dcut].copy()
            sday = sday_raw.copy()
            if not sday.empty:
                sday = sday.set_index("timestamp").resample("1D").last().dropna().reset_index()
        else:
            smin = shrs = sday = pd.DataFrame(columns=["timestamp", "equity"])

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
        self.setWindowTitle("Monitor Saldo Real Deriv")
        self.resize(1600, 900)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setObjectName("HeaderCard")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(12)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)
        self.lbl_title = QtWidgets.QLabel("SALDO REAL DERIV ACTUAL")
        self.lbl_title.setObjectName("Title")
        self.lbl_source = QtWidgets.QLabel("FUENTE: --")
        self.lbl_source.setObjectName("BadgeWarn")
        top_row.addWidget(self.lbl_title, 1)
        top_row.addWidget(self.lbl_source, 0)
        header_layout.addLayout(top_row)

        self.lbl_big = QtWidgets.QLabel("--")
        self.lbl_big.setObjectName("Big")
        self.lbl_big.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_big.setMinimumHeight(118)
        header_layout.addWidget(self.lbl_big)

        meta_row = QtWidgets.QHBoxLayout()
        meta_row.setSpacing(10)
        self.lbl_refresh = QtWidgets.QLabel("REFRESCO: ACTIVO")
        self.lbl_refresh.setObjectName("MetaBox")
        self.lbl_now = QtWidgets.QLabel("HORA ACTUAL: --")
        self.lbl_now.setObjectName("MetaNow")
        self.lbl_last = QtWidgets.QLabel("ÚLTIMA ACT: --")
        self.lbl_last.setObjectName("MetaLast")
        meta_row.addWidget(self.lbl_refresh)
        meta_row.addWidget(self.lbl_now, 1)
        meta_row.addWidget(self.lbl_last)
        header_layout.addLayout(meta_row)
        root.addWidget(header)

        self.graphics = pg.GraphicsLayoutWidget()
        root.addWidget(self.graphics, 1)

        self.p_min = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self._style_plot(self.p_min, "MINUTOS · lectura rápida")

        self.p_hour = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self._style_plot(self.p_hour, "HORAS · comportamiento reciente")

        self.p_day = self.graphics.addPlot(row=2, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self._style_plot(self.p_day, "DÍAS · tendencia general")

        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setObjectName("Warn")
        root.addWidget(self.lbl_warn)

        self.lbl_help = QtWidgets.QLabel("Teclas: [1]REAL [2]DEMO [3]ALL [F]Fullscreen [P]Pausa [R]Reset [Q]Salir")
        self.lbl_help.setObjectName("Help")
        root.addWidget(self.lbl_help)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b0f14; color: #d9e2f2; }
            #HeaderCard { background: #0f1622; border: 1px solid #203047; border-radius: 14px; }
            #Title { font-size: 20px; color: #b9d3ff; font-weight: 800; letter-spacing: 0.6px; }
            #Big { font-size: 98px; color: #ecfff3; font-weight: 900; padding: 10px 0 14px 0; }
            #MetaBox { font-size: 14px; color: #bdd6ff; background: #101e31; border: 1px solid #263e5f; border-radius: 10px; padding: 8px 12px; font-weight: 650; }
            #MetaNow { font-size: 18px; color: #ecf6ff; background: #153153; border: 1px solid #2f5e92; border-radius: 10px; padding: 8px 14px; font-weight: 800; }
            #MetaLast { font-size: 14px; color: #d9e8ff; background: #12263d; border: 1px solid #2c4b72; border-radius: 10px; padding: 8px 12px; font-weight: 650; }
            #BadgeMaster { font-size: 13px; color: #041d13; background: #72f8b1; border: 1px solid #9dffd0; border-radius: 13px; padding: 4px 11px; font-weight: 900; }
            #BadgeObserved { font-size: 13px; color: #02222b; background: #67efff; border: 1px solid #8ff6ff; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeLive { font-size: 13px; color: #0b2a1f; background: #9ef7d8; border: 1px solid #c0ffe8; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeNeutral { font-size: 13px; color: #d8e7ff; background: #23364f; border: 1px solid #3d5c81; border-radius: 13px; padding: 4px 11px; font-weight: 800; }
            #BadgeWarn { font-size: 13px; color: #3d2a00; background: #ffd67f; border: 1px solid #ffe09e; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #BadgeBad { font-size: 13px; color: #390000; background: #ff9c9c; border: 1px solid #ffb8b8; border-radius: 13px; padding: 4px 11px; font-weight: 850; }
            #Warn { font-size: 12px; color: #ffc374; font-weight: 600; }
            #Help { font-size: 10px; color: #7690b2; }
            """
        )
        pg.setConfigOptions(antialias=True, background="#0b0f14", foreground="#d9e2f2")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(max(1000, int(REFRESH_SEGUNDOS * 1000)))

        if FULLSCREEN_INICIAL:
            self.showMaximized()

        self.refresh()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k == QtCore.Qt.Key_1:
            self.view = "REAL"
            self.refresh(force=True)
        elif k == QtCore.Qt.Key_2:
            self.view = "DEMO"
            self.refresh(force=True)
        elif k == QtCore.Qt.Key_3:
            self.view = "ALL"
            self.refresh(force=True)
        elif k == QtCore.Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == QtCore.Qt.Key_P:
            self.paused = not self.paused
            self.refresh(force=True)
        elif k == QtCore.Qt.Key_R:
            self.p_min.enableAutoRange(); self.p_hour.enableAutoRange(); self.p_day.enableAutoRange()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(ev)

    @staticmethod
    def _style_plot(plot: pg.PlotItem, title: str):
        plot.setTitle(f"<span style='color:#cfe2ff;font-size:13pt;font-weight:700'>{title}</span>")
        plot.setLabel("left", "Dinero (USD)")
        plot.showGrid(x=True, y=True, alpha=0.05)
        axis_left = plot.getAxis("left")
        axis_bottom = plot.getAxis("bottom")
        axis_left.setTextPen(pg.mkPen("#b9d0ee"))
        axis_bottom.setTextPen(pg.mkPen("#a8bfdc"))
        axis_left.setPen(pg.mkPen("#35506f"))
        axis_bottom.setPen(pg.mkPen("#35506f"))

    @staticmethod
    def _plot_series(plot: pg.PlotItem, s: pd.DataFrame, color="#66d9ff", endpoint="#d8ff6e"):
        plot.clear()
        if s.empty:
            return
        if plot.legend is None:
            plot.addLegend(offset=(10, 8))
        else:
            plot.legend.clear()
        x = (s["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = s["equity"].to_numpy(dtype=float)
        plot.plot(x, y, pen=pg.mkPen(color, width=3.6), name="Serie real")
        plot.plot([x[-1]], [y[-1]], pen=pg.mkPen("#ffffff33"), symbol="o", symbolSize=11, symbolBrush=endpoint, name="Último")

        imax = int(np.argmax(y))
        imin = int(np.argmin(y))
        plot.plot([x[imax]], [y[imax]], pen=None, symbol="t", symbolSize=11, symbolBrush="#ffd36b", name="Máximo")
        plot.plot([x[imin]], [y[imin]], pen=None, symbol="t1", symbolSize=11, symbolBrush="#ff8f8f", name="Mínimo")

        if len(y) >= 6:
            dy = np.diff(y)
            std = float(np.std(dy)) if len(dy) else 0.0
            if std > 0:
                j = int(np.argmax(np.abs(dy)))
                if abs(float(dy[j])) >= 2.0 * std:
                    plot.plot(
                        [x[j + 1]],
                        [y[j + 1]],
                        pen=None,
                        symbol="s",
                        symbolSize=10,
                        symbolBrush="#f4a4ff",
                        name="Salto",
                    )

    def refresh(self, force: bool = False):
        if self.paused and not force:
            return
        try:
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
                self.lbl_source.setObjectName("BadgeMaster")
                self.lbl_big.setStyleSheet("color:#72f8b1;")
            elif src == "OBSERVADO":
                self.lbl_source.setObjectName("BadgeObserved")
                self.lbl_big.setStyleSheet("color:#67efff;")
            elif src == "LIVE":
                self.lbl_source.setObjectName("BadgeLive")
                self.lbl_big.setStyleSheet("color:#9ef7d8;")
            elif src == "STALE":
                self.lbl_source.setObjectName("BadgeWarn")
                self.lbl_big.setStyleSheet("color:#ffe9b8;")
            elif src in ("ESTIMADO", "SIN DATOS REALES"):
                self.lbl_source.setObjectName("BadgeBad")
                self.lbl_big.setStyleSheet("color:#ffb9b9;")
            else:
                self.lbl_source.setObjectName("BadgeNeutral")
                self.lbl_big.setStyleSheet("color:#d8e7ff;")
            self.lbl_source.style().unpolish(self.lbl_source)
            self.lbl_source.style().polish(self.lbl_source)

            self._plot_series(self.p_min, snap.series_minutes, color="#69f5ff", endpoint="#ccfbff")
            self._plot_series(self.p_hour, snap.series_hours, color="#7ca8ff", endpoint="#d7e2ff")
            self._plot_series(self.p_day, snap.series_days, color="#7ff0b9", endpoint="#dcffe9")

            # curva estimada solo auxiliar y tenue (si no hay real)
            if snap.series_minutes.empty and not snap.series_est.empty:
                aux = snap.series_est.tail(200)
                x = (aux["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
                y = aux["equity"].to_numpy(dtype=float)
                self.p_hour.plot(x, y, pen=pg.mkPen("#888888", width=1.0, style=QtCore.Qt.DashLine))

            if snap.warnings:
                compact = [w.strip()[:96] + ("…" if len(w.strip()) > 96 else "") for w in snap.warnings[:2]]
                self.lbl_warn.setText("⚠ " + " | ".join(compact))
            else:
                self.lbl_warn.setText("")
        except Exception as e:
            self.lbl_warn.setText(f"⚠ Error monitor: {e}")
            traceback.print_exc()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = DashboardWindow(DataEngine(Path(__file__).resolve().parent))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
