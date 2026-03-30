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


def _now() -> datetime:
    return datetime.now(timezone.utc)


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

    def _read_master_live(self) -> Optional[Tuple[float, datetime]]:
        p = self.base_dir / SALDO_LIVE_FILE
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            v = _safe_float(obj.get("saldo_real"), default=np.nan)
            if not np.isfinite(v):
                return None
            ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", utc=True)
            if pd.isna(ts):
                ts = pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)
            return float(v), ts.to_pydatetime()
        except Exception:
            return None

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

        master = self._read_master_live() if view == "REAL" else None
        observed = self._parse_observed(view)
        estimated = self._build_estimated(view)

        source = "SIN DATOS REALES"
        saldo_actual: Optional[float] = None
        last_update: Optional[datetime] = None
        real_series = pd.DataFrame(columns=["timestamp", "equity"])

        if master is not None:
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
                warnings.append("SALDO REAL NO DISPONIBLE. Se oculta saldo principal estimado.")

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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("SALDO REAL DERIV ACTUAL")
        self.lbl_title.setObjectName("Title")
        self.lbl_big = QtWidgets.QLabel("--")
        self.lbl_big.setObjectName("Big")
        self.lbl_meta = QtWidgets.QLabel("FUENTE SALDO: -- | REFRESCO: ACTIVO | HORA ACTUAL: -- | ÚLTIMA ACT: --")
        self.lbl_meta.setObjectName("Meta")

        root.addWidget(self.lbl_title)
        root.addWidget(self.lbl_big)
        root.addWidget(self.lbl_meta)

        self.graphics = pg.GraphicsLayoutWidget()
        root.addWidget(self.graphics, 1)

        self.p_min = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self.p_min.setTitle("MINUTOS")
        self.p_min.setLabel("left", "Dinero (USD)")
        self.p_min.showGrid(x=True, y=True, alpha=0.08)

        self.p_hour = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self.p_hour.setTitle("HORAS")
        self.p_hour.setLabel("left", "Dinero (USD)")
        self.p_hour.showGrid(x=True, y=True, alpha=0.08)

        self.p_day = self.graphics.addPlot(row=2, col=0, axisItems={"bottom": SmartDateAxis("bottom")})
        self.p_day.setTitle("DÍAS")
        self.p_day.setLabel("left", "Dinero (USD)")
        self.p_day.showGrid(x=True, y=True, alpha=0.08)

        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setObjectName("Warn")
        root.addWidget(self.lbl_warn)

        self.lbl_help = QtWidgets.QLabel("Teclas: [1]REAL [2]DEMO [3]ALL [F]Fullscreen [P]Pausa [R]Reset [Q]Salir")
        self.lbl_help.setObjectName("Help")
        root.addWidget(self.lbl_help)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b0f14; color: #d9e2f2; }
            #Title { font-size: 16px; color: #8aa4cf; font-weight: 600; }
            #Big { font-size: 56px; color: #f1f6ff; font-weight: 800; }
            #Meta { font-size: 13px; color: #a7c0e8; }
            #Warn { font-size: 12px; color: #ffb86b; }
            #Help { font-size: 11px; color: #9fb7d9; }
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
    def _plot_series(plot: pg.PlotItem, s: pd.DataFrame, color="#66d9ff"):
        plot.clear()
        if s.empty:
            return
        x = (s["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = s["equity"].to_numpy(dtype=float)
        plot.plot(x, y, pen=pg.mkPen(color, width=2.0))
        plot.plot([x[-1]], [y[-1]], pen=None, symbol="o", symbolSize=7, symbolBrush="#d8ff6e")

    def refresh(self, force: bool = False):
        if self.paused and not force:
            return
        try:
            snap = self.engine.build_snapshot(self.view)
            self.lbl_big.setText(_fmt_money(snap.saldo_actual))
            refresh_state = "PAUSADO" if self.paused else "ACTIVO"
            last = snap.last_update.strftime("%Y-%m-%d %H:%M:%S UTC") if snap.last_update else "--"
            self.lbl_meta.setText(
                f"FUENTE SALDO: {snap.source} | REFRESCO: {refresh_state} | HORA ACTUAL: {snap.now.strftime('%H:%M:%S UTC')} | ÚLTIMA ACT: {last}"
            )

            self._plot_series(self.p_min, snap.series_minutes, color="#7ee6ff")
            self._plot_series(self.p_hour, snap.series_hours, color="#6cc5ff")
            self._plot_series(self.p_day, snap.series_days, color="#66ffb2")

            # curva estimada solo auxiliar y tenue (si no hay real)
            if snap.series_minutes.empty and not snap.series_est.empty:
                aux = snap.series_est.tail(200)
                x = (aux["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
                y = aux["equity"].to_numpy(dtype=float)
                self.p_hour.plot(x, y, pen=pg.mkPen("#888888", width=1.0, style=QtCore.Qt.DashLine))

            self.lbl_warn.setText("⚠ " + " | ".join(snap.warnings[:4]) if snap.warnings else "")
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
