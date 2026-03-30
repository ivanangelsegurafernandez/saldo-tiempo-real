#!/usr/bin/env python3
"""
Dashboard de saldo/equity profesional en tiempo real (PyQtGraph + PySide6).

Dependencias requeridas:
- Python 3.10+
- pandas
- numpy
- pyqtgraph
- PySide6

Dependencias opcionales:
- ninguno (solo archivos opcionales de datos)

Instalación rápida:
    pip install pandas numpy pyqtgraph PySide6

Ejemplo de ejecución:
    python monitor_saldo_pro.py

Archivos principales:
- registro_enriquecido_fulll*.csv (obligatorio para histórico)

Archivos opcionales:
- LOG_SALDOS (txt/log con líneas como "Saldo cuenta REAL: 1234 USD")
- retiros.csv (fecha,monto,comentario)

Teclas:
- 1 = vista REAL
- 2 = vista DEMO
- 3 = vista TOTAL
- F = fullscreen on/off
- P = pausar/reanudar refresco
- R = reset zoom
- Q = salir
"""

from __future__ import annotations

import glob
import math
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

# =========================
# Configuración manual
# =========================
CUENTA_OBJETIVO = "REAL"  # "REAL" | "DEMO" | "ALL"
SALDO_INICIAL_REAL = 1000.0
SALDO_INICIAL_DEMO = 10000.0
VENTANA_HORAS = 9
DIAS_GRAFICA = 14
REFRESH_SEGUNDOS = 5
LOG_SALDOS = "LOG_SALDOS"
FULLSCREEN_INICIAL = False
STALE_SALDO_SEGUNDOS = 120
MARGEN_SEGURIDAD_COLCHON = 0.25

CSV_PATTERN = "registro_enriquecido_fulll*.csv"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_money(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:,.2f} USD"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:,.2f}%"


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip().replace("$", "").replace("USD", "")
            x = x.replace(" ", "").replace(",", "")
        return float(x)
    except Exception:
        return default


@dataclass
class LiveBalance:
    real: Optional[float] = None
    demo: Optional[float] = None
    ts: Optional[datetime] = None
    source_path: Optional[str] = None


@dataclass
class Snapshot:
    equity: pd.DataFrame  # serie principal (observada si existe)
    equity_secondary: pd.DataFrame  # reconstrucción estimada desde CSV
    equity_recent: pd.DataFrame
    hourly_profile: pd.DataFrame
    cards: Dict[str, str]
    metrics: Dict[str, float]
    warnings: List[str]
    view: str
    update_ts: datetime


class SmartDateAxis(pg.DateAxisItem):
    """Eje temporal más legible: HH:MM en ventanas cortas, fecha+hora en ventanas largas."""

    def tickStrings(self, values, scale, spacing):
        if not values:
            return []
        span = max(values) - min(values)
        labels = []
        for v in values:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
            if span <= 36 * 3600:
                labels.append(dt.strftime("%H:%M"))
            else:
                labels.append(dt.strftime("%d-%m %H:%M"))
        return labels


class DataEngine:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.last_live = LiveBalance()
        self.last_observed_trace = ""

    def discover_csv(self) -> List[Path]:
        files = sorted([Path(p) for p in glob.glob(str(self.base_dir / CSV_PATTERN))])
        return files

    def _read_csv_robust(self, path: Path, warnings: List[str]) -> pd.DataFrame:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception as exc:
                last_err = exc
        warnings.append(f"CSV corrupto/no legible: {path.name} ({last_err})")
        return pd.DataFrame()

    def _parse_timestamp(self, df: pd.DataFrame, warnings: List[str]) -> pd.Series:
        ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        if "fecha" in df.columns:
            t = pd.to_datetime(df["fecha"], errors="coerce", utc=True)
            ts = ts.fillna(t)
        if "ts" in df.columns:
            t = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            ts = ts.fillna(t)
        if "epoch" in df.columns:
            e = pd.to_numeric(df["epoch"], errors="coerce")
            t = pd.to_datetime(e, errors="coerce", utc=True, unit="s")
            ts = ts.fillna(t)
        if ts.isna().all():
            warnings.append("No se pudo parsear timestamp (fecha/ts/epoch).")
        return ts

    def _read_live_balance(self, warnings: List[str]) -> LiveBalance:
        candidates = [self.base_dir / LOG_SALDOS]
        candidates.extend(self.base_dir.glob("*.log"))
        candidates.extend(self.base_dir.glob("*.txt"))

        for p in candidates:
            if not p.exists() or not p.is_file():
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            real = None
            demo = None
            rr = re.findall(r"SALDO\s+EN\s+CUENTA\s+REAL\s+DERIV\s*:\s*([-\d\.,]+)(?:\s*USD)?", txt, flags=re.IGNORECASE)
            if not rr:
                rr = re.findall(r"Saldo\s+cuenta\s+REAL(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", txt, flags=re.IGNORECASE)
            dr = re.findall(r"Saldo\s+cuenta\s+DEMO(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", txt, flags=re.IGNORECASE)
            if rr:
                real = _safe_float(rr[-1])
            if dr:
                demo = _safe_float(dr[-1])
            if real is not None or demo is not None:
                dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                return LiveBalance(real=real, demo=demo, ts=dt, source_path=str(p))

        warnings.append("Saldo LIVE no detectado (LOG_SALDOS/log/txt opcional).")
        return LiveBalance()

    def _extract_timestamp_from_line(self, line: str) -> Optional[datetime]:
        patterns = [
            r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
            r"(\d{2}/\d{2}/\d{4}[ T]\d{2}:\d{2}:\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                try:
                    dt = pd.to_datetime(m.group(1), errors="coerce", utc=True)
                    if pd.notna(dt):
                        return dt.to_pydatetime()
                except Exception:
                    pass
        return None

    def _read_observed_balance_series(self, view: str, warnings: List[str]) -> pd.DataFrame:
        if view == "ALL":
            # ALL se construye combinando REAL + DEMO observados si existen.
            real = self._read_observed_balance_series("REAL", warnings)
            demo = self._read_observed_balance_series("DEMO", warnings)
            if real.empty and demo.empty:
                return pd.DataFrame(columns=["timestamp", "equity"])
            if real.empty:
                return demo.rename(columns={"equity": "equity"})[["timestamp", "equity"]]
            if demo.empty:
                return real.rename(columns={"equity": "equity"})[["timestamp", "equity"]]
            rr = real.rename(columns={"equity": "real"}).sort_values("timestamp")
            dd = demo.rename(columns={"equity": "demo"}).sort_values("timestamp")
            z = pd.merge_asof(rr, dd, on="timestamp", direction="nearest", tolerance=pd.Timedelta("12h"))
            z["real"] = z["real"].ffill()
            z["demo"] = z["demo"].ffill()
            z["equity"] = (z["real"].fillna(0.0) + z["demo"].fillna(0.0)).astype(float)
            return z[["timestamp", "equity"]].dropna(subset=["timestamp"])

        tag = "REAL" if view == "REAL" else "DEMO"
        if tag == "REAL":
            patterns = [
                re.compile(r"SALDO\s+EN\s+CUENTA\s+REAL\s+DERIV\s*:\s*([-\d\.,]+)(?:\s*USD)?", flags=re.IGNORECASE),
                re.compile(r"Saldo\s+cuenta\s+REAL(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", flags=re.IGNORECASE),
            ]
        else:
            patterns = [
                re.compile(r"Saldo\s+cuenta\s+DEMO(?:\s*\([^)]*\))?\s*:\s*([-\d\.,]+)(?:\s*USD)?", flags=re.IGNORECASE),
            ]

        candidates = [self.base_dir / LOG_SALDOS]
        candidates.extend(sorted(self.base_dir.glob("*.log")))
        candidates.extend(sorted(self.base_dir.glob("*.txt")))

        rows: List[Tuple[datetime, float]] = []
        file_hits: Dict[str, int] = {}
        for p in candidates:
            if not p.exists() or not p.is_file():
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            if not lines:
                continue
            base_dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) - timedelta(seconds=len(lines))
            for idx, line in enumerate(lines):
                m = None
                for pat in patterns:
                    m = pat.search(line)
                    if m:
                        break
                if m is None:
                    continue
                val = _safe_float(m.group(1))
                if val is None or (isinstance(val, float) and not np.isfinite(val)):
                    continue
                ts = self._extract_timestamp_from_line(line)
                if ts is None:
                    ts = base_dt + timedelta(seconds=idx)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                rows.append((ts.astimezone(timezone.utc), float(val)))
                file_hits[p.name] = file_hits.get(p.name, 0) + 1

        if not rows:
            self.last_observed_trace = f"{tag}: 0 coincidencias en {len(candidates)} archivos revisados"
            return pd.DataFrame(columns=["timestamp", "equity"])

        self.last_observed_trace = f"{tag}: {len(rows)} coincidencias ({file_hits})"
        d = pd.DataFrame(rows, columns=["timestamp", "equity"]).sort_values("timestamp")
        d = d.drop_duplicates(subset=["timestamp", "equity"], keep="last").reset_index(drop=True)
        return d

    def _read_retiros(self, warnings: List[str]) -> pd.DataFrame:
        p = self.base_dir / "retiros.csv"
        if not p.exists():
            return pd.DataFrame(columns=["fecha", "monto", "comentario"])
        try:
            d = pd.read_csv(p)
            if "fecha" in d.columns:
                d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce", utc=True)
            else:
                d["fecha"] = pd.NaT
            d["monto"] = pd.to_numeric(d.get("monto"), errors="coerce").fillna(0.0)
            d["comentario"] = d.get("comentario", "").astype(str)
            return d[["fecha", "monto", "comentario"]]
        except Exception as exc:
            warnings.append(f"No se pudo leer retiros.csv: {exc}")
            return pd.DataFrame(columns=["fecha", "monto", "comentario"])

    def _build_equity(self, d: pd.DataFrame, view: str, warnings: List[str]) -> pd.DataFrame:
        if d.empty:
            return pd.DataFrame(columns=["timestamp", "equity", "pnl"])
        if "ganancia_perdida" not in d.columns:
            warnings.append("Falta columna crítica ganancia_perdida; equity histórica limitada.")
            return pd.DataFrame(columns=["timestamp", "equity", "pnl"])

        x = d.copy()
        x["timestamp"] = self._parse_timestamp(x, warnings)
        x["pnl"] = pd.to_numeric(x["ganancia_perdida"], errors="coerce").fillna(0.0)

        if "trade_status" in x.columns:
            st = x["trade_status"].astype(str).str.upper().str.strip()
            x = x[st == "CERRADO"]

        if "cuenta" in x.columns:
            cuenta = x["cuenta"].astype(str).str.upper().str.strip()
        elif "modo" in x.columns:
            cuenta = x["modo"].astype(str).str.upper().str.strip()
        else:
            cuenta = pd.Series("REAL", index=x.index)
            warnings.append("Sin columna cuenta/modo: se asume REAL.")

        x["cuenta_norm"] = cuenta.replace({"TOTAL": "ALL"})
        x = x.dropna(subset=["timestamp"]).sort_values("timestamp")
        x = x.drop_duplicates(subset=["timestamp", "pnl", "cuenta_norm"], keep="last")

        if view in ("REAL", "DEMO"):
            y = x[x["cuenta_norm"].str.contains(view, na=False)].copy()
            base = SALDO_INICIAL_REAL if view == "REAL" else SALDO_INICIAL_DEMO
            y["equity"] = base + y["pnl"].cumsum()
            return y[["timestamp", "equity", "pnl"]]

        r = x[x["cuenta_norm"].str.contains("REAL", na=False)].copy()
        dmo = x[x["cuenta_norm"].str.contains("DEMO", na=False)].copy()
        r["equity_real"] = SALDO_INICIAL_REAL + r["pnl"].cumsum()
        dmo["equity_demo"] = SALDO_INICIAL_DEMO + dmo["pnl"].cumsum()
        r = r[["timestamp", "equity_real"]]
        dmo = dmo[["timestamp", "equity_demo"]]
        z = pd.merge_asof(
            r.sort_values("timestamp"),
            dmo.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("12h"),
        )
        z["equity_real"] = z["equity_real"].ffill().fillna(SALDO_INICIAL_REAL)
        z["equity_demo"] = z["equity_demo"].ffill().fillna(SALDO_INICIAL_DEMO)
        z["equity"] = z["equity_real"] + z["equity_demo"]
        z["pnl"] = z["equity"].diff().fillna(0.0)
        return z[["timestamp", "equity", "pnl"]]

    def build_snapshot(self, view: str) -> Snapshot:
        warnings: List[str] = []
        files = self.discover_csv()
        if not files:
            warnings.append(f"No se encontraron CSV {CSV_PATTERN} en {self.base_dir}.")
            empty = pd.DataFrame(columns=["timestamp", "equity", "pnl"])
            observed = self._read_observed_balance_series(view, warnings)
            primary = observed if not observed.empty else empty
            recent = primary[primary["timestamp"] >= (_now_utc() - timedelta(hours=VENTANA_HORAS))].copy() if not primary.empty else primary
            return Snapshot(primary, empty, recent, pd.DataFrame(), {}, {}, warnings, view, _now_utc())

        dfs = [self._read_csv_robust(p, warnings) for p in files]
        data = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if dfs else pd.DataFrame()

        eq_est = self._build_equity(data, view=view, warnings=warnings)
        eq_est = eq_est.sort_values("timestamp") if not eq_est.empty else eq_est
        observed = self._read_observed_balance_series(view, warnings)

        live = self._read_live_balance(warnings)
        self.last_live = live

        now = _now_utc()
        rec_start = now - timedelta(hours=VENTANA_HORAS)

        source = "ESTIMADO"
        primary = eq_est[["timestamp", "equity"]].copy() if not eq_est.empty else pd.DataFrame(columns=["timestamp", "equity"])
        observed_reliable = len(observed) >= 1
        if observed_reliable:
            primary = observed.copy()
            source = "REAL OBSERVADO"
            age = (now - observed["timestamp"].max().to_pydatetime()).total_seconds()
            if age > STALE_SALDO_SEGUNDOS:
                source = "STALE"
            if len(observed) < 3:
                warnings.append("Historial observado parcial (pocos puntos), pero se prioriza SALDO OBSERVADO.")
        else:
            warnings.append(f"Sin coincidencias observadas para {view}. Trace: {self.last_observed_trace}")

        live_val = live.real if view == "REAL" else (live.demo if view == "DEMO" else None)
        if source == "ESTIMADO" and live_val is not None and np.isfinite(live_val):
            ts_live = live.ts if live.ts else now
            primary = pd.DataFrame([{"timestamp": ts_live, "equity": float(live_val)}])
            source = "LIVE"

        eq_recent = primary[primary["timestamp"] >= rec_start].copy() if not primary.empty else primary.copy()

        hourly = pd.DataFrame(columns=["hour", "delta_sum"])
        cards: Dict[str, str] = {}
        metrics: Dict[str, float] = {}

        if not primary.empty:
            s = primary.set_index("timestamp")["equity"].sort_index()
            saldo_principal = float(s.iloc[-1])
            s_recent = s[s.index >= rec_start]
            chg1h = float(s_recent.iloc[-1] - s_recent.iloc[0]) if len(s_recent) > 1 else 0.0
            estimado = float(eq_est["equity"].iloc[-1]) if not eq_est.empty else np.nan

            h = primary.copy().sort_values("timestamp")
            h["delta"] = h["equity"].diff().fillna(0.0)
            h["hour"] = h["timestamp"].dt.hour
            hourly = h.groupby("hour", as_index=False).agg(delta_sum=("delta", "sum"))

            cards = {
                "SALDO ACTUAL": _fmt_money(saldo_principal),
                "FUENTE SALDO": source,
                "VISTA": view,
                "ÚLTIMA ACTUALIZACIÓN": s.index[-1].strftime("%Y-%m-%d %H:%M:%S UTC"),
                "HORA ACTUAL": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "SALDO ESTIMADO (CSV)": _fmt_money(estimado),
            }
            metrics = {
                "current_display": saldo_principal,
            }
        else:
            warnings.append("Sin serie de saldo observado ni equity estimado para graficar.")

        if source == "ESTIMADO" and not eq_est.empty:
            warnings.append("Sin saldo observado ni LIVE suficiente: mostrando ESTIMADO desde CSV.")
        elif source == "LIVE":
            warnings.append("Usando LIVE por ausencia de historial observado suficiente.")
        else:
            warnings.append(f"Trace observado activo: {self.last_observed_trace}")

        return Snapshot(primary, eq_est, eq_recent, hourly, cards, metrics, warnings, view, now)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.t = QtWidgets.QLabel(title)
        self.t.setObjectName("CardTitle")
        self.v = QtWidgets.QLabel("--")
        self.v.setObjectName("CardValue")
        lay.addWidget(self.t)
        lay.addWidget(self.v)

    def set_value(self, value: str):
        self.v.setText(value)


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: DataEngine):
        super().__init__()
        self.engine = engine
        self.view = CUENTA_OBJETIVO if CUENTA_OBJETIVO in ("REAL", "DEMO", "ALL") else "REAL"
        self.paused = False
        self.setWindowTitle("Monitor Saldo Pro - Dashboard Equity Tiempo Real")
        self.resize(1700, 980)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.balance_title = QtWidgets.QLabel("SALDO REAL DERIV ACTUAL")
        self.balance_title.setObjectName("BalanceTitle")
        self.balance_value = QtWidgets.QLabel("--")
        self.balance_value.setObjectName("BalanceValue")
        self.status_line = QtWidgets.QLabel("Estado: LIVE")
        self.status_line.setObjectName("StatusLine")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.balance_title, 0)
        top.addWidget(self.balance_value, 1)
        top.addWidget(self.status_line, 1)
        root.addLayout(top)

        self.cards_widget = QtWidgets.QWidget()
        self.cards_grid = QtWidgets.QGridLayout(self.cards_widget)
        self.cards_grid.setContentsMargins(0, 0, 0, 0)
        self.cards_grid.setHorizontalSpacing(6)
        self.cards_grid.setVerticalSpacing(6)
        root.addWidget(self.cards_widget)

        self.card_order = [
            "FUENTE SALDO", "VISTA", "HORA ACTUAL", "ÚLTIMA ACTUALIZACIÓN", "SALDO ESTIMADO (CSV)"
        ]
        self.cards: Dict[str, MetricCard] = {}
        for i, k in enumerate(self.card_order):
            card = MetricCard(k)
            self.cards[k] = card
            r, c = divmod(i, 5)
            self.cards_grid.addWidget(card, r, c)

        self.graphics = pg.GraphicsLayoutWidget()
        root.addWidget(self.graphics, 1)

        axis_main = SmartDateAxis(orientation="bottom")
        self.p_main = self.graphics.addPlot(row=0, col=0, colspan=2, axisItems={"bottom": axis_main})
        self.p_main.setTitle("Tiempo vs Dinero (USD)")
        self.p_main.setLabel("left", "Dinero (USD)")
        self.p_main.showGrid(x=True, y=True, alpha=0.08)

        axis_recent = SmartDateAxis(orientation="bottom")
        self.p_recent = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": axis_recent})
        self.p_recent.setTitle(f"Últimas {VENTANA_HORAS} horas")
        self.p_recent.setLabel("left", "Dinero (USD)")
        self.p_recent.showGrid(x=True, y=True, alpha=0.08)

        self.p_hour = self.graphics.addPlot(row=1, col=1)
        self.p_hour.setTitle("Comportamiento por Hora del Día")
        self.p_hour.showGrid(x=True, y=True, alpha=0.08)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setObjectName("WarningLabel")
        root.addWidget(self.warning_label)

        self.help_label = QtWidgets.QLabel(
            "Teclas: [1] REAL  [2] DEMO  [3] TOTAL  [F] Fullscreen  [P] Pausa  [R] Reset Zoom  [Q] Salir"
        )
        self.help_label.setObjectName("HelpLabel")
        root.addWidget(self.help_label)

        self._apply_style()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(max(1000, int(REFRESH_SEGUNDOS * 1000)))

        if FULLSCREEN_INICIAL:
            self.showMaximized()

        self.startup_check()
        self.update_dashboard()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b0f14; color: #d9e2f2; }
            #BalanceTitle { font-size: 16px; color: #8aa4cf; font-weight: 600; }
            #BalanceValue { font-size: 44px; color: #e8f1ff; font-weight: 700; letter-spacing: 0.5px; }
            #StatusLine { font-size: 14px; color: #a7c0e8; }
            #MetricCard { background: #121a24; border: 1px solid #223043; border-radius: 8px; }
            #CardTitle { color: #88a4cc; font-size: 11px; font-weight: 600; }
            #CardValue { color: #e5f0ff; font-size: 17px; font-weight: 700; }
            #WarningLabel { color: #ffb86b; font-size: 12px; }
            #HelpLabel { color: #9fb7d9; font-size: 11px; }
            """
        )
        pg.setConfigOptions(antialias=True, background="#0b0f14", foreground="#d9e2f2")

    def startup_check(self):
        files = self.engine.discover_csv()
        log_exists = any([(self.engine.base_dir / LOG_SALDOS).exists(), *[bool(list(self.engine.base_dir.glob(ext))) for ext in ("*.log", "*.txt")]])
        ret_exists = (self.engine.base_dir / "retiros.csv").exists()
        print("=" * 80)
        print("INICIO SEGURO DASHBOARD")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Vista inicial: {self.view}")
        print(f"CSV encontrados ({len(files)}): {[p.name for p in files[:8]]}{' ...' if len(files) > 8 else ''}")
        print(f"LOG_SALDOS/log/txt detectado: {'SI' if log_exists else 'NO (opcional)'}")
        print(f"retiros.csv detectado: {'SI' if ret_exists else 'NO (opcional)'}")
        print("=" * 80)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k == QtCore.Qt.Key_1:
            self.view = "REAL"
            self.update_dashboard(force=True)
        elif k == QtCore.Qt.Key_2:
            self.view = "DEMO"
            self.update_dashboard(force=True)
        elif k == QtCore.Qt.Key_3:
            self.view = "ALL"
            self.update_dashboard(force=True)
        elif k == QtCore.Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif k == QtCore.Qt.Key_P:
            self.paused = not self.paused
            self.update_dashboard(force=True)
        elif k == QtCore.Qt.Key_R:
            self.p_main.enableAutoRange()
            self.p_recent.enableAutoRange()
            self.p_hour.enableAutoRange()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(ev)

    def _plot_main(self, snap: Snapshot):
        self.p_main.clear()
        if snap.equity.empty:
            self.p_main.setTitle("Curva Principal Saldo Observado / Estimado (sin datos)")
            return
        x = (snap.equity["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = snap.equity["equity"].to_numpy(dtype=float)

        curve = self.p_main.plot(x, y, pen=pg.mkPen("#5ad4ff", width=2.5), name="Equity")
        baseline = np.full_like(y, y.min())
        curve0 = self.p_main.plot(x, baseline, pen=pg.mkPen(None))
        fill = pg.FillBetweenItem(curve, curve0, brush=pg.mkBrush(90, 180, 255, 45))
        self.p_main.addItem(fill)

        self.p_main.plot([x[-1]], [y[-1]], pen=None, symbol="o", symbolSize=9, symbolBrush="#c6f56e")
        txt = pg.TextItem(f"{y[-1]:,.2f}", color="#e8f1ff", anchor=(0, 1))
        txt.setPos(float(x[-1]), float(y[-1]))
        self.p_main.addItem(txt)

        if len(x) > 2:
            span_target = DIAS_GRAFICA * 24 * 3600
            right = float(x[-1])
            left = max(float(x[0]), right - span_target)
            self.p_main.setXRange(left, right, padding=0.02)

        if not snap.equity_secondary.empty:
            xs = (snap.equity_secondary["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
            ys = snap.equity_secondary["equity"].to_numpy(dtype=float)
            self.p_main.plot(xs, ys, pen=pg.mkPen("#d6d6d6", width=1.2, style=QtCore.Qt.DashLine))

    def _plot_recent(self, snap: Snapshot):
        self.p_recent.clear()
        if snap.equity_recent.empty:
            return
        x = (snap.equity_recent["timestamp"].astype("int64") / 1e9).to_numpy(dtype=float)
        y = snap.equity_recent["equity"].to_numpy(dtype=float)
        self.p_recent.plot(x, y, pen=pg.mkPen("#77e3ff", width=2.0))
        if len(x) > 1:
            self.p_recent.setXRange(float(x[0]), float(x[-1]), padding=0.02)

        piso = snap.metrics.get("piso_operativo")
        techo = snap.metrics.get("techo_reciente")
        if piso is not None:
            self.p_recent.addItem(pg.InfiniteLine(piso, angle=0, pen=pg.mkPen("#ff9f50", width=1.5, style=QtCore.Qt.DashLine)))
        if techo is not None:
            self.p_recent.addItem(pg.InfiniteLine(techo, angle=0, pen=pg.mkPen("#5df2a3", width=1.0, style=QtCore.Qt.DotLine)))
        if piso is not None and techo is not None and techo > piso:
            band = QtWidgets.QGraphicsRectItem(float(x.min()), piso, float(x.max() - x.min()), float(techo - piso))
            band.setBrush(pg.mkBrush(70, 120, 80, 25))
            band.setPen(pg.mkPen(None))
            self.p_recent.addItem(band)

    def _plot_hourly(self, snap: Snapshot):
        self.p_hour.clear()
        if snap.hourly_profile.empty:
            return
        h = snap.hourly_profile.sort_values("hour")
        x = h["hour"].to_numpy(dtype=float)
        y = h["delta_sum"].to_numpy(dtype=float)
        brushes = [pg.mkBrush("#4cd964" if v >= 0 else "#ff5c5c") for v in y]
        bars = pg.BarGraphItem(x=x, height=y, width=0.8, brushes=brushes)
        self.p_hour.addItem(bars)
        self.p_hour.plot(x, pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy(), pen=pg.mkPen("#78b8ff", width=2))

    def update_dashboard(self, force: bool = False):
        if self.paused and not force:
            return
        try:
            snap = self.engine.build_snapshot(self.view)
            saldo_txt = snap.cards.get("SALDO ACTUAL", "--")
            self.balance_value.setText(saldo_txt)

            src = snap.cards.get("FUENTE SALDO", "ESTIMADO")
            refresh_state = "PAUSADO" if self.paused else "ACTIVO"
            self.status_line.setText(
                f"REFRESCO: {refresh_state} | FUENTE SALDO: {src} | HORA ACTUAL: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
            )

            for k, card in self.cards.items():
                card.set_value(snap.cards.get(k, "--"))

            self._plot_main(snap)
            self._plot_recent(snap)
            self._plot_hourly(snap)

            if snap.warnings:
                self.warning_label.setText("⚠ " + " | ".join(snap.warnings[:4]))
            else:
                self.warning_label.setText("")

        except Exception as exc:
            self.warning_label.setText(f"⚠ Error de actualización (monitor sigue vivo): {exc}")
            traceback.print_exc()


def main():
    app = QtWidgets.QApplication(sys.argv)
    engine = DataEngine(base_dir=Path(__file__).resolve().parent)
    w = DashboardWindow(engine)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
