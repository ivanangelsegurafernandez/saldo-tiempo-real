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
FULLSCREEN_INICIAL = True
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
    equity: pd.DataFrame
    equity_recent: pd.DataFrame
    hourly_profile: pd.DataFrame
    daily_close: pd.DataFrame
    cards: Dict[str, str]
    metrics: Dict[str, float]
    warnings: List[str]
    view: str
    update_ts: datetime


class DataEngine:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.last_live = LiveBalance()

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
            t = pd.to_datetime(df["fecha"], errors="coerce", utc=True, infer_datetime_format=True)
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
            rr = re.findall(r"Saldo\s+cuenta\s+REAL\s*:\s*([\d\.,]+)", txt, flags=re.IGNORECASE)
            dr = re.findall(r"Saldo\s+cuenta\s+DEMO\s*:\s*([\d\.,]+)", txt, flags=re.IGNORECASE)
            if rr:
                real = _safe_float(rr[-1])
            if dr:
                demo = _safe_float(dr[-1])
            if real is not None or demo is not None:
                dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                return LiveBalance(real=real, demo=demo, ts=dt, source_path=str(p))

        warnings.append("Saldo LIVE no detectado (LOG_SALDOS/log/txt opcional).")
        return LiveBalance()

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
            return Snapshot(empty, empty, empty, empty, {}, {}, warnings, view, _now_utc())

        dfs = [self._read_csv_robust(p, warnings) for p in files]
        data = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if dfs else pd.DataFrame()

        eq = self._build_equity(data, view=view, warnings=warnings)
        eq = eq.sort_values("timestamp") if not eq.empty else eq

        live = self._read_live_balance(warnings)
        self.last_live = live
        retiros = self._read_retiros(warnings)

        now = _now_utc()
        rec_start = now - timedelta(hours=VENTANA_HORAS)
        eq_recent = eq[eq["timestamp"] >= rec_start].copy() if not eq.empty else eq.copy()

        hourly = pd.DataFrame(columns=["hour", "pnl_mean", "pnl_sum"])
        daily = pd.DataFrame(columns=["date", "close"])
        cards: Dict[str, str] = {}
        metrics: Dict[str, float] = {}

        if not eq.empty:
            s = eq.set_index("timestamp")["equity"]
            peak = float(s.max())
            trough_recent = float(eq_recent["equity"].min()) if not eq_recent.empty else float(s.min())
            current_recon = float(s.iloc[-1])
            running_max = s.cummax()
            dd = ((s - running_max) / running_max.replace(0, np.nan) * 100).fillna(0)
            max_dd = float(dd.min()) if not dd.empty else 0.0

            h1 = now - timedelta(hours=1)
            d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
            s_h1 = s[s.index >= h1]
            s_d = s[s.index >= d0]
            chg1h = float(s_h1.iloc[-1] - s_h1.iloc[0]) if len(s_h1) > 1 else 0.0
            chgday = float(s_d.iloc[-1] - s_d.iloc[0]) if len(s_d) > 1 else 0.0

            pnl_by_trade = eq[["timestamp", "pnl"]].copy()
            today = pnl_by_trade[pnl_by_trade["timestamp"] >= d0]
            trades_today = int(len(today))
            winrate_today = float((today["pnl"] > 0).mean() * 100) if trades_today > 0 else np.nan

            h = eq.copy()
            h["hour"] = h["timestamp"].dt.hour
            hourly = h.groupby("hour", as_index=False).agg(pnl_mean=("pnl", "mean"), pnl_sum=("pnl", "sum"))
            if not hourly.empty:
                best_hour = int(hourly.loc[hourly["pnl_sum"].idxmax(), "hour"])
                worst_hour = int(hourly.loc[hourly["pnl_sum"].idxmin(), "hour"])
            else:
                best_hour = worst_hour = -1

            dly = eq.copy()
            dly["date"] = dly["timestamp"].dt.floor("D")
            daily = dly.groupby("date", as_index=False).agg(close=("equity", "last"))
            daily = daily.tail(DIAS_GRAFICA)

            piso_operativo = trough_recent
            colchon = max(0.0, current_recon - piso_operativo)
            volatilidad = float(eq_recent["equity"].std()) if len(eq_recent) > 2 else 0.0
            retiro_sugerido = max(0.0, colchon - (volatilidad * MARGEN_SEGURIDAD_COLCHON))

            if len(eq_recent) > 10:
                y = eq_recent["equity"].to_numpy()
                xline = np.arange(len(y), dtype=float)
                slope = np.polyfit(xline, y, 1)[0]
                norm = np.std(y) + 1e-6
                score = slope / norm
                trend = "SUBIENDO" if score > 0.08 else ("BAJANDO" if score < -0.08 else "LATERAL")
            else:
                trend = "LATERAL"

            draw_recent = 0.0
            if not eq_recent.empty:
                rs = eq_recent.set_index("timestamp")["equity"]
                rm = rs.cummax()
                draw_recent = float((((rs - rm) / rm.replace(0, np.nan)) * 100).min())

            if colchon <= 0:
                estado = "RIESGO"
            elif draw_recent < -4:
                estado = "TENSO"
            elif retiro_sugerido > 0 and draw_recent > -2:
                estado = "RETIRABLE"
            elif draw_recent > -1.5:
                estado = "ESTABLE"
            else:
                estado = "CONSERVADORA"

            calidad = "NEUTRAL"
            if chg1h > 0 and chgday > 0 and colchon > volatilidad:
                calidad = "MOMENTO FUERTE"
            elif chg1h < 0 and chgday < 0:
                calidad = "MOMENTO DÉBIL"

            live_val = live.real if view == "REAL" else (live.demo if view == "DEMO" else None)
            source = "RECONSTRUIDO"
            saldo_grande = current_recon
            if live_val is not None:
                age = (now - live.ts).total_seconds() if live.ts else 1e9
                if age <= STALE_SALDO_SEGUNDOS:
                    source = "LIVE"
                    saldo_grande = live_val
                else:
                    source = "STALE"
                    saldo_grande = live_val

            diff_live_eq = (live_val - current_recon) if live_val is not None else np.nan
            retiros_total = float(retiros["monto"].sum()) if not retiros.empty else 0.0
            saldo_post_retiros = current_recon - retiros_total

            cards = {
                "SALDO ACTUAL": _fmt_money(saldo_grande),
                "ORIGEN": source,
                "VISTA": view,
                "PICO HISTÓRICO": _fmt_money(peak),
                f"PISO {VENTANA_HORAS}H": _fmt_money(piso_operativo),
                "DRAWDOWN MÁX": _fmt_pct(max_dd),
                "CAMBIO 1H": _fmt_money(chg1h),
                "CAMBIO DÍA": _fmt_money(chgday),
                "TRADES HOY": str(trades_today),
                "WINRATE HOY": _fmt_pct(winrate_today),
                "MEJOR HORA": f"{best_hour:02d}:00" if best_hour >= 0 else "--",
                "PEOR HORA": f"{worst_hour:02d}:00" if worst_hour >= 0 else "--",
                "TENDENCIA": trend,
                "COLCHÓN RETIRABLE": _fmt_money(retiro_sugerido),
                "ZONA ACTUAL": estado,
                "DIF LIVE-EQ": _fmt_money(diff_live_eq),
                "CALIDAD MOMENTO": calidad,
                "RETIROS ACUM": _fmt_money(retiros_total),
                "SALDO POST-RETIROS": _fmt_money(saldo_post_retiros),
            }

            metrics = {
                "current_recon": current_recon,
                "current_display": saldo_grande,
                "pico": peak,
                "piso_operativo": piso_operativo,
                "retiro_sugerido": retiro_sugerido,
                "techo_reciente": float(eq_recent["equity"].max()) if not eq_recent.empty else peak,
            }

        return Snapshot(eq, eq_recent, hourly, daily, cards, metrics, warnings, view, now)


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

        self.balance_title = QtWidgets.QLabel("SALDO ACTUAL")
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
            "ORIGEN", "VISTA", "PICO HISTÓRICO", f"PISO {VENTANA_HORAS}H", "DRAWDOWN MÁX", "CAMBIO 1H",
            "CAMBIO DÍA", "TRADES HOY", "WINRATE HOY", "MEJOR HORA", "PEOR HORA", "TENDENCIA",
            "COLCHÓN RETIRABLE", "ZONA ACTUAL", "DIF LIVE-EQ", "CALIDAD MOMENTO", "RETIROS ACUM", "SALDO POST-RETIROS"
        ]
        self.cards: Dict[str, MetricCard] = {}
        for i, k in enumerate(self.card_order):
            card = MetricCard(k)
            self.cards[k] = card
            r, c = divmod(i, 6)
            self.cards_grid.addWidget(card, r, c)

        self.graphics = pg.GraphicsLayoutWidget()
        root.addWidget(self.graphics, 1)

        axis_main = pg.DateAxisItem(orientation="bottom")
        self.p_main = self.graphics.addPlot(row=0, col=0, colspan=2, axisItems={"bottom": axis_main})
        self.p_main.setTitle("Curva Principal Equity / Balance")
        self.p_main.showGrid(x=True, y=True, alpha=0.2)

        axis_recent = pg.DateAxisItem(orientation="bottom")
        self.p_recent = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": axis_recent})
        self.p_recent.setTitle(f"Últimas {VENTANA_HORAS} horas")
        self.p_recent.showGrid(x=True, y=True, alpha=0.2)

        self.p_hour = self.graphics.addPlot(row=1, col=1)
        self.p_hour.setTitle("Comportamiento por Hora del Día")
        self.p_hour.showGrid(x=True, y=True, alpha=0.2)

        axis_daily = pg.DateAxisItem(orientation="bottom")
        self.p_daily = self.graphics.addPlot(row=2, col=0, colspan=2, axisItems={"bottom": axis_daily})
        self.p_daily.setTitle("Cierre Diario")
        self.p_daily.showGrid(x=True, y=True, alpha=0.2)

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
            self.showFullScreen()

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
            self.p_daily.enableAutoRange()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(ev)

    def _plot_main(self, snap: Snapshot):
        self.p_main.clear()
        if snap.equity.empty:
            self.p_main.setTitle("Curva Principal Equity / Balance (sin datos)")
            return
        x = snap.equity["timestamp"].astype("int64") / 1e9
        y = snap.equity["equity"].to_numpy(dtype=float)

        curve = self.p_main.plot(x, y, pen=pg.mkPen("#5ad4ff", width=2.5), name="Equity")
        baseline = np.full_like(y, y.min())
        curve0 = self.p_main.plot(x, baseline, pen=pg.mkPen(None))
        fill = pg.FillBetweenItem(curve, curve0, brush=pg.mkBrush(90, 180, 255, 45))
        self.p_main.addItem(fill)

        self.p_main.plot([x.iloc[-1]], [y[-1]], pen=None, symbol="o", symbolSize=9, symbolBrush="#c6f56e")
        txt = pg.TextItem(f"{y[-1]:,.2f}", color="#e8f1ff", anchor=(0, 1))
        txt.setPos(float(x.iloc[-1]), float(y[-1]))
        self.p_main.addItem(txt)

        if len(y) > 2:
            ridx = max(0, len(y) - 300)
            y_recent = y[ridx:]
            x_recent = x.iloc[ridx:]
            i_max = int(np.argmax(y_recent))
            i_min = int(np.argmin(y_recent))
            self.p_main.plot([x_recent.iloc[i_max]], [y_recent[i_max]], pen=None, symbol="t", symbolBrush="#4cff9d", symbolSize=11)
            self.p_main.plot([x_recent.iloc[i_min]], [y_recent[i_min]], pen=None, symbol="t1", symbolBrush="#ff6e6e", symbolSize=11)

    def _plot_recent(self, snap: Snapshot):
        self.p_recent.clear()
        if snap.equity_recent.empty:
            return
        x = snap.equity_recent["timestamp"].astype("int64") / 1e9
        y = snap.equity_recent["equity"].to_numpy(dtype=float)
        self.p_recent.plot(x, y, pen=pg.mkPen("#77e3ff", width=2.0))

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
        y = h["pnl_sum"].to_numpy(dtype=float)
        brushes = [pg.mkBrush("#4cd964" if v >= 0 else "#ff5c5c") for v in y]
        bars = pg.BarGraphItem(x=x, height=y, width=0.8, brushes=brushes)
        self.p_hour.addItem(bars)
        self.p_hour.plot(x, pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy(), pen=pg.mkPen("#78b8ff", width=2))

    def _plot_daily(self, snap: Snapshot):
        self.p_daily.clear()
        if snap.daily_close.empty:
            return
        x = snap.daily_close["date"].astype("int64") / 1e9
        y = snap.daily_close["close"].to_numpy(dtype=float)
        self.p_daily.plot(x, y, pen=pg.mkPen("#ffd166", width=2.2), symbol="o", symbolSize=6, symbolBrush="#ffd166")

    def update_dashboard(self, force: bool = False):
        if self.paused and not force:
            return
        try:
            snap = self.engine.build_snapshot(self.view)
            saldo_txt = snap.cards.get("SALDO ACTUAL", "--")
            self.balance_value.setText(saldo_txt)

            src = snap.cards.get("ORIGEN", "RECONSTRUIDO")
            state = "PAUSADO" if self.paused else "LIVE"
            self.status_line.setText(
                f"Estado: {state} | Última actualización: {snap.update_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | Fuente saldo grande: {src}"
            )

            for k, card in self.cards.items():
                card.set_value(snap.cards.get(k, "--"))

            self._plot_main(snap)
            self._plot_recent(snap)
            self._plot_hourly(snap)
            self._plot_daily(snap)

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
