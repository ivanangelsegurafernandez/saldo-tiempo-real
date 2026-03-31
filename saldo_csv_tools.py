from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

DISPLAY_TIMEZONE = "America/Lima"
_RAW_HEADERS = [
    "ts_utc",
    "ts_lima",
    "epoch",
    "saldo_real",
    "status",
    "source",
    "event_type",
    "saldo_delta",
    "sample_id",
]
_AGG_HEADERS = ["saldo_open", "saldo_high", "saldo_low", "saldo_close", "saldo_mean", "samples"]


def _tz_lima():
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(DISPLAY_TIMEZONE)
    except Exception:
        return timezone.utc


def _to_utc_dt(ts_utc: Optional[str], epoch: Optional[float]) -> datetime:
    if ts_utc:
        try:
            dt = datetime.fromisoformat(str(ts_utc).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def build_sample_id(epoch: float, saldo_real: float, status: str, source: str) -> str:
    key = f"{float(epoch):.6f}|{float(saldo_real):.8f}|{str(status).strip().upper()}|{str(source).strip().upper()}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _read_last_valid_row(path: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return None
    last_row = None
    try:
        with open(path, "r", encoding="utf-8", newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if row.get("sample_id"):
                    last_row = row
    except Exception:
        return None
    return last_row


def append_raw_balance_sample(
    raw_csv_path: str,
    *,
    ts_utc: str,
    epoch: float,
    saldo_real: Optional[float],
    status: str,
    source: str,
    event_type: str,
) -> Tuple[bool, Optional[str], Dict[str, object]]:
    """Append a sample to raw CSV if not duplicate by last sample_id."""
    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    if saldo_real is None:
        return False, "saldo_real_none", {}

    dt_utc = _to_utc_dt(ts_utc, epoch)
    dt_lima = dt_utc.astimezone(_tz_lima())
    epoch_f = float(epoch)
    saldo_f = float(saldo_real)
    sid = build_sample_id(epoch_f, saldo_f, status, source)

    last_row = _read_last_valid_row(raw_csv_path)
    if isinstance(last_row, dict) and str(last_row.get("sample_id", "")).strip() == sid:
        return False, "duplicate_last_sample_id", {"sample_id": sid}

    last_saldo = None
    if isinstance(last_row, dict):
        try:
            last_saldo = float(last_row.get("saldo_real"))
        except Exception:
            last_saldo = None
    delta = 0.0 if last_saldo is None else float(saldo_f - last_saldo)

    row = {
        "ts_utc": dt_utc.isoformat(),
        "ts_lima": dt_lima.isoformat(),
        "epoch": f"{epoch_f:.6f}",
        "saldo_real": f"{saldo_f:.10f}",
        "status": str(status),
        "source": str(source),
        "event_type": str(event_type),
        "saldo_delta": f"{delta:.10f}",
        "sample_id": sid,
    }

    write_header = (not os.path.exists(raw_csv_path)) or os.path.getsize(raw_csv_path) <= 0
    try:
        with open(raw_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_RAW_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        # fallback: recreate with header and append current row
        try:
            with open(raw_csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_RAW_HEADERS)
                writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            return False, f"write_error:{e}", row

    return True, None, row


def _load_raw_rows(raw_csv_path: str) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    if not os.path.exists(raw_csv_path):
        return out
    try:
        with open(raw_csv_path, "r", encoding="utf-8", newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    saldo = float(row.get("saldo_real"))
                except Exception:
                    continue
                dt_utc = _to_utc_dt(row.get("ts_utc"), row.get("epoch"))
                out.append((dt_utc, saldo))
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out


def _build_agg_rows(rows: List[Tuple[datetime, float]], period: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[float]] = {}
    for dt_utc, saldo in rows:
        dt_local = dt_utc.astimezone(_tz_lima())
        if period == "minute":
            key_dt = dt_local.replace(second=0, microsecond=0)
            key = key_dt.strftime("%Y-%m-%d %H:%M:00")
        else:
            key_dt = dt_local.replace(minute=0, second=0, microsecond=0)
            key = key_dt.strftime("%Y-%m-%d %H:00:00")
        grouped.setdefault(key, []).append(float(saldo))

    out: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        vals = grouped[key]
        n = len(vals)
        if n <= 0:
            continue
        out.append(
            {
                ("minute_lima" if period == "minute" else "hour_lima"): key,
                "saldo_open": f"{vals[0]:.10f}",
                "saldo_high": f"{max(vals):.10f}",
                "saldo_low": f"{min(vals):.10f}",
                "saldo_close": f"{vals[-1]:.10f}",
                "saldo_mean": f"{(sum(vals)/n):.10f}",
                "samples": str(n),
            }
        )
    return out


def _write_agg_csv(path: str, rows: List[Dict[str, object]], time_col: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[time_col] + _AGG_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def refresh_aggregate_csvs(
    raw_csv_path: str,
    min_csv_path: str,
    hour_csv_path: str,
    cache_state: Optional[Dict[str, object]] = None,
) -> Tuple[bool, str, Dict[str, object]]:
    state = cache_state if isinstance(cache_state, dict) else {}
    try:
        st = os.stat(raw_csv_path)
        sig = (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        sig = None
    if sig is None:
        return False, "raw_missing", state
    if state.get("raw_sig") == sig and os.path.exists(min_csv_path) and os.path.exists(hour_csv_path):
        return True, "unchanged", state

    rows = _load_raw_rows(raw_csv_path)
    if not rows:
        return False, "raw_empty_or_invalid", state

    _write_agg_csv(min_csv_path, _build_agg_rows(rows, "minute"), "minute_lima")
    _write_agg_csv(hour_csv_path, _build_agg_rows(rows, "hour"), "hour_lima")
    state["raw_sig"] = sig
    state["raw_rows"] = len(rows)
    return True, "rebuilt", state
