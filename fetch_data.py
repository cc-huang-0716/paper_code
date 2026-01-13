# -*- coding: utf-8 -*-
"""
fetch_data.py
Download Taiwan stock daily quotes in bulk (loop dates, not tickers).
- TWSE (listed): https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date=YYYYMMDD&type=ALLBUT0999
- TPEx (OTC): parse Daily Stock Quotes page and auto-find "Download CSV" link.

Outputs:
  out_dir/
    twse_quotes_YYYY-MM.parquet
    tpex_quotes_YYYY-MM.parquet
State:
  state.json (for resume)

Usage examples:
  python fetch_data.py --start 2015-01-01 --end 2015-03-31 --out ./data
  python fetch_data.py --start 2015-01-01 --end 2024-12-31 --out ./data --resume
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import requests
import certifi

SSL_VERIFY = False

# -----------------------------
# Helpers
# -----------------------------
def iso_to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def iso_to_yyyymm(d: datetime) -> str:
    return d.strftime("%Y-%m")

def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(path: str, state: dict) -> None:
    # 直接覆寫，不用 tmp + replace（Windows 友善）
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def polite_sleep(base: float = 0.6, jitter: float = 0.6) -> None:
    time.sleep(base + random.random() * jitter)

def to_numeric_clean(series: pd.Series) -> pd.Series:
    # remove commas, whitespace; keep NaN for invalid like "--"
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"--": None, "nan": None, "None": None, "": None})
    return pd.to_numeric(s, errors="coerce")


# -----------------------------
# TWSE (Listed)
# -----------------------------
TWSE_URL = "https://www.twse.com.tw/exchangeReport/MI_INDEX"  # stable usage in many examples :contentReference[oaicite:2]{index=2}

def warmup_twse(session: requests.Session) -> None:
    # Get cookies; helps reduce blocks
    session.get("https://www.twse.com.tw/", headers={"User-Agent": "Mozilla/5.0"}, timeout=30,verify=SSL_VERIFY)

def fetch_twse_payload(date_yyyymmdd: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    params = {
        "response": "json",
        "date": date_yyyymmdd,
        "type": "ALLBUT0999",
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.twse.com.tw/",
        "Connection": "keep-alive",
    }

    r = session.get(TWSE_URL, params=params, headers=headers, timeout=30,verify=SSL_VERIFY)

    if r.status_code != 200:
        print(f"[TWSE] HTTP {r.status_code} {date_yyyymmdd} sample={r.text[:120]!r}")
        return None

    text = (r.text or "").strip()
    if not text:
        print(f"[TWSE] EMPTY {date_yyyymmdd}")
        return None

    # Sometimes HTML is returned (maintenance / block page)
    if text[0] not in "{[":
        print(f"[TWSE] NOT_JSON {date_yyyymmdd} sample={text[:120]!r}")
        return None

    try:
        payload = r.json()
    except Exception:
        print(f"[TWSE] JSON_DECODE_FAIL {date_yyyymmdd} sample={text[:200]!r}")
        return None

    # Non-trading day often has stat != OK
    if str(payload.get("stat", "")).upper() != "OK":
        return None

    return payload

def parse_twse_quotes(payload: Dict[str, Any], date_yyyymmdd: str) -> pd.DataFrame:
    """
    MI_INDEX response has 'tables' list. We pick the table that looks like stock quotes.
    """
    tables = payload.get("tables", [])
    for t in tables:
        fields = t.get("fields", [])
        data = t.get("data", [])
        if ("證券代號" in fields) and ("收盤價" in fields) and len(data) > 200:
            df = pd.DataFrame(data, columns=fields)
            df["date"] = date_yyyymmdd
            # Standardize column names (keep original too if you want)
            # Common fields include: 證券代號, 證券名稱, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌(+/-), 漲跌價差, 成交筆數
            return df

    raise ValueError("TWSE: cannot find quotes table in payload")

def normalize_twse(df: pd.DataFrame) -> pd.DataFrame:
    # Map to a compact schema
    colmap = {
        "證券代號": "stock_id",
        "證券名稱": "name",
        "開盤價": "open",
        "最高價": "high",
        "最低價": "low",
        "收盤價": "close",
        "成交股數": "volume",
        "成交金額": "amount",
        "成交筆數": "trades",
    }
    out = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}).copy()

    # Keep only known columns + date
    keep = ["date", "stock_id", "name", "open", "high", "low", "close", "volume", "amount", "trades"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    # numeric clean
    for c in ["open", "high", "low", "close", "volume", "amount", "trades"]:
        if c in out.columns:
            out[c] = to_numeric_clean(out[c])

    # types: keep RAM small
    if "stock_id" in out.columns:
        out["stock_id"] = out["stock_id"].astype(str)
    for c in ["open", "high", "low", "close", "amount"]:
        if c in out.columns:
            out[c] = out[c].astype("float32")
    for c in ["volume", "trades"]:
        if c in out.columns:
            # volume can be huge; int64 safer
            out[c] = out[c].astype("int64")

    return out


# -----------------------------
# TPEx (OTC) via HTML -> CSV link
# -----------------------------
TPEX_PAGE = "https://www.tpex.org.tw/en/stock/aftertrading/DAILY_CLOSE_quotes/stk_quote.php"  # :contentReference[oaicite:3]{index=3}

def warmup_tpex(session: requests.Session) -> None:
    session.get("https://www.tpex.org.tw/", headers={"User-Agent": "Mozilla/5.0"}, timeout=30,verify=SSL_VERIFY)

def find_csv_link_in_html(html: str) -> Optional[str]:
    """
    TPEx page includes a "Download CSV" link. We try to find a URL that ends with .csv or contains 'download' and 'csv'.
    Because the exact parameterization can change, we search heuristically.
    """
    # Common patterns: href="...csv..." or download links embedded in scripts
    candidates = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    for href in candidates:
        h = href.lower()
        if "csv" in h and ("download" in h or "export" in h or h.endswith(".csv")):
            return href

    # fallback: look for explicit .csv URLs
    m = re.search(r'(https?://[^"\']+?\.csv[^"\']*)', html, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None

def fetch_tpex_csv(date_obj: datetime, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    Fetch TPEx daily close quotes for all OTC stocks for given date.
    We first load the page with date parameter, then parse and follow CSV link.
    """
    # The page has a Date selector; parameters can differ. We'll try a few common ones.
    # If none works, you can still use the page manually to see its querystring.
    date_try_params = [
        {"d": date_obj.strftime("%Y/%m/%d")},  # yyyy/mm/dd
        {"date": date_obj.strftime("%Y/%m/%d")},
        {"l": "en-us", "d": date_obj.strftime("%Y/%m/%d")},
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.tpex.org.tw/",
    }

    page_html = None
    final_url = None

    for params in date_try_params:
        r = session.get(TPEX_PAGE, params=params, headers=headers, timeout=30,verify=SSL_VERIFY)
        if r.status_code != 200:
            continue
        text = (r.text or "").strip()
        if len(text) < 200:
            continue
        page_html = text
        final_url = r.url
        break

    if not page_html:
        print(f"[TPEx] PAGE_FAIL {date_obj.strftime('%Y%m%d')}")
        return None

    csv_link = find_csv_link_in_html(page_html)
    if not csv_link:
        # Not fatal; TPEx may change page structure
        print(f"[TPEx] CSV_LINK_NOT_FOUND {date_obj.strftime('%Y%m%d')} page={final_url}")
        return None

    # Make link absolute if relative
    if csv_link.startswith("/"):
        csv_url = "https://www.tpex.org.tw" + csv_link
    elif csv_link.startswith("http"):
        csv_url = csv_link
    else:
        # relative to current path
        csv_url = "https://www.tpex.org.tw" + os.path.join(os.path.dirname("/en/stock/aftertrading/DAILY_CLOSE_quotes/"), csv_link)

    # Fetch CSV
    headers_csv = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": final_url or "https://www.tpex.org.tw/",
    }
    r2 = session.get(csv_url, headers=headers_csv, timeout=60,verify=SSL_VERIFY)
    if r2.status_code != 200:
        print(f"[TPEx] CSV_HTTP_{r2.status_code} {date_obj.strftime('%Y%m%d')} sample={r2.text[:120]!r}")
        return None

    csv_text = (r2.text or "").strip()
    if not csv_text:
        print(f"[TPEx] CSV_EMPTY {date_obj.strftime('%Y%m%d')}")
        return None

    # Read CSV; TPEx CSV sometimes includes extra header lines; pandas can handle with python engine
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text), engine="python")
    except Exception as ex:
        print(f"[TPEx] CSV_PARSE_FAIL {date_obj.strftime('%Y%m%d')} err={ex}")
        return None

    df["date"] = date_obj.strftime("%Y%m%d")
    return df

def normalize_tpex(df: pd.DataFrame) -> pd.DataFrame:
    """
    TPEx CSV columns can vary. We'll try to map common ones.
    If mapping fails, we still keep raw columns plus date.
    """
    # Common English page columns often include something like:
    # 'Stock Code', 'Name', 'Close', 'Change', ...
    # We map what we can.
    colmap_candidates = {
        "Stock Code": "stock_id",
        "Code": "stock_id",
        "Security Code": "stock_id",
        "Name": "name",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Trade Volume": "volume",
        "Trade Value": "amount",
        "Value": "amount",
        "Number of Trades": "trades",
    }

    out = df.copy()
    for src, dst in colmap_candidates.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    # If we successfully got stock_id/close, compact schema
    keep = ["date", "stock_id", "name", "open", "high", "low", "close", "volume", "amount", "trades"]
    keep = [c for c in keep if c in out.columns]
    if "stock_id" in keep or "close" in keep:
        out = out[keep]

    # numeric clean
    for c in ["open", "high", "low", "close", "volume", "amount", "trades"]:
        if c in out.columns:
            out[c] = to_numeric_clean(out[c])

    if "stock_id" in out.columns:
        out["stock_id"] = out["stock_id"].astype(str)
    for c in ["open", "high", "low", "close", "amount"]:
        if c in out.columns:
            out[c] = out[c].astype("float32")
    for c in ["volume", "trades"]:
        if c in out.columns:
            out[c] = out[c].astype("int64")

    return out


# -----------------------------
# Main downloading loop
# -----------------------------
def iter_days(start: datetime, end: datetime) -> List[datetime]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days

def flush_month(bucket: List[pd.DataFrame], out_path: str) -> None:
    if not bucket:
        return
    df = pd.concat(bucket, ignore_index=True)
    df.to_parquet(out_path, index=False)  # requires pyarrow or fastparquet installed
    print(f"[WRITE] {out_path} rows={len(df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--resume", action="store_true", help="resume from state.json if exists")
    ap.add_argument("--no-tpex", action="store_true", help="disable TPEx download")
    ap.add_argument("--sleep", type=float, default=0.7, help="base sleep seconds between requests")
    args = ap.parse_args()

    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    out_dir = args.out
    ensure_dir(out_dir)

    state_path = os.path.join(out_dir, "state.json")
    state = load_state(state_path)

    if args.resume and state.get("last_date"):
        # resume from next day
        last = datetime.strptime(state["last_date"], "%Y%m%d")
        start = max(start, last + timedelta(days=1))
        print(f"[RESUME] last_date={state['last_date']} -> start={start.strftime('%Y-%m-%d')}")

    session = requests.Session()
    warmup_twse(session)
    if not args.no_tpex:
        warmup_tpex(session)

    twse_bucket: Dict[str, List[pd.DataFrame]] = {}
    tpex_bucket: Dict[str, List[pd.DataFrame]] = {}

    for day in iter_days(start, end):
        yyyymmdd = iso_to_yyyymmdd(day)
        yyyymm = iso_to_yyyymm(day)

        polite_sleep(base=args.sleep, jitter=args.sleep)

        # --- TWSE ---
        twse_ok = False
        for attempt in range(4):
            try:
                payload = fetch_twse_payload(yyyymmdd, session)
                if payload is None:
                    break  # non-trading day or blocked; no point retrying too much
                raw = parse_twse_quotes(payload, yyyymmdd)
                norm = normalize_twse(raw)
                if len(norm) > 0:
                    twse_bucket.setdefault(yyyymm, []).append(norm)
                    twse_ok = True
                break
            except Exception as ex:
                if attempt == 3:
                    print(f"[TWSE] FAIL {yyyymmdd} {ex}")
                time.sleep(1.0 * (attempt + 1))

        # --- TPEx ---
        tpex_ok = False
        if not args.no_tpex:
            polite_sleep(base=0.3, jitter=0.4)
            for attempt in range(3):
                try:
                    df_csv = fetch_tpex_csv(day, session)
                    if df_csv is None:
                        break
                    norm2 = normalize_tpex(df_csv)
                    if len(norm2) > 0:
                        tpex_bucket.setdefault(yyyymm, []).append(norm2)
                        tpex_ok = True
                    break
                except Exception as ex:
                    if attempt == 2:
                        print(f"[TPEx] FAIL {yyyymmdd} {ex}")
                    time.sleep(1.0 * (attempt + 1))

        # update state only if at least TWSE succeeded OR TPEx succeeded
        if twse_ok or tpex_ok:
            state["last_date"] = yyyymmdd
            save_state(state_path, state)
            print(f"[OK] {yyyymmdd} TWSE={twse_ok} TPEx={tpex_ok}")
        else:
            # not necessarily an error (holiday/weekend); keep going
            print(f"[SKIP] {yyyymmdd} (non-trading or unavailable)")

        # flush monthly when bucket gets big to keep RAM stable
        if yyyymm in twse_bucket and len(twse_bucket[yyyymm]) >= 15:
            out_path = os.path.join(out_dir, f"twse_quotes_{yyyymm}.parquet")
            flush_month(twse_bucket[yyyymm], out_path)
            twse_bucket[yyyymm] = []
        if yyyymm in tpex_bucket and len(tpex_bucket[yyyymm]) >= 15:
            out_path = os.path.join(out_dir, f"tpex_quotes_{yyyymm}.parquet")
            flush_month(tpex_bucket[yyyymm], out_path)
            tpex_bucket[yyyymm] = []

    # final flush
    for m, parts in twse_bucket.items():
        if parts:
            out_path = os.path.join(out_dir, f"twse_quotes_{m}.parquet")
            flush_month(parts, out_path)

    for m, parts in tpex_bucket.items():
        if parts:
            out_path = os.path.join(out_dir, f"tpex_quotes_{m}.parquet")
            flush_month(parts, out_path)

    print("[DONE]")


if __name__ == "__main__":
    main()
