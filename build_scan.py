#!/usr/bin/env python3
"""
build_scan.py — produces data.json for the RPCI dashboard, over the FULL NSE universe.

What it does, in order:
  1. Loads every NSE-listed equity (SERIES == "EQ") from NSE's official master list.
  2. Attaches a sector to each symbol (from a local sectors.csv if you have one,
     otherwise leaves it "Unclassified" — see notes at the bottom).
  3. Downloads ~1 year of daily prices per symbol and evaluates the eleven RPCI
     conditions.
  4. Computes Δ and "Held" against yesterday's data.json, appends today's point to
     the 30-session breadth history, and writes a fresh data.json.

Run once per day at 5:00 PM IST (cron / GitHub Actions notes at the bottom).

    pip install requests pandas yfinance
    python build_scan.py                 # full universe
    python build_scan.py --limit 50      # quick test on the first 50 names
    python build_scan.py --universe EQUITY_L.csv   # use a file you downloaded

IMPORTANT
  - yfinance has no API key but is NOT an official NSE feed and will rate-limit or
    fail on some names across ~2,000 symbols. For production, replace fetch_history()
    with a licensed vendor or your broker's API. Everything else stays the same.
  - A full run fetches thousands of series and takes a while (tune THROTTLE / workers).
"""

import argparse, io, json, sys, time, datetime, concurrent.futures as cf
import pandas as pd

# ------------------------------------------------------------------ tunables
NSE_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_HOME       = "https://www.nseindia.com"
THROTTLE       = 0.15     # seconds between price requests (be polite / avoid bans)
WORKERS        = 4        # parallel price downloads; keep low for free sources
RETRIES        = 2
HIST_TAIL      = 30       # sparkline length stored per stock
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ------------------------------------------------------------------ universe
def load_universe(local_csv=None):
    """Return dict {symbol: sector}. Prefers a local CSV; else fetches NSE master.
    NSE's EQUITY_L.csv has no sector column, so sector is enriched separately."""
    df = None
    if local_csv:
        df = pd.read_csv(local_csv)
        print(f"universe: read {local_csv}")
    else:
        import requests
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                          "Accept": "text/csv,*/*"})
        try:
            s.get(NSE_HOME, timeout=15)          # prime cookies (NSE blocks cold requests)
            r = s.get(NSE_MASTER_URL, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print("universe: fetched NSE master list")
        except Exception as e:
            print("could not fetch NSE master:", e, file=sys.stderr)
            print("download it manually from", NSE_MASTER_URL,
                  "and pass --universe EQUITY_L.csv", file=sys.stderr)
            sys.exit(1)

    df.columns = [c.strip().upper().replace(" ", "") for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
    symcol = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
    symbols = [str(x).strip() for x in df[symcol] if str(x).strip()]

    sectors = load_sector_map()
    universe = {sym: sectors.get(sym, "Unclassified") for sym in symbols}
    print(f"universe: {len(universe)} EQ symbols, "
          f"{sum(1 for v in universe.values() if v!='Unclassified')} with sector")
    return universe

def load_sector_map(path="sectors.csv"):
    """Optional symbol->sector map. CSV with headers: symbol,sector.
    NSE does not ship sector in EQUITY_L; supply this file for real sectors
    (e.g. exported from an index-constituents source). Missing file = no sectors."""
    try:
        m = pd.read_csv(path)
        m.columns = [c.strip().lower() for c in m.columns]
        return {str(r["symbol"]).strip(): str(r["sector"]).strip() for _, r in m.iterrows()}
    except Exception:
        return {}

# ------------------------------------------------------------------ indicators
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, 1e-9))

def macd_hist(close):
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()

def eleven_conditions(df):
    """df: daily OHLCV (oldest->newest) with Close, Volume. Returns
    (conds[11], close, dma20_pct, chg_pct, volume)."""
    close, vol = df["Close"], df["Volume"]
    ma20, ma50, ma200 = close.rolling(20).mean(), close.rolling(50).mean(), close.rolling(200).mean()
    c = close.iloc[-1]
    conds = [
        c > ma20.iloc[-1],                                # 1 Close > 20 DMA
        c > ma50.iloc[-1],                                # 2 Close > 50 DMA
        c > ma200.iloc[-1],                               # 3 Close > 200 DMA
        ma20.iloc[-1] > ma50.iloc[-1],                    # 4 20 DMA > 50 DMA
        rsi(close).iloc[-1] > 55,                         # 5 RSI(14) > 55
        vol.iloc[-1] > vol.rolling(20).mean().iloc[-1],   # 6 Volume > 20d avg
        c >= 0.95 * close.rolling(50).max().iloc[-1],     # 7 Within 5% of 50d high
        macd_hist(close).iloc[-1] > 0,                    # 8 MACD positive
        True,   # 9  Earnings growth +  → wire to your fundamentals source
        True,   # 10 ROE > 15%          → wire to your fundamentals source
        c > close.iloc[-2],                               # 11 Above prior pivot (proxy)
    ]
    conds = [bool(x) for x in conds]
    return (conds, round(float(c), 2),
            round(float((c/ma20.iloc[-1]-1)*100), 2),
            round(float((c/close.iloc[-2]-1)*100), 2),
            int(vol.iloc[-1]))

# ------------------------------------------------------------------ price source
def fetch_history(symbol):
    """Return a daily OHLCV DataFrame (>=60 rows) for one NSE symbol, or None.
    Swap this body for a licensed provider in production."""
    import yfinance as yf
    yf_sym = symbol.replace("&", "%26") + ".NS"
    for attempt in range(RETRIES + 1):
        try:
            df = yf.download(yf_sym, period="1y", interval="1d",
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and len(df) >= 60:
                return df.rename(columns=str.title)[["Close", "Volume"]].dropna()
            return None
        except Exception:
            if attempt < RETRIES:
                time.sleep(1 + attempt)
            else:
                return None

# ------------------------------------------------------------------ prev-day state
def load_prev(path="data.json"):
    """Yesterday's score + held count per symbol, and prior breadth history."""
    try:
        old = json.load(open(path))
        prev = {}
        for s in old["stocks"]:
            score = s["conds"].count(True) if isinstance(s["conds"][0], bool) else sum(s["conds"])
            prev[s["ticker"]] = (score, s.get("held", 0))
        return prev, old.get("breadth", {"cnt": [], "mean": []})
    except Exception:
        return {}, {"cnt": [], "mean": []}

# ------------------------------------------------------------------ per-symbol worker
def scan_one(sym, sector, prev):
    df = fetch_history(sym)
    time.sleep(THROTTLE)
    if df is None:
        return None
    try:
        conds, close, dma, chg, vol = eleven_conditions(df)
    except Exception:
        return None
    score = sum(conds)
    prev_score, prev_held = prev.get(sym, (score, 0))
    held = prev_held + 1 if score == prev_score else 1
    hist = [round(float(x), 2) for x in df["Close"].tail(HIST_TAIL)]
    return {"ticker": sym, "sector": sector, "conds": conds, "prev": prev_score,
            "close": close, "dma": dma, "chg": chg, "volume": vol,
            "held": held, "hist": hist}

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N symbols")
    ap.add_argument("--universe", help="local EQUITY_L.csv instead of fetching")
    ap.add_argument("--out", default="data.json")
    args = ap.parse_args()

    universe = load_universe(args.universe)
    items = list(universe.items())
    if args.limit:
        items = items[:args.limit]
    prev, prev_breadth = load_prev(args.out)

    out, done, total = [], 0, len(items)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(scan_one, sym, sec, prev): sym for sym, sec in items}
        for fut in cf.as_completed(futs):
            done += 1
            rec = fut.result()
            if rec:
                out.append(rec)
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} processed, {len(out)} scored", flush=True)

    if not out:
        print("no symbols scored — check your price source", file=sys.stderr)
        sys.exit(1)

    out.sort(key=lambda s: sum(s["conds"]), reverse=True)
    cnt_today  = sum(1 for s in out if sum(s["conds"]) >= 9)
    mean_today = round(sum(sum(s["conds"]) for s in out) / len(out), 2)
    cnt  = (prev_breadth.get("cnt", [])  + [cnt_today])[-30:]
    mean = (prev_breadth.get("mean", []) + [mean_today])[-30:]

    data = {
        "updated": datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=5, minutes=30))).isoformat(),
        "stocks": out,
        "breadth": {"cnt": cnt, "mean": mean},
    }
    json.dump(data, open(args.out, "w"), separators=(",", ":"))
    print(f"\nwrote {args.out} — {len(out)} names, {cnt_today} at 9+, mean {mean_today}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# SECTORS
#   NSE's EQUITY_L.csv has no sector/industry column, so full-universe sectors need
#   a second source. Easiest: build sectors.csv (columns: symbol,sector) once from an
#   index-constituents export or a fundamentals vendor, and drop it beside this script.
#   Symbols not in that file show as "Unclassified" — the dashboard's Sector filter
#   still works, they just group under one label until you supply the map.
#
# SCHEDULING AT 5:00 PM IST  (17:00 IST == 11:30 UTC)
#   cron on an IST server:   0 17 * * 1-5  cd /path/to/site && python3 build_scan.py
#   cron on a UTC server:    30 11 * * 1-5 cd /path/to/site && python3 build_scan.py
#
#   GitHub Actions (free; commits data.json next to the dashboard on Pages):
#     name: rpci-scan
#     on:
#       schedule: [{ cron: "30 11 * * 1-5" }]   # 17:00 IST, Mon-Fri
#       workflow_dispatch:
#     jobs:
#       scan:
#         runs-on: ubuntu-latest
#         steps:
#           - uses: actions/checkout@v4
#           - uses: actions/setup-python@v5
#             with: { python-version: "3.11" }
#           - run: pip install requests pandas yfinance
#           - run: python build_scan.py
#           - run: |
#               git config user.name  "rpci-bot"
#               git config user.email "bot@users.noreply.github.com"
#               git add data.json && git commit -m "scan $(date -u +%F)" || echo "no change"
#               git push
#
#   Host rpci-dashboard.html + data.json together (GitHub Pages, Netlify, S3, any
#   static host). The dashboard loads data.json and re-checks after 5 PM IST.
# ---------------------------------------------------------------------------
