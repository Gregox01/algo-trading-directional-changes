"""Fetch historical market data for the multi-asset DC edge study.

Runs on a GitHub Actions runner (unrestricted network). Produces gzipped CSVs in
./dataout with the same schema as the repo's ETHUSDT_15m.csv:
    Datetime,Open,High,Low,Close,Volume

Sources:
- Crypto: Binance public monthly kline archives (data.binance.vision), spot.
- FX: histdata.com 1-minute bars via the `histdata` package, resampled to 10m/1h.
  (FX volume is tick count at best; treated as 0. Timestamps are EST as shipped.)
- Stocks/ETFs: stooq.com daily CSV endpoint.

Every fetch failure is logged and skipped; MANIFEST.json records what succeeded.
"""

import calendar
import io
import json
import os
import zipfile
from datetime import date

import pandas as pd
import requests

OUT = "dataout"
os.makedirs(OUT, exist_ok=True)
manifest = {}

CRYPTO = {
    "BTCUSDT": "2017-08",
    "ETHUSDT": "2017-08",
    "BNBUSDT": "2017-11",
    "XRPUSDT": "2018-05",
    "SOLUSDT": "2020-08",
}
CRYPTO_INTERVALS = ["15m", "1h"]
FX_PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd"]
FX_YEARS = list(range(2015, 2026))
STOCKS = ["spy.us", "qqq.us", "aapl.us", "msft.us", "nvda.us", "amzn.us"]

TODAY = date.today()


def month_range(start_ym, end_ym):
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_binance(symbol, interval, start_ym):
    frames = []
    months_ok = []
    # archives lag ~1 month behind the current date
    end_ym = f"{TODAY.year:04d}-{TODAY.month - 1:02d}" if TODAY.month > 1 else f"{TODAY.year - 1}-12"
    for ym in month_range(start_ym, end_ym):
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{ym}.zip")
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = zf.namelist()[0]
                df = pd.read_csv(zf.open(name), header=None)
        except Exception as e:
            print(f"  {symbol} {interval} {ym}: FAILED {e}")
            continue
        # Some 2025+ archives ship a header row
        if not str(df.iloc[0, 0]).replace(".", "").isdigit():
            df = df.iloc[1:].reset_index(drop=True)
        df = df.iloc[:, :6]
        df.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]
        ts = pd.to_numeric(df["ts"])
        # microseconds since 2025-01, milliseconds before
        unit = ts.gt(1e14).map({True: "us", False: "ms"})
        dt = pd.Series(pd.NaT, index=df.index)
        dt[unit == "us"] = pd.to_datetime(ts[unit == "us"], unit="us")
        dt[unit == "ms"] = pd.to_datetime(ts[unit == "ms"], unit="ms")
        df["Datetime"] = dt
        frames.append(df[["Datetime", "Open", "High", "Low", "Close", "Volume"]])
        months_ok.append(ym)
    if not frames:
        return None, []
    out = pd.concat(frames).sort_values("Datetime").drop_duplicates("Datetime")
    return out, months_ok


def fetch_fx(pair):
    from histdata import download_hist_data as dl
    from histdata.api import Platform as P, TimeFrame as TF
    frames = []
    years_ok = []
    for year in FX_YEARS:
        try:
            path = dl(year=str(year), pair=pair, platform=P.GENERIC_ASCII,
                      time_frame=TF.ONE_MINUTE, output_directory="fxtmp")
            with zipfile.ZipFile(path) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(zf.open(name), sep=";", header=None,
                                 names=["dt", "Open", "High", "Low", "Close", "Volume"])
            df["Datetime"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
            frames.append(df[["Datetime", "Open", "High", "Low", "Close", "Volume"]])
            years_ok.append(year)
            print(f"  {pair} {year}: ok ({len(df)} rows)")
        except Exception as e:
            print(f"  {pair} {year}: FAILED {e}")
    if not frames:
        return None, []
    out = pd.concat(frames).sort_values("Datetime").drop_duplicates("Datetime")
    return out, years_ok


def resample_ohlc(df, rule):
    g = df.set_index("Datetime").resample(rule)
    out = pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
    }).dropna(subset=["Open", "Close"]).reset_index()
    return out


def fetch_stooq(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Date" not in df.columns or len(df) < 100:
        raise RuntimeError(f"unexpected stooq payload: {r.text[:100]}")
    df = df.rename(columns={"Date": "Datetime"})
    df = df[df["Datetime"] >= "2000-01-01"]
    return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, compression="gzip")
    print(f"WROTE {path}: {len(df)} rows, {os.path.getsize(path) / 1e6:.1f} MB")


for symbol, start in CRYPTO.items():
    for interval in CRYPTO_INTERVALS:
        print(f"Fetching {symbol} {interval}...")
        df, ok = fetch_binance(symbol, interval, start)
        if df is not None:
            save(df, f"{symbol}_{interval}.csv.gz")
            manifest[f"{symbol}_{interval}"] = {
                "rows": len(df), "months": len(ok),
                "start": str(df["Datetime"].iloc[0]), "end": str(df["Datetime"].iloc[-1]),
            }

for pair in FX_PAIRS:
    print(f"Fetching FX {pair}...")
    df, ok = fetch_fx(pair)
    if df is not None:
        for rule, tag in [("10min", "10m"), ("1h", "1h")]:
            r = resample_ohlc(df, rule)
            save(r, f"{pair.upper()}_{tag}.csv.gz")
            manifest[f"{pair.upper()}_{tag}"] = {
                "rows": len(r), "years": ok,
                "start": str(r["Datetime"].iloc[0]), "end": str(r["Datetime"].iloc[-1]),
            }

for ticker in STOCKS:
    print(f"Fetching stooq {ticker}...")
    try:
        df = fetch_stooq(ticker)
        name = ticker.split(".")[0].upper()
        save(df, f"{name}_1d.csv.gz")
        manifest[f"{name}_1d"] = {
            "rows": len(df),
            "start": str(df["Datetime"].iloc[0]), "end": str(df["Datetime"].iloc[-1]),
        }
    except Exception as e:
        print(f"  {ticker}: FAILED {e}")

with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
