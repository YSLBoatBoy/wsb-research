"""
wsb_common.py - shared code for the r/wallstreetbets research tools.

Data sources (all free, no API keys):
  * Reddit history: Arctic Shift archive API (a Pushshift successor)
  * Recent Reddit fallback: reddit.com/r/wallstreetbets/new.json
  * Ticker list: NASDAQ Trader symbol directory (all US-listed stocks/ETFs)
  * Prices: Yahoo Finance via the `yfinance` package (split/dividend adjusted)

Timing convention (important - this is what keeps the study free of look-ahead):
  every post is assigned to the FIRST market open (09:30 ET) after it was posted.
  That day is the post's "entry day": the earliest moment a trader who saw the
  post could actually buy. All returns are measured from that open onward.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("WSB_DATA_DIR", ROOT / "data"))
UA = "python:wsb-research:v1.0 (personal research script)"
ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
REDDIT_NEW_URL = "https://www.reddit.com/r/wallstreetbets/new.json"
NASDAQ_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]
ET = "America/New_York"
BENCH = "SPY"
POST_COLS = ["id", "title", "selftext", "score", "num_comments", "created_utc",
             "link_flair_text", "author"]


# --------------------------------------------------------------------------- calendar
class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_BDAY = CustomBusinessDay(calendar=NYSECalendar())


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE trading days between start and end (inclusive), tz-naive midnight timestamps."""
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(),
                         freq=NYSE_BDAY)


def add_trading_days(day, n: int) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + n * NYSE_BDAY if n else pd.Timestamp(day).normalize()


def assign_entry_day(created_utc: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Map each post timestamp (epoch seconds) to the first 09:30 ET open strictly after it."""
    opens = days.tz_localize(ET) + pd.Timedelta(hours=9, minutes=30)
    open_epoch = np.array([o.timestamp() for o in opens], dtype=float)
    ts = pd.to_numeric(created_utc, errors="coerce").to_numpy(dtype=float)
    idx = np.searchsorted(open_epoch, ts, side="right")
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days.to_numpy(dtype="datetime64[ns]")[idx[ok]]
    return pd.Series(out, index=created_utc.index)


# --------------------------------------------------------------------------- reddit
def _get_json(session, url, params, log, tries=6):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=40)
            if r.status_code == 429 or r.status_code >= 500:
                wait = 5 * (attempt + 1)
                log(f"    server busy ({r.status_code}); waiting {wait}s")
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            log(f"    network error: {e}; retrying")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Could not reach {url}")


def fetch_posts_arctic(start_epoch: int, end_epoch: int, subreddit="wallstreetbets",
                       log=print, pause=0.35) -> pd.DataFrame:
    """Download every post in [start, end) from the Arctic Shift archive, oldest first."""
    s = requests.Session()
    s.headers["User-Agent"] = UA
    fields = ",".join(POST_COLS)
    use_fields = True
    after, rows, calls = int(start_epoch), [], 0
    while after < end_epoch:
        params = dict(subreddit=subreddit, after=after, before=int(end_epoch), limit=100, sort="asc")
        if use_fields:
            params["fields"] = fields
        r = _get_json(s, ARCTIC_URL, params, log)
        if r.status_code == 400 and use_fields:
            use_fields = False
            continue
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            break
        rows.extend(data)
        last = max(int(float(d.get("created_utc", after))) for d in data)
        after = last if last > after else after + 1
        calls += 1
        if calls % 25 == 0:
            log(f"    ...{len(rows):,} posts so far (up to {pd.Timestamp(last, unit='s'):%Y-%m-%d %H:%M} UTC)")
        time.sleep(pause)
    return _clean_posts(pd.DataFrame(rows))


def fetch_posts_reddit_recent(max_pages=10, log=print) -> pd.DataFrame:
    """Fallback: newest ~1000 posts straight from reddit.com (no history beyond that)."""
    s = requests.Session()
    s.headers["User-Agent"] = UA
    after, rows = None, []
    for _ in range(max_pages):
        r = _get_json(s, REDDIT_NEW_URL, {"limit": 100, "after": after, "raw_json": 1}, log)
        r.raise_for_status()
        j = r.json()["data"]
        rows += [c["data"] for c in j["children"]]
        after = j.get("after")
        if not after:
            break
        time.sleep(2)
    return _clean_posts(pd.DataFrame(rows))


def _clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=POST_COLS)
    for c in POST_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[POST_COLS].copy()
    df["id"] = df["id"].astype(str)
    for c in ["title", "selftext", "link_flair_text", "author"]:
        df[c] = df[c].fillna("").astype(str)
    for c in ["score", "num_comments", "created_utc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["created_utc"] = df["created_utc"].astype("int64")
    return df.drop_duplicates("id").sort_values("created_utc").reset_index(drop=True)


def load_or_fetch_posts(start, end, cache_file: Path, log=print, allow_reddit_fallback=True):
    """Posts between start and end (tz-aware or UTC timestamps), cached incrementally on disk."""
    start_e, end_e = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached = _clean_posts(pd.read_csv(cache_file, dtype={"id": str})) if cache_file.exists() else _clean_posts(pd.DataFrame())
    parts = [cached]
    ranges = []
    if cached.empty:
        ranges.append((start_e, end_e))
    else:
        lo, hi = int(cached.created_utc.min()), int(cached.created_utc.max())
        if start_e < lo - 3600:
            ranges.append((start_e, lo))
        if end_e > hi + 60:
            ranges.append((hi, end_e))
    for a, b in ranges:
        log(f"  Downloading r/wallstreetbets posts {pd.Timestamp(a, unit='s'):%Y-%m-%d} -> "
            f"{pd.Timestamp(b, unit='s'):%Y-%m-%d %H:%M} UTC from Arctic Shift...")
        try:
            parts.append(fetch_posts_arctic(a, b, log=log))
        except Exception as e:  # archive down -> recent posts only
            log(f"  Arctic Shift failed ({e}).")
            if not allow_reddit_fallback:
                raise
            log("  Falling back to reddit.com/new (only the newest ~1000 posts).")
            parts.append(fetch_posts_reddit_recent(log=log))
    allp = _clean_posts(pd.concat(parts, ignore_index=True))
    # keep the newest copy of each post (scores keep changing)
    allp.to_csv(cache_file, index=False, compression="gzip" if cache_file.suffix == ".gz" else None)
    sel = allp[(allp.created_utc >= start_e) & (allp.created_utc < end_e)]
    return sel.reset_index(drop=True)


# --------------------------------------------------------------------------- tickers
FALLBACK_TICKERS = set("""
AAPL MSFT NVDA AMD TSLA AMZN GOOGL GOOG META NFLX INTC MU SMCI PLTR SOFI GME AMC BB NOK HOOD COIN
MSTR RIVN LCID NIO F GM BA DIS PYPL XYZ SHOP UBER LYFT SNAP PINS RBLX U ROKU ARM AVGO TSM QCOM
ASML ORCL CRM ADBE IBM DELL HPQ SPY QQQ IWM DIA TQQQ SQQQ SPXL SPXS UVXY TLT GLD SLV USO XLE XLF
JPM BAC WFC C GS MS SCHW V MA AXP KO PEP WMT TGT COST HD NKE SBUX MCD CMG LULU XOM CVX OXY COP
PFE MRNA JNJ LLY NVO UNH CVS ABBV BABA JD PDD BIDU CVNA UPST AFRM DKNG CHWY PTON BYND TLRY CGC
SNDL ASTS RKLB IONQ RGTI QBTS OKLO SMR NNE HIMS CELH ENPH FSLR RUN PLUG CHPT QS MARA RIOT CLSK
HUT IBIT CRWD PANW NET SNOW DDOG MDB ZS TTD APP DJT RDDT CRWV CAVA ELF ANF WBD PARA T VZ TMUS
""".split())

# words that are valid tickers but nearly always mean something else on WSB.
# They still count when written as a cashtag ($AI, $NOW, ...).
STOPWORDS = set("""
A I AM AN AND ANY ARE AS AT BE BIG BY CAN CEO CFO COO CTO DD DO EV EPS ETF FOR FOMO FED FYI GDP
CPI PPI GO HAS HE HIS HER HOLD IF IN IPO IS IT ITM OTM ATM LOL MY NEW NO NOW OF OK ON ONE OR OUT
PM PT RH SO THE TO UP US USA WSB YOLO ALL ATH BRO EDIT TLDR TL DR IMO IMHO JUST LIKE GAIN LOSS LOSE
MOON PUMP DUMP RUN REAL HUGE BEST GOOD BUY SELL CALL PUT EOD EOW EOY YTD QE IRS SEC FDA DOJ FTC
NYSE AI OPEN CAT TV FUN LOVE HOPE PLAY EAT LIFE CASH FREE NICE WOW BOOM FAST TECH PEAK RACE SAVE
SUN TRUE WELL WORK CAR KEY MAN MAIN POST LOW ELON GL HF DM TA RSI MACD SMA EMA IV OI PE PEG FCF
QOQ YOY MOM DAD WIFE BF GF NFA PDT ROI ROTH IRA LMAO WTF OMG NOT BUT WHY HOW WHO WHAT WHEN YES
VERY GET GOT SEE SAY DAY WEEK MONTH YEAR TIME NEXT LAST HIGH ARM USD EUR UK EU CNBC FAQ AMA PSA
DTE OP OG IRL ID AGI API APY FIRE GOAT HODL NEED SAFE BOTH EVER EDGE MOST FOUR CASH ONCE PLAN BEAT
""".split())

CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
BARE_RE = re.compile(r"\b([A-Z]{2,5})\b")

BULL_WORDS = {"call", "calls", "moon", "mooning", "rocket", "buy", "buying", "long", "bull", "bullish",
              "squeeze", "tendies", "undervalued", "rip", "breakout", "yolo", "printing", "lfg"}
BEAR_WORDS = {"put", "puts", "short", "shorting", "bear", "bearish", "drill", "crash", "dump", "dumping",
              "overvalued", "sell", "selling", "bagholder", "bagholding", "rug", "puke", "tank"}
BULL_EMOJI = ("\U0001F680", "\U0001F48E", "\U0001F315")       # rocket, diamond, full moon
BEAR_EMOJI = ("\U0001F308\U0001F43B", "\U0001F4C9")           # rainbow bear, chart down
WORD_RE = re.compile(r"[a-z']+")


def load_ticker_universe(log=print) -> set[str]:
    cache = DATA_DIR / "tickers.txt"
    if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
        return set(cache.read_text().split())
    syms = set()
    try:
        for url in NASDAQ_URLS:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            header = lines[0].split("|")
            test_col = header.index("Test Issue") if "Test Issue" in header else None
            for line in lines[1:]:
                if line.startswith("File Creation Time"):
                    continue
                cols = line.split("|")
                if test_col is not None and len(cols) > test_col and cols[test_col] == "Y":
                    continue
                sym = cols[0].strip()
                if re.fullmatch(r"[A-Z]{1,5}", sym):
                    syms.add(sym)
    except Exception as e:
        log(f"  Could not download the ticker list ({e}); using a built-in list of ~170 popular tickers.")
        return set(FALLBACK_TICKERS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(syms)))
    return syms


def extract_tickers(title: str, body: str, universe: set[str]) -> set[str]:
    found = set()
    for text in (title, body):
        if not text:
            continue
        for m in CASHTAG_RE.findall(text):
            t = m.upper()
            if t in universe:
                found.add(t)
        letters = [c for c in text if c.isalpha()]
        shouting = letters and sum(c.isupper() for c in letters) / len(letters) > 0.5
        if shouting:          # ALL-CAPS posts: only trust cashtags
            continue
        for m in BARE_RE.findall(text):
            if m in universe and m not in STOPWORDS:
                found.add(m)
    return found


def sentiment_score(text: str) -> float:
    """Crude WSB lexicon sentiment in [-1, 1]. Positive = bullish."""
    low = text.lower()
    words = WORD_RE.findall(low)
    bull = sum(w in BULL_WORDS for w in words) + sum(low.count(e) for e in BULL_EMOJI)
    bear = sum(w in BEAR_WORDS for w in words) + sum(low.count(e) for e in BEAR_EMOJI)
    return (bull - bear) / (bull + bear + 1)


# --------------------------------------------------------------------------- attention
def build_mentions(posts: pd.DataFrame, universe: set[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (post, ticker) with the post's entry day, weight and sentiment."""
    if posts.empty:
        return pd.DataFrame(columns=["day", "ticker", "post_id", "weight", "sentiment", "is_dd"])
    entry = assign_entry_day(posts["created_utc"], days)
    rows = []
    for p, day in zip(posts.itertuples(index=False), entry):
        if pd.isna(day):
            continue
        tick = extract_tickers(p.title, p.selftext if p.selftext not in ("[removed]", "[deleted]") else "", universe)
        if not tick:
            continue
        # popularity weight: every post counts 1, more for upvotes/comments (log-scaled)
        w = 1.0 + math.log1p(max(p.score, 0)) + 0.5 * math.log1p(max(p.num_comments, 0))
        sent = sentiment_score(f"{p.title} {p.selftext[:3000]}")
        is_dd = "dd" in p.link_flair_text.lower() or "due diligence" in p.link_flair_text.lower()
        for t in tick:
            rows.append((day, t, p.id, w, sent, is_dd))
    return pd.DataFrame(rows, columns=["day", "ticker", "post_id", "weight", "sentiment", "is_dd"])


def aggregate_attention(mentions: pd.DataFrame) -> pd.DataFrame:
    """Daily per-ticker attention table: attn (weighted), mentions (#posts), sentiment, dd_posts."""
    if mentions.empty:
        return pd.DataFrame(columns=["day", "ticker", "attn", "mentions", "sentiment", "dd_posts"])
    g = mentions.groupby(["day", "ticker"])
    out = pd.DataFrame({
        "attn": g["weight"].sum(),
        "mentions": g["post_id"].nunique(),
        "sentiment": g["sentiment"].mean(),
        "dd_posts": g["is_dd"].sum(),
    }).reset_index()
    return out


def detect_spikes(daily: pd.DataFrame, days: pd.DatetimeIndex, lookback=20, z=2.0,
                  min_mentions=5, std_floor=1.0, cooldown=5) -> pd.DataFrame:
    """A ticker 'trends' on day d when its attention is >= z std devs above its own
    trailing `lookback`-day average (prior days only) AND it got >= min_mentions posts.
    Repeat triggers within `cooldown` trading days of the last trigger are dropped."""
    cols = ["day", "ticker", "z", "attn", "mentions", "sentiment", "dd_posts"]
    if daily.empty:
        return pd.DataFrame(columns=cols)
    wide = {k: daily.pivot_table(index="day", columns="ticker", values=k, aggfunc="sum")
            .reindex(days).fillna(0.0) for k in ["attn", "mentions", "dd_posts"]}
    sent = daily.pivot_table(index="day", columns="ticker", values="sentiment").reindex(days)
    a = wide["attn"]
    prior = a.shift(1).rolling(lookback, min_periods=5)
    mu, sd = prior.mean(), prior.std().clip(lower=std_floor)
    zs = (a - mu) / sd
    hit = (zs >= z) & (wide["mentions"] >= min_mentions) & mu.notna()
    ev = hit.stack()
    ev = ev[ev].index.to_frame(index=False)
    ev.columns = ["day", "ticker"]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    ev["z"] = [zs.at[d, t] for d, t in zip(ev.day, ev.ticker)]
    ev["attn"] = [a.at[d, t] for d, t in zip(ev.day, ev.ticker)]
    ev["mentions"] = [wide["mentions"].at[d, t] for d, t in zip(ev.day, ev.ticker)]
    ev["sentiment"] = [sent.at[d, t] for d, t in zip(ev.day, ev.ticker)]
    ev["dd_posts"] = [wide["dd_posts"].at[d, t] for d, t in zip(ev.day, ev.ticker)]
    # cooldown
    pos = {d: i for i, d in enumerate(days)}
    keep, last = [], {}
    for r in ev.sort_values(["day", "z"], ascending=[True, False]).itertuples(index=False):
        i = pos[r.day]
        if r.ticker in last and i - last[r.ticker] < cooldown:
            continue
        last[r.ticker] = i
        keep.append(r)
    return pd.DataFrame(keep, columns=cols).reset_index(drop=True)


def filter_events(events: pd.DataFrame, flt: str) -> pd.DataFrame:
    if flt == "bullish":
        return events[events.sentiment > 0]
    if flt == "bearish":
        return events[events.sentiment < 0]
    if flt == "dd":
        return events[events.dd_posts > 0]
    return events


# --------------------------------------------------------------------------- prices
def fetch_prices(tickers, start, end, log=print):
    """Adjusted daily open/close from Yahoo Finance. Returns (opens, closes) wide frames."""
    import yfinance as yf

    tickers = sorted(set(tickers) | {BENCH})
    opens, closes = [], []
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        log(f"  Prices {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        df = yf.download(chunk, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                         end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False, threads=True, group_by="column")
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            o, c = df["Open"], df["Close"]
        else:
            o, c = df[["Open"]].rename(columns={"Open": chunk[0]}), df[["Close"]].rename(columns={"Close": chunk[0]})
        opens.append(o)
        closes.append(c)
        time.sleep(1)
    if not closes:
        raise RuntimeError("No price data returned by Yahoo Finance.")
    o = pd.concat(opens, axis=1)
    c = pd.concat(closes, axis=1)
    o.index = pd.DatetimeIndex(o.index).tz_localize(None).normalize()
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    o, c = o.loc[:, ~o.columns.duplicated()], c.loc[:, ~c.columns.duplicated()]
    good = [t for t in c.columns if c[t].notna().sum() > 0]
    return o[good].astype(float), c[good].astype(float)


# --------------------------------------------------------------------------- demo data
def make_demo_data(end=None, n_days=130, n_tickers=40, seed=7, effect=True):
    """Synthetic posts + prices so the whole pipeline can be tested offline.
    With effect=True, hype spikes are followed by a short pump then a fade (so the
    analysis has something to find). It is NOT real market data."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end or pd.Timestamp.now(tz=ET).date())
    days = trading_days(end - pd.Timedelta(days=int(n_days * 1.5)), end)[-n_days:]
    letters = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    tickers = sorted({"".join(rng.choice(letters, 4)) for _ in range(n_tickers * 2)})[:n_tickers]
    universe = set(tickers) | {BENCH}
    n = len(days)
    base = rng.uniform(0.2, 3.0, len(tickers))
    spike = rng.random((n, len(tickers))) < 0.015
    spike[:25] = False
    counts = rng.poisson(base, (n, len(tickers))) + spike * rng.poisson(25, (n, len(tickers)))
    # prices
    gap = rng.normal(0, 0.01, (n, len(tickers)))
    intra = rng.normal(0.0003, 0.025, (n, len(tickers)))
    if effect:
        for i, j in zip(*np.nonzero(spike)):
            gap[i, j] += 0.04                      # the run-up that drew the attention
            for k, d in enumerate([0.015, 0.012, 0.010, -0.010, -0.010, -0.012, -0.012, -0.010]):
                if i + k < n:
                    intra[i + k, j] += d
    logp = np.cumsum(gap + intra, axis=0) + np.log(rng.uniform(5, 200, len(tickers)))
    close = np.exp(logp)
    open_ = np.exp(logp - intra)
    spy_r = rng.normal(0.0004, 0.009, n)
    spy_c = 500 * np.exp(np.cumsum(spy_r))
    spy_o = spy_c / np.exp(rng.normal(0, 0.004, n))
    closes = pd.DataFrame(close, index=days, columns=tickers)
    opens = pd.DataFrame(open_, index=days, columns=tickers)
    closes[BENCH], opens[BENCH] = spy_c, spy_o
    # posts: spread in the window (previous open, this open]
    open_ts = np.array([(d.tz_localize(ET) + pd.Timedelta(hours=9, minutes=30)).timestamp() for d in days])
    rows, pid = [], 0
    for i in range(1, n):
        lo, hi = open_ts[i - 1], open_ts[i]
        for j, t in enumerate(tickers):
            for _ in range(counts[i, j]):
                pid += 1
                bull = rng.random() < (0.75 if spike[i, j] else 0.5)
                title = f"${t} " + ("calls to the moon \U0001F680" if bull else "puts, this will drill")
                rows.append(dict(id=f"d{pid}", title=title, selftext="", score=int(rng.lognormal(2, 1.5)),
                                 num_comments=int(rng.lognormal(1.5, 1.2)),
                                 created_utc=int(rng.uniform(lo + 1, hi - 1)),
                                 link_flair_text="DD" if rng.random() < 0.1 else "Discussion", author="demo"))
    posts = _clean_posts(pd.DataFrame(rows))
    return posts, universe, opens, closes


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
