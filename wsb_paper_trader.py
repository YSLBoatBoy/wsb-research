"""
wsb_paper_trader.py - paper trades r/wallstreetbets trends with fake money.

  python wsb_paper_trader.py run       # do today's update (schedule this every weekday ~8:15 AM Central)
  python wsb_paper_trader.py status    # show the portfolio
  python wsb_paper_trader.py reset --capital 100000
  python wsb_paper_trader.py replay --days 60   # replay the last 60 days as if it had run every morning
  python wsb_paper_trader.py replay --demo      # offline test on synthetic data

Rules (identical to the backtest in wsb_analyze.py):
  * Each morning before the 09:30 ET open, count WSB posts per ticker since the previous open.
  * A ticker "trends" when that count/popularity is far above its own last-20-day normal.
  * Trending tickers are bought (or shorted, if the strategy says so) at today's open,
    in equal slots of equity / max_pos, and sold at the close `hold` trading days later.
  * Fills use the real daily open/close from Yahoo Finance, applied on the next run.

Strategy settings come from data/best_config.json (written by wsb_analyze.py) or the
defaults below. Nothing here touches a real brokerage account.
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

import wsb_common as wc

DEFAULT_CFG = dict(hold=3, side="long", filter="all", z=2.0, lookback=20, min_mentions=5,
                   max_pos=10, cost_bps=10)
HISTORY_DAYS = 90       # calendar days of posts kept for the "normal attention" baseline
BACKUPS = 5             # extra ranked signals queued in case a top pick has no price


class Book:
    def __init__(self, folder: wc.Path):
        self.dir = wc.ensure_dir(folder)
        self.file = self.dir / "book.json"
        self.s = json.loads(self.file.read_text()) if self.file.exists() else None

    def init(self, capital, cfg):
        self.s = dict(config=cfg, start_capital=capital, cash=capital, last_equity=capital,
                      positions=[], pending=[], last_processed=None, spy_start=None,
                      created=pd.Timestamp.now().isoformat(timespec="seconds"))
        for f in ["trades.csv", "equity.csv", "signals.csv"]:
            (self.dir / f).unlink(missing_ok=True)
        self.save()

    def save(self):
        self.file.write_text(json.dumps(self.s, indent=2, default=str))

    def append(self, name, rows):
        if not rows:
            return
        p = self.dir / name
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


def load_cfg():
    p = wc.DATA_DIR / "best_config.json"
    cfg = dict(DEFAULT_CFG)
    if p.exists():
        cfg.update({k: v for k, v in json.loads(p.read_text()).items() if k in DEFAULT_CFG})
    return cfg


# --------------------------------------------------------------------------- core step
def process_fills(book: Book, price_fn, upto: pd.Timestamp, log):
    s = book.s
    cfg = s["config"]
    sign = lambda side: 1 if side == "long" else -1
    cf = cfg["cost_bps"] / 1e4
    need = {p["t"] for p in s["positions"]} | {p["t"] for p in s["pending"]} | {wc.BENCH}
    start = pd.Timestamp(s["last_processed"]) if s["last_processed"] else None
    pend_days = [pd.Timestamp(p["entry_day"]) for p in s["pending"]]
    first = min([d for d in [start] + pend_days if d is not None], default=None)
    if first is None or first > upto:
        return
    opens, closes = price_fn(sorted(need), first - pd.Timedelta(days=10), upto)
    if wc.BENCH not in closes.columns:
        log("  No benchmark prices yet; will retry next run.")
        return
    spy = closes[wc.BENCH].dropna()
    closes_ff = closes.ffill(limit=5)
    days = [d for d in spy.index if d <= upto and (start is None or d > start)]
    for d in days:
        trades, fills = [], []
        # 1) fills at the open, best-ranked first, until slots are full
        todays = sorted([p for p in s["pending"] if pd.Timestamp(p["entry_day"]) == d], key=lambda p: p["rank"])
        for p in s["pending"]:
            if pd.Timestamp(p["entry_day"]) < d:
                log(f"  {d:%Y-%m-%d} cancelled stale order {p['t']}")
        held = {p["t"] for p in s["positions"]}
        for p in todays:
            if len(s["positions"]) >= cfg["max_pos"]:
                break
            px = opens.at[d, p["t"]] if p["t"] in opens.columns and d in opens.index else np.nan
            if p["t"] in held or not (px > 0):
                continue
            notional = s["last_equity"] / cfg["max_pos"]
            sh = notional / px
            s["cash"] -= sign(p["side"]) * sh * px + notional * cf
            s["positions"].append(dict(t=p["t"], side=p["side"], sh=sh, entry_px=float(px),
                                       entry_day=f"{d:%Y-%m-%d}", exit_day=p["exit_day"], last=float(px)))
            held.add(p["t"])
            fills.append(f"{p['side'].upper()} {p['t']} @ {px:.2f}")
        s["pending"] = [p for p in s["pending"] if pd.Timestamp(p["entry_day"]) > d]
        # 2) mark to market and exits at the close
        keep = []
        for p in s["positions"]:
            c = closes_ff.at[d, p["t"]] if p["t"] in closes_ff.columns and d in closes_ff.index else np.nan
            if c > 0:
                p["last"] = float(c)
            if pd.Timestamp(p["exit_day"]) <= d:
                xp = p["last"]
                s["cash"] += sign(p["side"]) * p["sh"] * xp - p["sh"] * xp * cf
                ret = sign(p["side"]) * (xp / p["entry_px"] - 1) - cf - cf * xp / p["entry_px"]
                trades.append(dict(ticker=p["t"], side=p["side"], entry_day=p["entry_day"], exit_day=f"{d:%Y-%m-%d}",
                                   entry_px=round(p["entry_px"], 4), exit_px=round(xp, 4), ret=round(ret, 5),
                                   pnl=round(p["sh"] * p["entry_px"] * ret, 2)))
            else:
                keep.append(p)
        s["positions"] = keep
        eq = s["cash"] + sum(sign(p["side"]) * p["sh"] * p["last"] for p in keep)
        s["last_equity"] = eq
        if s["spy_start"] is None:
            s["spy_start"] = float(spy.loc[d])
        book.append("trades.csv", trades)
        book.append("equity.csv", [dict(day=f"{d:%Y-%m-%d}", equity=round(eq, 2), cash=round(s["cash"], 2),
                                         positions=len(keep), spy=round(float(spy.loc[d]), 4))])
        s["last_processed"] = f"{d:%Y-%m-%d}"
        msg = "; ".join(fills + [f"SOLD {t['ticker']} {t['ret'] * 100:+.1f}%" for t in trades])
        log(f"  {d:%Y-%m-%d}  equity ${eq:,.0f}" + (f"  | {msg}" if msg else ""))


def queue_signals(book: Book, posts, universe, now_et: pd.Timestamp, log):
    s, cfg = book.s, book.s["config"]
    now_et = naive_et(now_et)
    today = now_et.normalize()
    cal = wc.trading_days(today - pd.Timedelta(days=HISTORY_DAYS + 10), today + pd.Timedelta(days=30))
    open_today = today + pd.Timedelta(hours=9, minutes=30)
    if today in cal and now_et < open_today:
        entry = today
    else:
        entry = cal[cal > today][0]
    days = cal[cal <= entry]
    daily = wc.aggregate_attention(wc.build_mentions(posts, universe, days))
    daily = daily[daily.ticker != wc.BENCH]
    ev = wc.detect_spikes(daily, days, cfg["lookback"], cfg["z"], cfg["min_mentions"])
    ev = wc.filter_events(ev[ev.day == entry], cfg["filter"]).sort_values("z", ascending=False)
    s["pending"] = [p for p in s["pending"] if pd.Timestamp(p["entry_day"]) != entry]  # idempotent re-runs
    busy = {p["t"] for p in s["positions"]} | {p["t"] for p in s["pending"]}
    ev = ev[~ev.ticker.isin(busy)].head(cfg["max_pos"] + BACKUPS)
    exit_day = wc.add_trading_days(entry, int(cfg["hold"]) - 1)
    new = []
    for rank, r in enumerate(ev.itertuples(index=False)):
        new.append(dict(t=r.ticker, side=cfg["side"], entry_day=f"{entry:%Y-%m-%d}", exit_day=f"{exit_day:%Y-%m-%d}",
                        rank=rank, z=round(float(r.z), 2), mentions=int(r.mentions),
                        sentiment=round(float(r.sentiment), 2)))
    s["pending"] += new
    book.append("signals.csv", [dict(generated=now_et.isoformat(timespec="minutes"), **n) for n in new])
    return entry, new


def naive_et(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(wc.ET).tz_localize(None) if ts.tzinfo else ts


def step(book, posts, universe, price_fn, now_et, log=print):
    """One morning's work: settle everything up to the last completed session, then queue new signals."""
    now_et = naive_et(now_et)
    today = now_et.normalize()
    cal = wc.trading_days(today - pd.Timedelta(days=15), today + pd.Timedelta(days=10))
    closed_today = today in cal and now_et >= today + pd.Timedelta(hours=16, minutes=15)
    completed = today if closed_today else cal[cal < today][-1]
    process_fills(book, price_fn, completed, log)
    entry, new = queue_signals(book, posts, universe, now_et, log)
    book.save()
    return entry, new


# --------------------------------------------------------------------------- display
def print_status(book: Book):
    s = book.s
    if not s:
        print("No paper portfolio yet. Run:  python wsb_paper_trader.py run")
        return
    cfg = s["config"]
    eq = s["last_equity"]
    print(f"\nWSB paper portfolio  (strategy: {cfg['side']} {cfg['filter']} trending posts, hold {cfg['hold']}d, "
          f"max {cfg['max_pos']} positions)")
    print(f"  Equity ${eq:,.2f}   cash ${s['cash']:,.2f}   since start {eq / s['start_capital'] - 1:+.2%}"
          f"   (last settled session: {s['last_processed'] or 'none yet'})")
    eqf = book.dir / "equity.csv"
    if eqf.exists() and s["spy_start"]:
        e = pd.read_csv(eqf)
        print(f"  SPY over the same period: {e.spy.iloc[-1] / s['spy_start'] - 1:+.2%}")
    if s["positions"]:
        print("  Open positions:")
        for p in s["positions"]:
            sgn = 1 if p["side"] == "long" else -1
            print(f"    {p['side']:<5} {p['t']:<6} in {p['entry_day']} @ {p['entry_px']:.2f}  now {p['last']:.2f}  "
                  f"({sgn * (p['last'] / p['entry_px'] - 1):+.1%})  exits at close {p['exit_day']}")
    if s["pending"]:
        print("  Orders waiting for the open:")
        for p in sorted(s["pending"], key=lambda p: (p["entry_day"], p["rank"])):
            print(f"    {p['entry_day']}  #{p['rank'] + 1} {p['side']} {p['t']:<6} z={p['z']}  posts={p['mentions']}  "
                  f"sentiment={p['sentiment']:+.2f}")
    tf = book.dir / "trades.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        print(f"  Closed trades: {len(t)}  win rate {(t.ret > 0).mean():.0%}  avg {t.ret.mean():+.2%}  "
              f"total P&L ${t.pnl.sum():,.2f}")
    print(f"  Files: {book.dir}\n")


# --------------------------------------------------------------------------- commands
def live_price_fn(log):
    return lambda tickers, start, end: wc.fetch_prices(tickers, start, end, log=lambda *_: None)


def cmd_run(args):
    book = Book(wc.DATA_DIR / "paper")
    if not book.s:
        book.init(args.capital, load_cfg())
        print(f"Started a new paper portfolio with ${args.capital:,.0f} using {book.s['config']}")
    now_et = pd.Timestamp.now(tz=wc.ET)
    print("Fetching recent r/wallstreetbets posts...")
    posts = wc.load_or_fetch_posts(now_et - pd.Timedelta(days=HISTORY_DAYS), now_et,
                                   wc.DATA_DIR / "posts_live.csv.gz")
    universe = wc.load_ticker_universe()
    print("Settling trades and checking for new trends...")
    entry, new = step(book, posts, universe, live_price_fn(print), now_et, print)
    print(f"Queued {len(new)} signal(s) for the {entry:%a %b %d} open.")
    print_status(book)


def cmd_replay(args):
    folder = wc.DATA_DIR / ("paper_replay_demo" if args.demo else "paper_replay")
    shutil.rmtree(folder, ignore_errors=True)
    book = Book(folder)
    cfg = load_cfg() if not args.demo else dict(DEFAULT_CFG)
    book.init(args.capital, cfg)
    if args.demo:
        posts, universe, opens, closes = wc.make_demo_data(n_days=110)
        days = closes.index
    else:
        now = pd.Timestamp.now(tz=wc.ET)
        start = now - pd.Timedelta(days=args.days + HISTORY_DAYS)
        posts = wc.load_or_fetch_posts(start, now, wc.DATA_DIR / "posts_live.csv.gz")
        universe = wc.load_ticker_universe()
        days = wc.trading_days(now - pd.Timedelta(days=args.days), now.tz_localize(None))
        days = days[days < now.tz_localize(None).normalize()]
        opens = closes = None
    cache = {}

    def price_fn(tickers, start, end):
        if args.demo:
            cols = [t for t in tickers if t in closes.columns]
            return opens.loc[:end, cols], closes.loc[:end, cols]
        key = tuple(sorted(tickers))
        missing = [t for t in key if t not in cache.get("cols", set())]
        if missing:
            o, c = wc.fetch_prices(sorted(set(missing) | cache.get("cols", set())),
                                   days[0] - pd.Timedelta(days=15), days[-1], log=lambda *_: None)
            cache.update(o=o, c=c, cols=set(c.columns))
        cols = [t for t in tickers if t in cache["c"].columns]
        return cache["o"].loc[:end, cols], cache["c"].loc[:end, cols]

    ts = posts.created_utc.to_numpy()
    warm = 30 if args.demo else 0
    for d in days[warm:]:
        now_et = d + pd.Timedelta(hours=9, minutes=29)
        cut = (now_et.tz_localize(wc.ET)).timestamp()
        lo = cut - HISTORY_DAYS * 86400
        p = posts[(ts < cut) & (ts >= lo)]
        step(book, p, universe, price_fn, now_et, log=print if args.verbose else (lambda *_: None))
    print_status(book)
    if args.demo:   # cross-check against the vectorised backtest
        import wsb_analyze as wa
        cal = closes.index
        daily = wc.aggregate_attention(wc.build_mentions(posts, universe, cal))
        daily = daily[daily.ticker != wc.BENCH]
        ev = wc.filter_events(wc.detect_spikes(daily, cal, cfg["lookback"], cfg["z"], cfg["min_mentions"]), cfg["filter"])
        sub = cal[warm:cal.get_loc(pd.Timestamp(book.s["last_processed"])) + 1]
        eq, tr = wa.backtest(ev[ev.day.isin(sub)], opens, closes, sub, cfg["hold"], cfg["side"], cfg["max_pos"], cfg["cost_bps"], args.capital)
        t = pd.read_csv(book.dir / "trades.csv") if (book.dir / "trades.csv").exists() else pd.DataFrame()
        print(f"Cross-check vs backtest over the same days: backtest {len(tr)} trades / equity ${eq.iloc[-1]:,.2f};"
              f" paper trader {len(t)} closed trades (+{len(book.s['positions'])} open) / equity ${book.s['last_equity']:,.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="settle fills and queue today's signals")
    r.add_argument("--capital", type=float, default=100_000)
    sub.add_parser("status", help="show the portfolio")
    z = sub.add_parser("reset", help="start over with fresh paper money")
    z.add_argument("--capital", type=float, default=100_000)
    rp = sub.add_parser("replay", help="simulate past mornings with the same code path")
    rp.add_argument("--days", type=int, default=60)
    rp.add_argument("--capital", type=float, default=100_000)
    rp.add_argument("--demo", action="store_true")
    rp.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "status":
        print_status(Book(wc.DATA_DIR / "paper"))
    elif a.cmd == "reset":
        b = Book(wc.DATA_DIR / "paper")
        b.init(a.capital, load_cfg())
        print(f"Reset. New paper portfolio: ${a.capital:,.0f}, strategy {b.s['config']}")
    elif a.cmd == "replay":
        cmd_replay(a)


if __name__ == "__main__":
    main()
