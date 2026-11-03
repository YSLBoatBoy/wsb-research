"""
wsb_analyze.py - Does a ticker trending on r/wallstreetbets predict its stock price?

  python wsb_analyze.py                 # last ~3 months of real data (needs internet)
  python wsb_analyze.py --days 150      # longer history
  python wsb_analyze.py --demo          # offline test on synthetic data

What it does
  1. Downloads every r/wallstreetbets post in the window and finds the tickers in them.
  2. Builds a daily "attention" score per ticker (posts, upvotes, comments).
  3. Flags "trending" days: attention far above that ticker's own recent normal.
  4. Measures what the stock did AFTER it trended (vs. SPY), and BEFORE it trended
     (does hype follow price, or lead it?).
  5. Backtests simple paper strategies (buy/short trending tickers, several holding
     periods) with trading costs, and tests the best one on data it was not tuned on.
  6. Writes report.html, CSVs, and best_config.json (used by wsb_paper_trader.py).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from html import escape

import numpy as np
import pandas as pd

import wsb_common as wc

HORIZONS = [1, 3, 5, 10]


# --------------------------------------------------------------------------- statistics
def forward_abnormal(opens, closes, h):
    """Return from the entry-day open to the close h-1 days later, minus SPY's same return."""
    r = closes.shift(-(h - 1)) / opens - 1
    return r.sub(r[wc.BENCH], axis=0).drop(columns=wc.BENCH)


def pre_runup(closes, k=5):
    """Return over the k sessions BEFORE the entry day (close t-1 vs close t-1-k), minus SPY."""
    r = closes.shift(1) / closes.shift(1 + k) - 1
    return r.sub(r[wc.BENCH], axis=0).drop(columns=wc.BENCH)


def lookup(wide, events):
    out = []
    for d, t in zip(events.day, events.ticker):
        out.append(wide.at[d, t] if (t in wide.columns and d in wide.index) else np.nan)
    return np.array(out, dtype=float)


def bootstrap_p(observed_vals, pool, n_boot=4000, seed=1):
    """Two-sided p-value: how often a random set of (same tickers, random days) is as extreme."""
    observed_vals = observed_vals[~np.isnan(observed_vals)]
    pool = pool[~np.isnan(pool)]
    if len(observed_vals) < 3 or len(pool) < 20:
        return np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(pool, (n_boot, len(observed_vals))).mean(axis=1)
    center = pool.mean()
    obs = observed_vals.mean()
    return float((np.abs(means - center) >= abs(obs - center)).mean())


def event_study(events, opens, closes):
    tick = sorted(set(events.ticker) & set(closes.columns))
    rows = []
    specs = [(f"Before: 5 days leading up", pre_runup(closes, 5))] + \
            [(f"After: {h} day{'s' if h > 1 else ''}", forward_abnormal(opens, closes, h)) for h in HORIZONS]
    per_event = events.copy()
    for label, wide in specs:
        vals = lookup(wide, events)
        per_event[label] = vals
        v = vals[~np.isnan(vals)]
        pool = wide[tick].to_numpy().ravel() if tick else np.array([])
        n = len(v)
        rows.append({
            "window": label,
            "events": n,
            "avg_vs_spy": v.mean() if n else np.nan,
            "median_vs_spy": np.median(v) if n else np.nan,
            "beat_spy_rate": (v > 0).mean() if n else np.nan,
            "t_stat": v.mean() / (v.std(ddof=1) / math.sqrt(n)) if n > 2 and v.std() > 0 else np.nan,
            "normal_day_avg": np.nanmean(pool) if len(pool) else np.nan,
            "p_value": bootstrap_p(v, pool),
        })
    return pd.DataFrame(rows), per_event


def abnormal_attention(daily, days, tickers, lookback):
    a = daily.pivot_table(index="day", columns="ticker", values="attn", aggfunc="sum").reindex(days).fillna(0.0)
    a = a.reindex(columns=tickers).fillna(0.0)
    la = np.log1p(a)
    return la - la.shift(1).rolling(lookback, min_periods=5).mean()


def spearman_pooled(x: pd.DataFrame, y: pd.DataFrame):
    s = pd.concat([x.stack(), y.stack()], axis=1, keys=["x", "y"]).dropna()
    if len(s) < 30:
        return np.nan, len(s)
    return s["x"].rank().corr(s["y"].rank()), len(s)


def lead_lag(abn_attn, closes, max_lag=5):
    """Correlation of abnormal attention on day t with the stock's SPY-adjusted return on day t+k.
    k<0: returns BEFORE the hype (hype chasing price).  k>=1: returns AFTER (predictive power)."""
    r = closes.pct_change(fill_method=None)
    r = r.sub(r[wc.BENCH], axis=0)[abn_attn.columns]
    rows = []
    for k in range(-max_lag, max_lag + 1):
        c, n = spearman_pooled(abn_attn, r.shift(-k))
        rows.append({"lag_days": k, "spearman": c, "n": n})
    return pd.DataFrame(rows)


def quintiles(abn_attn, opens, closes, h=5):
    fwd = forward_abnormal(opens, closes, h)[abn_attn.columns]
    s = pd.concat([abn_attn.stack(), fwd.stack()], axis=1, keys=["a", "r"]).dropna()
    s = s[s.a != 0]
    if len(s) < 50:
        return pd.DataFrame(columns=["bucket", "avg_fwd", "n"])
    s["bucket"] = pd.qcut(s.a.rank(method="first"), 5, labels=["1 (quietest)", "2", "3", "4", "5 (hottest)"])
    g = s.groupby("bucket", observed=True)["r"]
    return pd.DataFrame({"avg_fwd": g.mean(), "n": g.size()}).reset_index()


# --------------------------------------------------------------------------- backtest
def backtest(events, opens, closes, days, hold=5, side="long", max_pos=10, cost_bps=10,
             capital=100_000.0):
    """Daily simulation. Signal on entry day d -> buy (or short) at d's open, exit at the
    close hold-1 sessions later. Equal-size slots of equity/max_pos. Costs both ways."""
    s = 1 if side == "long" else -1
    cf = cost_bps / 1e4
    closes_ff = closes.ffill(limit=5)
    by_day = {d: g.sort_values("z", ascending=False) for d, g in events.groupby("day")}
    cash, pos, eq_hist, trades = capital, [], [], []
    last_eq = capital
    for i, d in enumerate(days):
        if d in by_day:
            held = {p["t"] for p in pos}
            for r in by_day[d].itertuples(index=False):
                if len(pos) >= max_pos:
                    break
                t = r.ticker
                if t in held or t not in opens.columns:
                    continue
                px = opens.at[d, t] if d in opens.index else np.nan
                if not (px > 0):
                    continue
                notional = last_eq / max_pos
                sh = notional / px
                cash -= s * sh * px + notional * cf
                pos.append(dict(t=t, sh=sh, px=px, d=d, exit_i=i + hold - 1, last=px))
                held.add(t)
        keep = []
        for p in pos:
            c = closes_ff.at[d, p["t"]] if d in closes_ff.index else np.nan
            if c > 0:
                p["last"] = c
            if p["exit_i"] <= i or i == len(days) - 1:
                xp = p["last"]
                cash += s * p["sh"] * xp - p["sh"] * xp * cf
                trades.append(dict(ticker=p["t"], side=side, entry_day=p["d"], exit_day=d,
                                   entry_px=p["px"], exit_px=xp,
                                   ret=s * (xp / p["px"] - 1) - cf - cf * xp / p["px"]))
            else:
                keep.append(p)
        pos = keep
        last_eq = cash + sum(s * p["sh"] * p["last"] for p in pos)
        eq_hist.append(last_eq)
    eq = pd.Series(eq_hist, index=days, dtype=float)
    return eq, pd.DataFrame(trades)


def perf(eq, trades, spy_close):
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    spy = spy_close.reindex(eq.index).ffill()
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0.0
    return {
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "spy_return": spy.iloc[-1] / spy.iloc[0] - 1,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "trades": len(trades),
        "win_rate": (trades.ret > 0).mean() if len(trades) else np.nan,
        "avg_trade": trades.ret.mean() if len(trades) else np.nan,
    }


def run_grid(events, opens, closes, days, max_pos, cost_bps, min_trades=8):
    rows, curves = [], {}
    for flt in ["all", "bullish", "dd"]:
        ev = wc.filter_events(events, flt)
        for side in ["long", "short"]:
            for hold in HORIZONS:
                eq, tr = backtest(ev, opens, closes, days, hold, side, max_pos, cost_bps)
                p = perf(eq, tr, closes[wc.BENCH])
                name = f"{side} {flt} posts, hold {hold}d"
                rows.append(dict(strategy=name, side=side, filter=flt, hold=hold, **p))
                curves[name] = eq
    df = pd.DataFrame(rows)
    df["eligible"] = df.trades >= min_trades
    return df, curves


# --------------------------------------------------------------------------- report
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def pct(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{nd}f}%"


def table_html(df, fmt):
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(f"<td>{escape(fmt.get(c, str)(v))}</td>" for c, v in r.items()) + "</tr>"
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def make_report(path, ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    imgs = {}

    ll = ctx["lead_lag"]
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#9aa4b2" if k < 1 else "#2f6fde" for k in ll.lag_days]
    ax.bar(ll.lag_days, ll.spearman, color=colors)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xlabel("days relative to the hype (negative = before, positive = after)")
    ax.set_ylabel("correlation")
    ax.set_title("Does hype follow the price, or predict it?")
    imgs["leadlag"] = fig_b64(fig)

    q = ctx["quintiles"]
    if len(q):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(q.bucket.astype(str), q.avg_fwd * 100, color="#2f6fde")
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_ylabel("avg next-5-day return vs SPY (%)")
        ax.set_xlabel("ticker-days sorted by how unusually hot they were on WSB")
        imgs["quint"] = fig_b64(fig)

    es = ctx["event_df"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(es.window, es.avg_vs_spy * 100, color=["#9aa4b2"] + ["#2f6fde"] * (len(es) - 1))
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_ylabel("avg return vs SPY (%)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Trending tickers: before vs after the hype")
    imgs["event"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    spy = ctx["spy_close"].reindex(ctx["days"]).ffill()
    ax.plot(spy.index, spy / spy.iloc[0] * 100, color="#333", lw=1.2, label="SPY (buy & hold)")
    for name, c in ctx["show_curves"].items():
        ax.plot(c.index, c / c.iloc[0] * 100, lw=1.3, label=name)
    if ctx["split_day"] is not None:
        ax.axvline(ctx["split_day"], color="#c33", ls="--", lw=0.8)
        ax.text(ctx["split_day"], ax.get_ylim()[1], "  tuning | test", color="#c33", va="top", fontsize=8)
    ax.set_ylabel("value of $100")
    ax.legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    imgs["equity"] = fig_b64(fig)

    g = ctx["grid"].copy()
    g = g.sort_values("sharpe", ascending=False)
    grid_tbl = table_html(
        g[["strategy", "total_return", "spy_return", "sharpe", "max_drawdown", "trades", "win_rate", "avg_trade"]],
        {"total_return": pct, "spy_return": pct, "max_drawdown": pct, "win_rate": lambda v: pct(v, 0)[1:],
         "avg_trade": pct, "sharpe": lambda v: f"{v:.2f}", "trades": lambda v: f"{int(v)}"})
    ev_tbl = table_html(
        es[["window", "events", "avg_vs_spy", "median_vs_spy", "beat_spy_rate", "normal_day_avg", "p_value"]],
        {"avg_vs_spy": pct, "median_vs_spy": pct, "beat_spy_rate": lambda v: pct(v, 0)[1:],
         "normal_day_avg": pct, "p_value": lambda v: "–" if np.isnan(v) else f"{v:.3f}",
         "events": lambda v: f"{int(v)}"})
    top = ctx["top_tickers"]
    top_tbl = table_html(top, {"posts": lambda v: f"{int(v)}", "trend_days": lambda v: f"{int(v)}"})
    wf = ctx["walk_forward"]

    findings = "".join(f"<li>{f}</li>" for f in ctx["findings"])
    demo_banner = ("<div class='banner'>DEMO RUN on synthetic data - these numbers are made up and only show "
                   "that the pipeline works. Run without --demo for real results.</div>") if ctx["demo"] else ""
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSB Hype vs Returns</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d2329;--mut:#5f6b78;--line:#e3e5e8;--acc:#2f6fde}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16191d;--fg:#e8eaed;--mut:#9aa4b2;--line:#2c3138}} img{{background:#fff;border-radius:6px}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 8px;border-top:1px solid var(--line);padding-top:20px}}
.mut{{color:var(--mut);font-size:13px}} img{{max-width:100%;display:block;margin:10px 0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--mut);font-weight:600}}
.banner{{background:#fff3cd;color:#664d03;padding:10px 14px;border-radius:6px;margin:12px 0}}
.verdict{{border-left:4px solid var(--acc);padding:4px 14px;margin:14px 0}}
</style></head><body><main>
<h1>Does r/wallstreetbets hype predict stock returns?</h1>
<div class="mut">{escape(ctx['subtitle'])}</div>
{demo_banner}
<div class="verdict"><ul>{findings}</ul></div>

<h2>1. What happened around trending days</h2>
<p class="mut">A "trending day" is when a ticker's WSB attention was at least {ctx['z']} standard deviations above its own
trailing {ctx['lookback']}-day average, with at least {ctx['min_mentions']} posts. Returns are measured from the first market
open after the posts (when you could actually act) and shown relative to SPY. <b>p_value</b> compares against the same
tickers on random days (bootstrap); below 0.05 is the usual bar for "unlikely to be luck".</p>
<img src="data:image/png;base64,{imgs['event']}">
{ev_tbl}

<h2>2. Hype vs price, day by day</h2>
<p class="mut">Grey bars: correlation between unusual attention and the stock's move on earlier days (hype chasing price).
Blue bars: correlation with later days (hype predicting price). Day 0 includes the overnight move, so it mixes both.</p>
<img src="data:image/png;base64,{imgs['leadlag']}">
{f'<p class="mut">Every ticker-day sorted into five equal groups by how unusually hot it was, then the average next-5-day return vs SPY for each group.</p><img src="data:image/png;base64,{imgs["quint"]}">' if 'quint' in imgs else ''}

<h2>3. Paper-trading backtest</h2>
<p class="mut">Start $100,000. On each trending day, open up to {ctx['max_pos']} equal-size positions at the open, hold for N days,
exit at the close. {ctx['cost_bps']} bps cost charged on every buy and sell. Short = betting the hype fades.</p>
<img src="data:image/png;base64,{imgs['equity']}">
<p><b>Honest test (walk-forward):</b> {wf}</p>
{grid_tbl}

<h2>4. Most-discussed tickers</h2>
{top_tbl}

<h2>Caveats</h2>
<ul class="mut">
<li>Three months is a short window; one meme rally can dominate the averages. Re-run with --days 250+ for more confidence.</li>
<li>Upvote counts in the archive are often captured soon after posting, so they understate how popular a post became.</li>
<li>Ticker detection is pattern-based; words like "AI" or "NOW" only count as $AI / $NOW. Some noise remains.</li>
<li>Trending events overlap in time (the whole market moves together), so t-stats overstate certainty; the bootstrap p-value is the better guide.</li>
<li>Testing 24 strategies and reporting the best guarantees one looks good by luck. Trust only the walk-forward (test-half) number.</li>
<li>Fills assume you get the exact opening price; meme stocks often gap and have wide spreads. Shorting needs borrow, which can be costly or unavailable.</li>
</ul>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=120, help="calendar days of history (default 120: ~3 months + warm-up)")
    ap.add_argument("--lookback", type=int, default=20, help="trading days used as each ticker's 'normal' attention")
    ap.add_argument("--z", type=float, default=2.0, help="how unusual attention must be to count as trending")
    ap.add_argument("--min-mentions", type=int, default=5, help="minimum posts on the day to count as trending")
    ap.add_argument("--min-total", type=int, default=8, help="ignore tickers with fewer total posts than this")
    ap.add_argument("--max-pos", type=int, default=10, help="max simultaneous positions in the backtest")
    ap.add_argument("--cost-bps", type=float, default=10, help="trading cost per side in basis points")
    ap.add_argument("--out", default=str(wc.DATA_DIR / "report"), help="output folder")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline test)")
    a = ap.parse_args()
    log = print
    out = wc.ensure_dir(wc.Path(a.out))

    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=a.days)
    print("1/6  Getting r/wallstreetbets posts")
    if a.demo:
        posts, universe, opens, closes = wc.make_demo_data(n_days=int(a.days * 252 / 365))
        cal = closes.index
    else:
        posts = wc.load_or_fetch_posts(start, now, wc.DATA_DIR / "posts.csv.gz", log=log)
        universe = wc.load_ticker_universe(log=log)
        cal = wc.trading_days(start - pd.Timedelta(days=5), now.tz_convert(wc.ET).tz_localize(None) + pd.Timedelta(days=10))
    print(f"     {len(posts):,} posts")
    if posts.empty:
        raise SystemExit("No posts downloaded - check your internet connection.")

    print("2/6  Finding tickers and building daily attention")
    mentions = wc.build_mentions(posts, universe, cal)
    daily = wc.aggregate_attention(mentions)
    totals = daily.groupby("ticker")["mentions"].sum().sort_values(ascending=False)
    keep = totals[totals >= a.min_total].index.tolist()
    keep = [t for t in keep if t != wc.BENCH] if len(keep) > 1 else keep
    print(f"     {len(totals):,} tickers mentioned; {len(keep)} with >= {a.min_total} posts")

    print("3/6  Downloading prices")
    if not a.demo:
        opens, closes = wc.fetch_prices(keep, start - pd.Timedelta(days=45), now.tz_convert(wc.ET).tz_localize(None), log=log)
    days = closes.index[(closes.index >= cal.min()) & (closes.index <= cal.max())]
    days = days[closes.loc[days, wc.BENCH].notna()]
    tick = [t for t in keep if t in closes.columns]
    opens, closes = opens.loc[days, tick + [wc.BENCH]], closes.loc[days, tick + [wc.BENCH]]
    daily = daily[daily.ticker.isin(tick)]

    print("4/6  Detecting trending days and measuring returns")
    events = wc.detect_spikes(daily, days, a.lookback, a.z, a.min_mentions)
    print(f"     {len(events)} trending events across {events.ticker.nunique() if len(events) else 0} tickers")
    if len(events) < 5:
        print("     Very few events - try --days 200, --z 1.5 or --min-mentions 3.")
    event_df, per_event = event_study(events, opens, closes)
    abn = abnormal_attention(daily, days, tick, a.lookback)
    ll = lead_lag(abn, closes)
    qt = quintiles(abn, opens, closes)

    print("5/6  Backtesting paper strategies")
    grid, curves = run_grid(events, opens, closes, days, a.max_pos, a.cost_bps)
    # walk-forward: pick the best config on the first half, judge it on the second half
    split = len(days) // 2
    d1, d2 = days[:split], days[split:]
    tr_grid, _ = run_grid(events[events.day.isin(d1)], opens, closes, d1, a.max_pos, a.cost_bps, min_trades=5)
    elig = tr_grid[tr_grid.eligible]
    best = (elig if len(elig) else tr_grid).sort_values("sharpe", ascending=False).iloc[0]
    ev2 = wc.filter_events(events[events.day.isin(d2)], best["filter"])
    eq2, tr2 = backtest(ev2, opens, closes, d2, int(best.hold), best.side, a.max_pos, a.cost_bps)
    p2 = perf(eq2, tr2, closes[wc.BENCH])
    wf_text = (f"tuned on {d1[0]:%b %d}–{d1[-1]:%b %d}, the best setup was <b>{escape(best.strategy)}</b> "
               f"(Sharpe {best.sharpe:.2f}). On the unseen {d2[0]:%b %d}–{d2[-1]:%b %d} period it returned "
               f"<b>{pct(p2.get('total_return', np.nan))}</b> over {p2.get('trades', 0)} trades vs SPY "
               f"{pct(p2.get('spy_return', np.nan))} (Sharpe {p2.get('sharpe', 0):.2f}).")

    # findings in plain English
    f = []
    after5 = event_df[event_df.window == "After: 5 days"].iloc[0]
    before = event_df.iloc[0]
    f.append(f"Across {int(after5.events)} trending events, the 5 days <b>before</b> the hype averaged "
             f"{pct(before.avg_vs_spy)} vs SPY (p={before.p_value:.3f}); the 5 days <b>after</b> averaged "
             f"{pct(after5.avg_vs_spy)} (p={after5.p_value:.3f}).")
    past = ll[ll.lag_days < 0].spearman.abs().mean()
    fut = ll[ll.lag_days > 0].spearman.abs().mean()
    f.append("Attention is more strongly tied to <b>past</b> price moves than future ones - hype mostly follows the price."
             if past > fut else "Attention is at least as tied to <b>future</b> price moves as past ones.")
    sig = event_df[(event_df.window.str.startswith("After")) & (event_df.p_value < 0.05)]
    if len(sig):
        f.append("Statistically notable after-hype windows: " + ", ".join(
            f"{w} ({pct(v)})" for w, v in zip(sig.window, sig.avg_vs_spy)) + ".")
    else:
        f.append("None of the after-hype windows differ from random days by more than chance would explain (p &lt; 0.05).")
    beat = p2.get("total_return", 0) > p2.get("spy_return", 0) and p2.get("sharpe", 0) > 0
    f.append(("The walk-forward strategy beat SPY on unseen data - interesting, but one short test period is not proof. "
              "Paper trade it for a few months before believing it.")
             if beat else "The best tuned strategy did <b>not</b> beat SPY on unseen data - no reliable edge found here.")

    best_full = grid[grid.eligible].sort_values("sharpe", ascending=False)
    show = {}
    if len(best_full):
        show[best_full.iloc[0].strategy + " (best in hindsight)"] = curves[best_full.iloc[0].strategy]
    base_name = "long all posts, hold 5d"
    if base_name in curves and base_name not in [k.split(" (")[0] for k in show]:
        show[base_name] = curves[base_name]
    show["walk-forward pick (test half, restarts at 100)"] = eq2

    top = (daily.groupby("ticker").agg(posts=("mentions", "sum")).join(
        events.groupby("ticker").size().rename("trend_days")).fillna(0)
        .sort_values("posts", ascending=False).head(20).reset_index())

    print("6/6  Writing report")
    subtitle = (f"{'DEMO · ' if a.demo else ''}{len(posts):,} posts · {days[0]:%b %d, %Y} – {days[-1]:%b %d, %Y} · "
                f"{len(tick)} tickers · generated {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    make_report(out / "report.html", dict(
        demo=a.demo, subtitle=subtitle, findings=f, event_df=event_df, lead_lag=ll, quintiles=qt,
        grid=grid, show_curves=show, spy_close=closes[wc.BENCH], days=days, split_day=d2[0],
        walk_forward=wf_text, top_tickers=top, z=a.z, lookback=a.lookback, min_mentions=a.min_mentions,
        max_pos=a.max_pos, cost_bps=a.cost_bps))
    daily.to_csv(out / "daily_attention.csv", index=False)
    per_event.to_csv(out / "trending_events.csv", index=False)
    grid.to_csv(out / "strategy_results.csv", index=False)
    ll.to_csv(out / "lead_lag.csv", index=False)
    cfg = dict(hold=int(best.hold), side=best.side, filter=best["filter"], z=a.z, lookback=a.lookback,
               min_mentions=a.min_mentions, max_pos=a.max_pos, cost_bps=a.cost_bps,
               chosen_on=f"{d1[0]:%Y-%m-%d}..{d1[-1]:%Y-%m-%d}",
               test_result=dict(total_return=p2.get("total_return"), spy_return=p2.get("spy_return"),
                                trades=p2.get("trades")))
    (out / "best_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    if not a.demo:
        (wc.DATA_DIR / "best_config.json").write_text(json.dumps(cfg, indent=2, default=float))

    print("\n" + "=" * 70)
    for line in f:
        print(" -", line.replace("<b>", "").replace("</b>", "").replace("&lt;", "<"))
    print("=" * 70)
    print(f"Report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
