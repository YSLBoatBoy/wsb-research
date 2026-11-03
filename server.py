"""
server.py - container entrypoint: job scheduler + web dashboard.

Schedule (container time zone, set with TZ):
  * Paper trader:  every weekday at PAPER_TIME (default 08:15, i.e. before the 09:30 ET open)
  * Analysis:      weekly on ANALYZE_DAY at ANALYZE_TIME (default Sunday 18:00),
                   and once at startup if no report exists yet.
Dashboard on port 8080: portfolio, equity vs SPY, trades, report, logs, "run now" buttons.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

APP = Path(__file__).resolve().parent
DATA = Path(os.environ.get("WSB_DATA_DIR", "/data"))
LOGS = DATA / "logs"
TZ = ZoneInfo(os.environ.get("TZ", "America/Chicago"))
PAPER_TIME = os.environ.get("PAPER_TIME", "08:15")
ANALYZE_DAY = os.environ.get("ANALYZE_DAY", "sun").lower()[:3]
ANALYZE_TIME = os.environ.get("ANALYZE_TIME", "18:00")
ANALYZE_DAYS = os.environ.get("ANALYZE_DAYS", "120")
CAPITAL = os.environ.get("PAPER_CAPITAL", "100000")
PORT = int(os.environ.get("PORT", "8080"))
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
JOBS = {
    "analyze": ["wsb_analyze.py", "--days", ANALYZE_DAYS],
    "paper": ["wsb_paper_trader.py", "run", "--capital", CAPITAL],
}

lock = threading.Lock()
STATE_FILE = DATA / "server_state.json"
state = {"running": None, "last": {}, "done": {}}


def now():
    return dt.datetime.now(TZ)


def save_state():
    try:
        STATE_FILE.write_text(json.dumps({k: state[k] for k in ("last", "done")}, indent=2))
    except OSError:
        pass


def run_job(name: str) -> bool:
    if not lock.acquire(blocking=False):
        return False

    def go():
        state["running"] = name
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"{name}.log"
        # keep logs small
        if log.exists() and log.stat().st_size > 2_000_000:
            log.write_text(log.read_text(errors="replace")[-500_000:])
        started = now()
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n===== {started:%Y-%m-%d %H:%M:%S %Z} - {name} =====\n")
                f.flush()
                rc = subprocess.run([sys.executable, str(APP / JOBS[name][0]), *JOBS[name][1:]],
                                    stdout=f, stderr=subprocess.STDOUT, cwd=APP).returncode
                f.write(f"===== finished with exit code {rc} =====\n")
        except Exception as e:  # noqa: BLE001
            rc = -1
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"failed to start: {e}\n")
        state["last"][name] = {"at": started.isoformat(timespec="seconds"), "ok": rc == 0,
                               "minutes": round((now() - started).total_seconds() / 60, 1)}
        state["running"] = None
        save_state()
        lock.release()

    threading.Thread(target=go, daemon=True).start()
    return True


def due(key: str, hhmm: str, n: dt.datetime, window_h=3) -> bool:
    h, m = map(int, hhmm.split(":"))
    at = n.replace(hour=h, minute=m, second=0, microsecond=0)
    return at <= n < at + dt.timedelta(hours=window_h) and state["done"].get(key) != n.date().isoformat()


def scheduler():
    if not (DATA / "report" / "report.html").exists():
        print("No report yet - running the analysis once at startup.", flush=True)
        run_job("analyze")
    while True:
        n = now()
        if n.weekday() < 5 and due("paper", PAPER_TIME, n) and run_job("paper"):
            state["done"]["paper"] = n.date().isoformat()
            save_state()
        if DAYS[n.weekday()] == ANALYZE_DAY and due("analyze", ANALYZE_TIME, n, 12) and run_job("analyze"):
            state["done"]["analyze"] = n.date().isoformat()
            save_state()
        time.sleep(30)


# --------------------------------------------------------------------------- dashboard
def read_csv(p: Path):
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def money(x):
    x = float(x)
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def pct(x):
    return f"{float(x) * 100:+.2f}%"


def equity_svg(rows):
    if len(rows) < 2:
        return "<p class='mut'>The equity chart appears after a couple of trading days.</p>"
    eq = [float(r["equity"]) for r in rows]
    spy = [float(r["spy"]) for r in rows]
    a = [v / eq[0] * 100 for v in eq]
    b = [v / spy[0] * 100 for v in spy]
    lo, hi = min(a + b), max(a + b)
    pad = (hi - lo) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    W, H = 760, 240

    def pts(vals):
        return " ".join(f"{i / (len(vals) - 1) * W:.1f},{H - (v - lo) / (hi - lo) * H:.1f}" for i, v in enumerate(vals))
    y100 = H - (100 - lo) / (hi - lo) * H
    return f"""<svg viewBox="0 0 {W} {H + 22}" class="chart" role="img" aria-label="Equity vs SPY">
<line x1="0" x2="{W}" y1="{y100:.1f}" y2="{y100:.1f}" class="grid"/>
<polyline points="{pts(b)}" class="spy"/><polyline points="{pts(a)}" class="eq"/>
<text x="0" y="{H + 18}" class="lbl">{escape(rows[0]['day'])}</text>
<text x="{W}" y="{H + 18}" text-anchor="end" class="lbl">{escape(rows[-1]['day'])}</text></svg>
<div class="legend"><span class="k eq"></span>Paper portfolio {a[-1] - 100:+.2f}%
<span class="k spy"></span>SPY {b[-1] - 100:+.2f}%</div>"""


def table(head, rows):
    if not rows:
        return "<p class='mut'>None.</p>"
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def dashboard() -> str:
    book_f = DATA / "paper" / "book.json"
    book = json.loads(book_f.read_text()) if book_f.exists() else None
    eq_rows = read_csv(DATA / "paper" / "equity.csv")
    trades = read_csv(DATA / "paper" / "trades.csv")
    cfg_f = DATA / "best_config.json"
    best = json.loads(cfg_f.read_text()) if cfg_f.exists() else None

    if book:
        c = book["config"]
        sign = lambda s: 1 if s == "long" else -1
        summary = f"""<div class="stats">
<div><b>{money(book['last_equity'])}</b><span>equity</span></div>
<div><b>{pct(book['last_equity'] / book['start_capital'] - 1)}</b><span>since start</span></div>
<div><b>{(pct(float(eq_rows[-1]['spy']) / book['spy_start'] - 1) if eq_rows and book.get('spy_start') else '–')}</b><span>SPY same period</span></div>
<div><b>{len(trades)}</b><span>closed trades</span></div>
<div><b>{(f"{sum(float(t['ret']) > 0 for t in trades) / len(trades):.0%}" if trades else '–')}</b><span>win rate</span></div></div>
<p class="mut">Strategy: {escape(c['side'])} tickers trending on WSB ({escape(c['filter'])} posts), hold {c['hold']} trading days,
max {c['max_pos']} positions, {c['cost_bps']} bps cost per side. Started {escape(book['created'][:10])}, last settled session
{escape(book['last_processed'] or 'none yet')}.</p>"""
        pos = [[p["side"], p["t"], p["entry_day"], f"{p['entry_px']:.2f}", f"{p['last']:.2f}",
                pct(sign(p["side"]) * (p["last"] / p["entry_px"] - 1)), p["exit_day"]] for p in book["positions"]]
        pend = [[p["entry_day"], p["rank"] + 1, p["side"], p["t"], p["z"], p["mentions"], f"{p['sentiment']:+.2f}"]
                for p in sorted(book["pending"], key=lambda p: (p["entry_day"], p["rank"]))]
        tr = [[t["exit_day"], t["side"], t["ticker"], t["entry_day"], f"{float(t['entry_px']):.2f}", f"{float(t['exit_px']):.2f}", pct(t["ret"]),
               money(t["pnl"])] for t in reversed(trades[-40:])]
        body = f"""{summary}{equity_svg(eq_rows)}
<h2>Open positions</h2>{table(['side', 'ticker', 'bought', 'price', 'now', 'P&L', 'sells at close'], pos)}
<h2>Orders waiting for the open</h2>
<p class="mut">Ranked by how unusual the attention is (z). Only as many fill as there are free slots.</p>
{table(['open', '#', 'side', 'ticker', 'z', 'posts', 'sentiment'], pend)}
<h2>Recent closed trades</h2>{table(['sold', 'side', 'ticker', 'bought', 'buy', 'sell', 'return', 'P&L'], tr)}"""
    else:
        body = (f"<p>No paper portfolio yet. It starts at the next scheduled run ({PAPER_TIME} on a weekday) "
                "or when you press <b>Run paper trader now</b>.</p>")

    def last(name):
        l = state["last"].get(name)
        if state["running"] == name:
            return "running now…"
        return f"{l['at'][:16].replace('T', ' ')} · {'ok' if l['ok'] else 'FAILED - see log'} · {l['minutes']} min" if l else "never"

    tested = ""
    if best and best.get("test_result", {}).get("total_return") is not None:
        t = best["test_result"]
        tested = (f"<p class='mut'>Latest study's strategy pick: <b>{escape(best['side'])} {escape(best['filter'])}, "
                  f"hold {best['hold']}d</b> - on unseen data it returned {pct(t['total_return'])} vs SPY "
                  f"{pct(t['spy_return'])} over {t['trades']} trades. A new paper portfolio uses this pick; "
                  f"the running one keeps its original rules (use Reset to switch).</p>")
    refresh = '<meta http-equiv="refresh" content="15">' if state["running"] else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>WSB Paper Trader</title><style>
:root{{--bg:#fbfbfa;--fg:#1d2329;--mut:#5f6b78;--line:#e3e5e8;--card:#fff;--acc:#2f6fde;--spy:#8a94a3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16191d;--fg:#e8eaed;--mut:#9aa4b2;--line:#2c3138;--card:#1e2227;--acc:#6b9bff;--spy:#7d8796}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:24px 16px 60px}} h1{{font-size:24px;margin:0}} h2{{font-size:17px;margin:28px 0 6px}}
.mut{{color:var(--mut);font-size:13px}} a{{color:var(--acc)}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}}
button{{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
button:disabled{{opacity:.5;cursor:default}} form{{margin:0}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:16px 0}}
.stats div{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.stats b{{display:block;font-size:20px}} .stats span{{color:var(--mut);font-size:12px}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}}
.chart polyline{{fill:none;stroke-width:2}} .chart .eq{{stroke:var(--acc)}} .chart .spy{{stroke:var(--spy);stroke-dasharray:4 3}}
.chart .grid{{stroke:var(--line)}} .lbl{{fill:var(--mut);font-size:11px}}
.legend{{font-size:13px;color:var(--mut);margin:6px 0}} .k{{display:inline-block;width:14px;height:3px;margin:0 6px 3px 12px;vertical-align:middle}}
.k.eq{{background:var(--acc)}} .k.spy{{background:var(--spy)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:600}}
</style></head><body><main>
<h1>WSB Paper Trader</h1>
<div class="mut">Fake money only · {now():%a %b %d, %H:%M %Z}</div>
<div class="bar">
<a href="/report">Open latest correlation report</a> ·
<form method="post" action="/run/paper"><button {'disabled' if state['running'] else ''}>Run paper trader now</button></form>
<form method="post" action="/run/analyze"><button {'disabled' if state['running'] else ''}>Re-run study now</button></form>
<form method="post" action="/reset" onsubmit="return confirm('Wipe the paper portfolio and start over with fresh paper money?')"><button {'disabled' if state['running'] else ''}>Reset portfolio</button></form>
</div>
<div class="mut">Paper trader: {escape(last('paper'))} (<a href="/log/paper">log</a>) · runs weekdays {PAPER_TIME}<br>
Study: {escape(last('analyze'))} (<a href="/log/analyze">log</a>) · runs {ANALYZE_DAY.title()} {ANALYZE_TIME}</div>
{tested}{body}
<p class="mut">Downloads: <a href="/files/paper/trades.csv">trades.csv</a> · <a href="/files/paper/equity.csv">equity.csv</a> ·
<a href="/files/paper/signals.csv">signals.csv</a> · <a href="/files/report/trending_events.csv">trending_events.csv</a> ·
<a href="/files/report/strategy_results.csv">strategy_results.csv</a></p>
</main></body></html>"""


FILES = {"paper/trades.csv", "paper/equity.csv", "paper/signals.csv", "report/trending_events.csv",
         "report/strategy_results.csv", "report/daily_attention.csv", "report/lead_lag.csv"}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self.send(200, dashboard())
        if p == "/report":
            f = DATA / "report" / "report.html"
            return self.send(200, f.read_bytes()) if f.exists() else self.send(
                404, "No report yet - the study runs at startup and takes 10-30 minutes. <a href='/'>Back</a>")
        if p.startswith("/log/") and p[5:] in JOBS:
            f = LOGS / f"{p[5:]}.log"
            text = f.read_text(errors="replace")[-60_000:] if f.exists() else "No log yet."
            return self.send(200, text, "text/plain; charset=utf-8")
        if p.startswith("/files/") and p[7:] in FILES:
            f = DATA / p[7:]
            if f.exists():
                return self.send(200, f.read_bytes(), "text/csv",
                                 {"Content-Disposition": f'attachment; filename="{f.name}"'})
        return self.send(404, "Not found")

    def do_POST(self):
        p = self.path
        if p.startswith("/run/") and p[5:] in JOBS:
            run_job(p[5:])
        elif p == "/reset" and not state["running"]:
            subprocess.run([sys.executable, str(APP / "wsb_paper_trader.py"), "reset", "--capital", CAPITAL], cwd=APP)
        self.send(303, "", extra={"Location": "/"})

    def log_message(self, *a):
        pass


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            state.update(json.loads(STATE_FILE.read_text()))
        except ValueError:
            pass
    state["running"] = None
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"Dashboard on port {PORT}; time zone {TZ}; paper trader weekdays {PAPER_TIME}; "
          f"study {ANALYZE_DAY} {ANALYZE_TIME}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
