# WSB hype vs. the stock market

Two programs:

1. **`wsb_analyze.py`** – downloads the last ~3 months of r/wallstreetbets posts, finds the
   tickers, and tests whether a ticker *trending* on WSB is followed by its stock beating SPY.
   It also backtests 24 simple paper strategies and checks the best one on data it wasn't tuned on.
   Output: `data/report/report.html` plus CSVs.
2. **`wsb_paper_trader.py`** – paper trades the same rules going forward with fake money, so you
   can see whether any edge holds up in real time.

No API keys or accounts needed. Nothing touches a real brokerage.

## Run it on Unraid (recommended)

Every push to `main` makes GitHub Actions build the container and publish it to
`ghcr.io/yslboatboy/wsb-research:latest`. The container runs everything on its own:

- **Paper trader** every weekday at 8:15 AM Central (before the 9:30 ET open)
- **Correlation study** at first start, then every Sunday 6 PM
- **Dashboard** at `http://YOUR-SERVER:8088` – portfolio, equity vs SPY, trades, the report, logs, and "run now" buttons

### One-time GitHub step
After the first build finishes (Actions tab, ~3 min), open your profile → **Packages** → `wsb-research` →
**Package settings** → **Change visibility** → **Public**, so Unraid can pull it without logging in.
(Keeping it private works too: run `docker login ghcr.io` on Unraid with a GitHub token that has `read:packages`.)

### Install on Unraid
1. Open the Unraid terminal (>_ icon top right) and run:
   ```bash
   wget -O /boot/config/plugins/dockerMan/templates-user/my-wsb-research.xml \
     https://raw.githubusercontent.com/YSLBoatBoy/wsb-research/main/unraid/wsb-research.xml
   ```
2. **Docker** tab → **Add Container** → **Template** dropdown → pick **wsb-research** → **Apply**.
3. Click the container icon → **WebUI**. The first study takes 10–30 minutes (it downloads ~3 months of posts);
   the dashboard refreshes itself while it runs.

To update after code changes: push to GitHub, wait for the build, then in the Docker tab click
**Check for Updates** → **Apply update**. Your data in `/mnt/user/appdata/wsb-research` is kept.

Settings (template fields / environment variables): `TZ`, `PAPER_TIME`, `ANALYZE_DAY`, `ANALYZE_TIME`,
`ANALYZE_DAYS`, `PAPER_CAPITAL`. A `docker-compose.yml` is included if you use the Compose Manager plugin.

The paper portfolio keeps the rules it started with. When a new weekly study picks a different
strategy, the dashboard shows it; press **Reset portfolio** to switch.

## Run the study on GitHub instead
**Actions** tab → **Run WSB correlation study** → **Run workflow**. When it finishes, download the
`wsb-report` artifact at the bottom of the run page and open `report/report.html`.

## Run it on a regular computer

### Setup (once)

```bash
pip install -r requirements.txt
```

### 1. Run the correlation study

```bash
python wsb_analyze.py              # ~120 days (3 months + warm-up). First run takes 10–30 min to download posts.
python wsb_analyze.py --days 250   # more history = more trustworthy answer
python wsb_analyze.py --demo       # offline test with fake data
```

Open `data/report/report.html`. The key things to read:

- **Before vs after table** – does the stock go up *after* it trends (useful) or only *before* (hype chasing price)?
  `p_value` under 0.05 means "unlikely to be luck".
- **Hype vs price chart** – grey bars (past) vs blue bars (future).
- **Walk-forward line** – the only backtest number worth trusting. It picks the best strategy on the
  first half of the period and reports how it did on the second half.

Knobs: `--z` (how unusual attention must be, default 2.0), `--min-mentions` (default 5),
`--lookback` (days of "normal", default 20), `--max-pos`, `--cost-bps`.

### 2. Paper trade it

```bash
python wsb_paper_trader.py run        # do today's update
python wsb_paper_trader.py status     # portfolio, open positions, queued orders, P&L vs SPY
python wsb_paper_trader.py replay --days 60   # simulate the last 60 days through the same code
python wsb_paper_trader.py reset --capital 100000
```

It uses the strategy that `wsb_analyze.py` saved in `data/best_config.json` (default if missing:
buy trending tickers, hold 3 days, max 10 positions, 10 bps cost).

**Run it once every weekday morning before the open, ideally 8:00–8:25 AM Central.** Each run:
settles yesterday's fills using real open/close prices, then counts posts since the last open and
queues buys for today's open. Missing a day is fine – it catches up; orders for a missed open are cancelled.

Schedule it:

- **Mac/Linux** (`crontab -e`): `15 8 * * 1-5 cd /path/to/wsb_research && /usr/bin/python3 wsb_paper_trader.py run >> data/paper/log.txt 2>&1`
- **Windows** Task Scheduler: Daily trigger 8:15 AM, weekdays; action `python` with argument
  `wsb_paper_trader.py run`, "Start in" = this folder.

Records are in `data/paper/`: `trades.csv`, `equity.csv`, `signals.csv`, `book.json`.

## How "trending" is defined

Each post is assigned to the first 9:30 AM ET open after it was posted (the earliest you could act).
A ticker's daily attention = sum over its posts of `1 + log(1+upvotes) + 0.5·log(1+comments)`.
It trends when attention is ≥ z standard deviations above its own prior 20-day average and it has
≥ 5 posts that day. Repeat triggers within 5 trading days are ignored.

## Data sources & caveats

- Posts: Arctic Shift archive (falls back to reddit.com/new for only the newest ~1000 posts if the archive is down).
- Tickers: NASDAQ Trader symbol list; common words like AI/NOW/OPEN only count as `$AI` etc.
- Prices: Yahoo Finance via yfinance (adjusted).
- Archived upvote counts are often captured early, so they understate final popularity.
- Fills assume the exact open/close price; real meme-stock fills are worse. Shorts ignore borrow costs.
- 3 months is short; one meme rally can dominate. Treat any positive result as a hypothesis to paper trade, not proof.
