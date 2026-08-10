# Running 24/7 on a VPS

Short answer: **yes, but it must be a Windows VPS**, and the market is only open
~5 days a week anyway.

## The hard constraint

The `MetaTrader5` Python package is a **Windows-only** bridge that talks to a
*running, logged-in* MT5 terminal over local IPC. There is no Linux build and no
headless mode. So:

| VPS type | Works? |
|---|---|
| Windows VPS (Server 2019/2022, or Windows 10/11) | Yes — the supported path |
| Linux VPS + Wine | Sometimes; fragile, and the Python bridge often fails |
| Linux VPS, no Wine | No. `import MetaTrader5` fails outright |

Anything else in this project — ingestion, features, strategies, risk, journal —
is pure Python and runs anywhere. Only the MT5 connector and broker need Windows.

**Sizing:** 2 vCPU / 4 GB RAM is comfortable for a handful of symbols. The
terminal wants ~500 MB, and a `train` run is the only CPU spike. Put the VPS in a
region near your broker's servers if latency matters for scalping — for a
15-minute swing mode it does not.

## Setup

1. Install MetaTrader 5, log into the demo account, and **leave the terminal
   running**. The bridge attaches to it; if the terminal exits, the bot loses its
   feed.
2. Turn on **Algo Trading** (toolbar button green, or Tools → Options → Expert
   Advisors → "Allow algorithmic trading").
3. Install Python 3.10+ and this project's requirements.
4. Copy `.env` across with your credentials, or recreate it there.
5. Verify before automating anything:
   ```
   python -m quantbot doctor
   ```
   You want `mode=DEMO` with a balance. Anything else, fix it first.

## Keeping it alive

The bot must survive reboots, terminal crashes and Windows updates. Two things
need supervising: **the MT5 terminal** and **the bot**.

### The terminal

Put a shortcut to `terminal64.exe` in `shell:startup` so it launches at logon,
and set the VPS to log in automatically after a reboot. MT5 reconnects to the
broker on its own once running.

### The bot

`quantbot run` already loops on its own schedule, so you only need it restarted
if it dies. Use a scheduled task rather than a bare console window:

```powershell
# Run at logon, restart every 5 min if it stops, no time limit
schtasks /create /tn "QuantBot" /sc onlogon /rl highest ^
  /tr "cmd /c cd /d D:\QuantBot && python -m quantbot run --live >> logs\bot.log 2>&1"
```

Then in Task Scheduler → QuantBot → Settings, tick **"If the task fails, restart
every 5 minutes"** and untick "Stop the task if it runs longer than…".

A supervisor like [NSSM](https://nssm.cc/) is the tidier option if you would
rather run it as a real Windows service.

## What it does when the market is shut

Nothing harmful, by design:

- The pre-trade gate refuses on Saturdays, on Sundays before the Sydney open, and
  after the Friday cutoff (`sessions.py`).
- Outside the configured sessions for a mode it stands down and says so.
- `ingest` still runs, so the calendar and candles stay current.

So leaving it running all week is fine — it simply won't open positions when it
shouldn't.

## Things that will bite you

**The terminal logs out.** Demo accounts expire; brokers force updates. The bot
reports this clearly (`doctor` reads the terminal log), but nothing will trade
until you log in again. Check `python -m quantbot report` periodically.

**MT5 auto-updates and restarts.** After an update, Algo Trading sometimes comes
back **off** — which silently blocks the Python bridge. This has already happened
once during development. Re-check it after any update.

**The VPS clock.** Everything is UTC internally, but session boundaries are
computed from the system clock. Make sure Windows time sync is on.

**Broker server time ≠ UTC.** Handled for candles, but if you ever import CSVs
pass the right `--tz-offset`.

**Disk.** The SQLite database grows with candles and journal rows. It is small
(tens of MB per symbol-year), but `artifacts/` is where everything lives — put it
on a disk you back up, because the journal *is* the training data.

## Monitoring from elsewhere

```bash
python -m quantbot doctor     # connection, data freshness, recent alerts
python -m quantbot report     # journal, per-setup PnL, calibration
python -m quantbot gate       # progress against the promotion bar
```

Run those over RDP or SSH occasionally. A silent bot is usually a *correct* bot
standing down outside sessions — `doctor` tells you which.

## Before you leave it unattended

- `dry_run: false` and `broker.kind: mt5` are what actually place demo orders.
- `broker.allow_live` must stay `false`. It is the only thing between the bot and
  a real account, and the demo track record is the whole point of this phase.
- Set `risk.max_daily_loss_pct` to something you are comfortable with unattended;
  it is the circuit breaker that stops a bad day compounding.
