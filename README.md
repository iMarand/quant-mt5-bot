# QuantBot

An automated forex trading system that runs against a **MetaTrader 5 demo
account**, 24 hours a day, across multiple currency pairs and two trading styles.

It watches the economic calendar and the charts together. When a named,
readable setup fires — a pullback in an established trend, a fresh range break, a
rejection at a level, a reaction to an economic release — it sizes the trade
against a fixed risk budget, places it with a stop and target, and then manages
that stop as the trade develops. When nothing fires, or the market is closed, or
a high-impact release is minutes away, it does nothing and tells you why.

A machine-learning model rides along, but it is deliberately **not** in charge:
it can strengthen, weaken or veto a decision the rules already made, and it can
never invent a trade of its own.

This is an implementation of [architecture.md](architecture.md).

> **What this is not.** It is not investment advice, and it makes no claim of
> profitability. As of now it has demonstrated **no edge** on real market data —
> the demo run is the experiment, and `quantbot gate` is the pre-committed bar it
> has to clear before live capital is even a conversation. See
> [Where this actually stands](#where-this-actually-stands).

---

## How a trade gets made

Every 15 minutes (plus extra passes around high-impact news), for each pair and
each mode, the bot walks this pipeline. Any stage can stop it, and each one
records its reason:

```
1. CALENDAR   Do we have calendar data, and is it current?
              A calendar that stopped updating is more dangerous than none —
              the news veto would silently pass and we'd trade blind.

2. CLOCK      Market open? Is the price data fresh? Blackout date?

3. NEWS       High-impact release imminent -> stand down.
              Just released -> flag the news window as active.

4. SESSION    Is this pair's own money centre awake?
              USDJPY wants Tokyo; EURUSD wants London. A big release
              overrides this — the print moves the pair regardless.

5. SETUPS     Only now do the strategies look for a trigger.
              No setup -> no trade, however confident any model is.

6. MODEL      Adjusts confidence up or down; may veto. Never creates,
              never flips direction.

7. RISK       Sizing, stop, target, and ~10 independent veto conditions.
              This layer can refuse anything the strategy layer proposed.

8. EXECUTE    Order placed, journaled with the reason that caused it.

9. MANAGE     Every cycle thereafter: breakeven, partial, trail, time stop.
```

Output looks like this:

```
cycle @ 2026-08-10 20:11:00 UTC
  swing/EURUSD: stand down [session] outside london/newyork/london_ny_overlap
  swing/USDCAD short conf=0.720 [trend_pullback] regime=trending
      why: H1 trend down (adx 29); pullback rsi 42; momentum turning;
           model agreement 0.60 (x1.08)
  scalp/USDJPY long  conf=0.814 [breakout] regime=trending
      why: fresh break of 95% range; atr percentile 0.46; bands expanding
  veto EURUSD: max_positions: 3 open
  plan USDCAD short vol=0.72 entry=1.39376 sl=1.39583 tp=1.39013 rr=1.75
  fill swing/USDCAD short 0.72 @ 1.39376 (#57912397588)
```

A silent bot is usually a *correct* bot standing down. It always names the stage.

---

## Install

```bash
pip install -r requirements.txt
```

Then create `.env` from the template and fill in your demo credentials:

```
QUANTBOT_MT5_LOGIN=12345678
QUANTBOT_MT5_PASSWORD=your-demo-password
QUANTBOT_MT5_SERVER=MetaQuotes-Demo
```

`.env` is git-ignored. Use the **master** password, not the investor one —
investor connects read-only and orders would be rejected.

**MetaTrader 5 is Windows-only.** The `MetaTrader5` package is an IPC bridge to a
running, logged-in terminal; there is no Linux build. Everything else is pure
Python. Without MT5 you can still run the whole pipeline on the paper broker with
generated data (see [Running without a broker](#running-without-a-broker)).

## Quick start

```bash
python -m quantbot doctor                    # connection, data, models, alerts
python -m quantbot ingest                    # calendar + candles for all pairs
python -m quantbot predict                   # latest signal per pair, with its "why"
python -m quantbot train                     # optional: the assistant model
python -m quantbot run --broker mt5 --once   # one cycle, dry run, no orders
python -m quantbot run --broker mt5 --live   # the real loop, demo orders
python -m quantbot report                    # PnL by setup, by session, calibration
python -m quantbot gate                      # progress against the promotion bar
```

For unattended running set it in `config.yaml` instead of on the command line
(`dry_run: false`, `broker.kind: mt5`) so a scheduled task needs no arguments.
See [DEPLOY.md](DEPLOY.md).

---

## Trading modes: scalp and swing

Two styles run side by side. They are separate configurations, not just different
bar sizes — a scalp needs a tighter stop, a smaller budget, an earlier move to
breakeven, and an exit for trades that go nowhere:

| | swing | scalp |
|---|---|---|
| base timeframe | H1 | M5 |
| horizon | 12 bars (~half a day) | 6 bars (30 min) |
| stop / target | 2.0 / 3.5 × ATR | 1.2 / 1.8 × ATR |
| risk per trade | 0.5% | 0.25% |
| confidence bar | 0.60 | **0.66** — a tighter stop is less forgiving |
| breakeven at | +1.0R | +0.6R — get risk off fast |
| time stop | — | 24 bars |
| setups | trend_pullback, breakout, news_reaction, news_breakout | sr_rejection, mean_reversion, breakout, news_breakout |

Each mode manages only the positions it opened, so the scalp manager can never
trail a swing trade onto a 5-minute stop.

## Sessions and 24-hour coverage

Sessions only matter if you hold pairs that trade in them, so the default set of
five majors spans the whole cycle. With `sessions: []` each mode derives its
windows from each pair's own currencies:

```
UTC    tradeable pairs
00-05  USDJPY, AUDUSD          Tokyo / Sydney
06     USDJPY
07-08  EURUSD, GBPUSD, USDJPY  London opens
09-11  EURUSD, GBPUSD
12-20  all five                London/NY overlap, then New York
21     (none)                  daily rollover — spreads widen, by design
22-23  AUDUSD                  Sydney reopens
```

Session windows shift by an hour for northern-hemisphere daylight saving. The
weekend, the Friday-evening cutoff and the rollover hour are all refused
automatically.

`report` breaks realized PnL down by **session × mode**, which is how you find out
whether the thin Asian hours pay for their wider spreads. If they don't, restrict
`modes.*.sessions`.

---

## Strategy-first decisioning

**A named setup must fire before any trade is considered.** Order of operations
in [strategy/book.py](quantbot/strategy/book.py):

```
1. strategies vote      -> no trigger means no trade, full stop
2. conflicts resolved   -> opposing setups cancel rather than net out
3. confluence scored    -> agreeing setups raise conviction
4. model adjusts        -> may damp, boost or veto; may never create or flip
```

Step 4 comes last precisely so the model cannot manufacture a signal on a bar
where nothing fired. `strategy.model_role: off` runs pure rules with no ML.

### The setups

| Setup | Fires when |
|---|---|
| `trend_pullback` | HTF trend + ADX, base-TF pullback, momentum turning back |
| `breakout` | *Fresh* break of the rolling range with volatility expansion |
| `mean_reversion` | Band extreme **in a range only**, with a rejection candle |
| `sr_rejection` | Rejection wick at a swing high/low |
| `news_reaction` | After a release, trade the surprise once price confirms it |
| `news_breakout` | After a release, trade the break of the pre-news range |

Each returns a `quality` in 0..1 from how strongly the bar matched, mapping onto
confidence between `base_confidence` and `max_confidence`. A setup refuses to
fire when the features it needs are missing, so a disabled timeframe silently
disables its dependants rather than guessing.

Both news setups refuse to trade *before* a release: holding into a print is a
volatility bet, not a directional edge.

### Why this matters

Model-primary decisioning emitted a direction on every bar — it traded ~95% of
bars and paid spread on noise. Strategy-first triggers on ~17%, and every trade
carries the name of the rule that caused it, which is what lets `report` tell you
*which pattern* is losing money.

---

## Managing an open trade

Stops only ever move toward profit. A stop that can loosen is not a stop.

```
0. time stop     scalp that hasn't worked within its horizon -> close
1. breakeven+    at +Rbe, stop moves to entry + 0.1R
                 (entry exactly still loses the spread; the buffer makes the
                  worst case a small win rather than a scratch)
2. partial       at +Rp, scale out a fraction
3. trail         beyond +Rt, trail N × ATR behind price
```

Because trailing needs a trade to be *in profit*, a losing position keeps its
original stop. That is the design, not a fault.

### Sizing guards

Risk-based sizing alone is not enough, because size scales inversely with stop
distance — a very tight stop turns a small risk budget into an enormous position:

- `min_stop_spread_mult` — the stop must clear the spread by 4×
- `min_stop_points` — absolute floor (8 pips by default)
- `max_position_leverage` — hard cap on notional as a multiple of equity
- `max_risk_overshoot` — refuse when the minimum lot would risk more than budget
- `max_sl_distance_pct` — refuse an implausibly wide stop (broken ATR / bad data)

These exist because an early version placed a **3-pip stop with 8.29 lots at
11× leverage**. The stop was narrower than the spread; noise alone would take it
out, and the size amplified every bit of slippage.

## Resilience

Built for unattended running:

- **Auto-reconnect** — if the terminal drops, the cycle reconnects (3 attempts,
  backoff) before doing anything else.
- **Position adoption** — every cycle re-attaches to positions this process did
  not open, recovering risk distance from the existing stop and routing each back
  to the mode that opened it via the order comment. Without this, any trade that
  survived a restart belonged to no manager and was **silently never trailed
  again**.
- **Stop repair** — a position found without a stop loss gets one immediately. An
  unprotected position is the worst state this system can be in.
- **The loop survives a failed cycle** — a bot that dies at 3am with positions
  open is worse than one that skips a pass.

---

## Layout

| Path | Role |
|---|---|
| [quantbot/contracts.py](quantbot/contracts.py) | typed contracts every layer exchanges (§1.2) |
| [quantbot/connectors/](quantbot/connectors/) | calendar, market data, CSV import, compliance policy (§3) |
| [quantbot/features/](quantbot/features/) | indicators, event features, multi-timeframe aligner (§4) |
| [quantbot/strategy/](quantbot/strategy/) | **the setups and the book that runs them** |
| [quantbot/engine/](quantbot/engine/) | labeling, regime, rule + GBM ensemble, predictor (§5) |
| [quantbot/decision/](quantbot/decision/) | sessions, pre-trade gate, modes, risk, execution, trade manager (§6) |
| [quantbot/learning/](quantbot/learning/) | journal resolution, retraining, structure search (§7) |
| [quantbot/ops/](quantbot/ops/) | runner, scheduler, monitoring, backtest, promotion gate (§8) |
| [quantbot/storage/db.py](quantbot/storage/db.py) | time-series store, feature store, journal |

## Design decisions worth knowing

**Three independent safety interlocks.** `dry_run: true` blocks all order
sending; `broker.allow_live: false` blocks any REAL MT5 account; the risk layer
can veto anything the strategy layer proposed. Independent on purpose — one
failing does not open the gate.

**The backtest is walk-forward by default.** Replaying history with the
registry's active model is in-sample: that model was trained on those very bars,
and the equity curve is fiction. An early in-sample run reported profit factor
3.6 and Sharpe 31 from a model whose out-of-sample directional accuracy was
46.7%. `backtest` now refits every `--retrain-every` bars on data ending
`horizon + embargo` bars earlier; `--in-sample` exists for diagnostics and says
so loudly.

**Every prediction is journaled, including vetoed ones.** That makes "were my
vetoes right?" answerable later — the cheapest form of the counterfactual
reasoning in §7.4.

**Retraining promotes conservatively.** A new model replaces the active one only
if it beats the incumbent on the same walk-forward score; otherwise the run is
logged as `no_promotion`.

**Labels are triple-barrier, not "price up in N bars".** A move that only counts
after surviving a stop is the question the risk layer actually asks.

## Substitutions from the architecture's suggested stack

Made to keep the system runnable on one machine with no infrastructure. Each is
isolated behind an interface, so swapping back is local:

| Architecture suggests | Used here | Swap point |
|---|---|---|
| TimescaleDB / InfluxDB / PostgreSQL / Redis | SQLite (WAL) | `quantbot/storage/db.py` — the only module with SQL |
| `ta-lib` / `pandas-ta` | pure pandas/numpy indicators | `quantbot/features/indicators.py` |
| MLflow / W&B | `model_registry` table + metrics JSON | `Database.register_model` |
| Optuna | dependency-free evolutionary search | `quantbot/learning/search.py` |
| Airflow / Prefect | single-process scheduler loop | `quantbot/ops/scheduler.py` |
| Grafana / Prometheus | `report` / `gate` CLI + `alerts` table | `quantbot/ops/monitor.py` |

---

## Working with markets closed

Weekends block nothing that matters. Downloading history, training and
backtesting all work offline; only live fills need an open market.

If the MT5 bridge is unavailable, export bars from the terminal — right-click a
chart → **Save As**, or **Tools → History Center** — then:

```bash
python -m quantbot import-csv "C:\path\EURUSD_M15.csv" --symbol EURUSD --tz-offset 3
```

`--tz-offset` is your **broker server's** UTC offset. MT5 stamps exports in
server time; getting this wrong silently shifts every news-timing feature.

### Running without a broker

`config.sandbox.yaml` points at a separate database so synthetic bars can never
contaminate real ones:

```bash
python tools/seed_synthetic.py --config config.sandbox.yaml
python -m quantbot --config config.sandbox.yaml train
python -m quantbot --config config.sandbox.yaml backtest
```

Seeding **wipes existing candles by default**. Each run anchors an independent
random walk to "now", so appending interleaves two unrelated series into a
sawtooth that looks wildly predictable to a model and produces a nonsense ATR.

Expect the synthetic backtest to *lose* roughly the spread (profit factor ~0.8).
That is the correct result on noise, and it is the baseline proving the pipeline
isn't leaking. **A synthetic run that looks good means something is broken.**

## Connecting MetaTrader 5

The terminal must be running **and logged into an account**, with **Algo Trading
enabled** (toolbar button green, or Tools → Options → Expert Advisors).

**If `initialize` returns `(-6, 'Terminal: Authorization failed')`** the error is
not diagnostic on its own — it fires for several unrelated causes, including when
an account *appears* logged in. Run `python -m quantbot doctor`: it reads the
terminal's own log and reports the actual reason. The two that bite in practice
are an **expired demo account** (logged as `Invalid account`) and **MT5 turning
Algo Trading back off after an auto-update**.

## Calendar history is the one real data gap

Forex Factory publishes **only the current week** (`lastweek`/`nextweek` return
404). Live trading is unaffected — the journal accumulates events as the bot runs
— but the news setups have almost nothing to fire on when backtesting past
months. `backtest` reports calendar coverage and warns when it is thin, so
"0 news trades" is never misread as "news setups don't work". To backfill:

```bash
python -m quantbot import-calendar "history/calendar_2025.csv"
```

## Compliance

Calendar data comes from Forex Factory's **published weekly JSON feed**, not by
scraping HTML. Every outbound request passes through `FetchPolicy`: robots.txt
check, per-host rate limit, on-disk cache.

The Investing.com connector is **inert by default** — their ToS prohibits
automated scraping. It activates only with credentials for a feed you are
licensed to use. The genuinely useful part, cross-source disagreement detection,
lives in `cross_check()` and works with any second source.

---

## Where this actually stands

Being straight about it, because a trading bot that flatters itself is worse than
no bot:

- The **infrastructure** is built and tested: 125 tests, covering lookahead,
  label leakage, risk vetoes, stop management and session logic.
- The **model** scored 0.4621 directional accuracy out-of-sample on real EURUSD —
  *below chance*. Its most informative feature was time-of-day, and the rest were
  volatility measures. It learned *when* things move, not *which way*.
- The last **backtest** (an older, model-primary configuration) returned profit
  factor 0.83. Losing roughly the spread.
- The current **strategy-first, session-gated, multi-mode** system has **never
  been backtested** — sessions, modes and the pre-trade gate all came after the
  last run. The demo period *is* its first test.

So: the plumbing is correct and the risk controls are real, but no edge has been
demonstrated. That is the honest state, and the system is built to keep reporting
it honestly rather than to look good.

## Known limitations

- `PaperBroker` holds state in memory; positions do not survive a restart. (MT5
  positions do, and are re-adopted automatically.)
- The backtester models fixed spread and per-lot commission — no slippage model,
  and intrabar path is unknown, so a bar spanning both barriers is scored as a
  stop-out.
- `search --apply` applies evolved *strategy* genes in memory only; copy the
  printed values into `config.yaml` to persist them.
- Session DST boundaries use a simple date rule, so a few days a year sit an hour
  off. That is a rounding error against a session edge, not worth a timezone
  database.
- Phase 6 (counterfactual self-supervised learning, §7.4) is not implemented —
  correctly so. It has nothing to learn from until the journal has real history.

## Tests

```bash
python -m pytest tests -q
```

The ones worth reading first are the lookahead tests
(`test_indicators_use_no_future_data`,
`test_mtf_alignment_never_uses_an_unclosed_higher_tf_bar`) and
`test_model_cannot_create_a_trade_when_nothing_triggered`, which pins the
central design decision.

## Before any live-capital discussion

Run demo for a long time, then `quantbot gate`. It checks sample size,
directional accuracy, profit factor, Sharpe, max drawdown, confidence calibration
and stability across regimes. Set the bar in `config.yaml` **before** starting and
do not edit it afterwards to fit the results — that defeats the entire point. The
gate reports; it never flips `allow_live` for you.
