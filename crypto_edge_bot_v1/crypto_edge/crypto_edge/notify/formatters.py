"""Pure message builders. No I/O, no network -- fully unit-testable offline.

Formatting is deliberately separated from delivery so that the content of
every alert is verified by the test suite even though sending is not.
"""
from __future__ import annotations

from ..timeutils import fmt_duration, iso


def _money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _pct(v: float, dp: int = 2) -> str:
    return f"{'+' if v >= 0 else ''}{v:.{dp}f}%"


def bot_start(*, mode: str, exchange: str, equity: float, cash: float,
              open_positions: int, strategy: str, version: str,
              universe_size: int, telegram_ok: bool) -> str:
    return "\n".join([
        "🟢 BOT ONLINE",
        f"Mode: {mode}",
        f"Exchange: {exchange}",
        f"Equity: {_money(equity)}",
        f"Cash: {_money(cash)}",
        f"Open positions: {open_positions}",
        f"Strategy: {strategy} v{version}",
        f"Universe: {universe_size} tradable markets",
        "",
        "⚠️ PAPER TRADING — no real orders can be placed.",
    ])


def entry(*, symbol: str, side: str, entry_price: float, qty: float,
          position_value: float, pct_of_account: float, stop: float,
          dollar_risk: float, pct_risk: float, score: float,
          htf_regime: str, btc_regime: str, breadth: float,
          reasons: list[str], equity: float) -> str:
    lines = [
        f"🔵 ENTRY — {symbol} ({side.upper()})",
        f"Entry: ${entry_price:,.6g}",
        f"Qty: {qty:,.8g}",
        f"Value: {_money(position_value)} ({pct_of_account:.1f}% of account)",
        f"Stop: ${stop:,.6g}",
        f"Risk: {_money(dollar_risk)} ({pct_risk:.2f}% of account)",
        f"Score: {score:.1f}/100",
        f"Regime: HTF {htf_regime} | BTC {btc_regime} | breadth {breadth:.0f}%",
    ]
    if reasons:
        lines.append("Why: " + "; ".join(reasons[:4]))
    lines.append(f"Equity: {_money(equity)}")
    return "\n".join(lines)


def stop_update(*, symbol: str, prev_stop: float, new_stop: float, price: float,
                entry_price: float, qty: float, kind: str) -> str:
    locked = (new_stop - entry_price) * qty
    lines = [
        f"🔧 STOP UPDATE — {symbol} ({kind})",
        f"Previous: ${prev_stop:,.6g}",
        f"New: ${new_stop:,.6g}",
        f"Price: ${price:,.6g}",
    ]
    if locked > 0:
        lines.append(f"Locked-in profit: {_signed(locked)}")
    return "\n".join(lines)


def exit_trade(*, symbol: str, entry_price: float, exit_price: float,
               position_value: float, held_s: float, gross: float, fees: float,
               slippage: float, net: float, return_pct: float,
               account_return_pct: float, exit_reason: str, equity: float,
               total_pnl: float) -> str:
    win = net > 0
    head = "✅ PROFIT" if win else "🔴 LOSS"
    return "\n".join([
        f"{head} — {symbol}",
        f"Entry: ${entry_price:,.6g}",
        f"Exit: ${exit_price:,.6g}",
        f"Size: {_money(position_value)}",
        f"Held: {fmt_duration(held_s)}",
        f"Gross: {_signed(gross)}",
        f"Fees: -{_money(abs(fees))[1:] if fees >= 0 else _money(fees)}",
        f"Slippage: -${abs(slippage):,.2f}",
        f"NET {'PROFIT' if win else 'LOSS'}: {_signed(net)}",
        f"Trade return: {_pct(return_pct)}",
        f"Account impact: {_pct(account_return_pct)}",
        f"Exit reason: {exit_reason}",
        f"Account equity: {_money(equity)}",
        f"Total bot P&L: {_signed(total_pnl)}",
    ])


def circuit_breaker(*, kind: str, reason: str, equity: float,
                    action: str = "New entries stopped") -> str:
    return "\n".join([
        f"🚨 CIRCUIT BREAKER — {kind}",
        reason,
        f"Equity: {_money(equity)}",
        f"Action: {action}",
    ])


def heartbeat(*, uptime_s: float, equity: float, today_pnl: float,
              total_pnl: float, open_positions: int, btc_regime: str,
              breadth: float, signals_evaluated: int, last_data_ms: int,
              halted: bool) -> str:
    lines = [
        "💓 HEARTBEAT" + (" — HALTED" if halted else ""),
        f"Uptime: {fmt_duration(uptime_s)}",
        f"Equity: {_money(equity)}",
        f"Today P&L: {_signed(today_pnl)}",
        f"Total P&L: {_signed(total_pnl)}",
        f"Open positions: {open_positions}",
        f"BTC regime: {btc_regime} | breadth: {breadth:.0f}%",
        f"Signals evaluated: {signals_evaluated}",
        f"Last data update: {iso(last_data_ms)}",
    ]
    return "\n".join(lines)


def error(*, where: str, message: str) -> str:
    return f"⚠️ ERROR — {where}\n{message[:400]}"


def daily_report(*, date: str, uptime_s: float, rep: dict, top_setups: list[dict],
                 btc_regime: str, breadth: float, strategy_status: str,
                 daily_loss_remaining: float, events: list[str]) -> str:
    a, t, r, s = rep["account"], rep["trading"], rep["risk"], rep["sample"]
    day_pnl = a["total_pnl"]
    lines = [
        "📊 CRYPTO EDGE — DAILY REPORT",
        f"Date: {date}",
        f"Bot uptime: {fmt_duration(uptime_s)}",
        "",
        "ACCOUNT",
        f"Starting equity: {_money(a['starting_balance'])}",
        f"Current equity: {_money(a['current_equity'])}",
        f"Total P&L: {_signed(a['total_pnl'])}",
        f"Total return: {_pct(a['account_return_pct'])}",
        "",
        "TRADING",
        f"Closed trades: {t['closed_trades']}",
        f"Wins / Losses: {t['wins']} / {t['losses']}",
        f"Win rate: {t['win_rate_pct']:.1f}%",
        f"Profit factor: {t['profit_factor']:.2f}" if t["profit_factor"] != float("inf")
        else "Profit factor: n/a (no losses yet)",
        f"Expectancy/trade: {_signed(t['expectancy_per_trade'])}",
    ]
    lines.append("")
    lines.append("OPEN POSITIONS")
    if rep["open_positions"]:
        for p in rep["open_positions"]:
            lines.append(
                f"{p['symbol']} — ${p['entry']:,.6g} → ${p['current']:,.6g} — "
                f"{_signed(p['unrealized'])} — stop ${p['stop']:,.6g}")
    else:
        lines.append("(none)")

    lines += [
        "",
        "RISK",
        f"Current exposure: {r['current_exposure_pct']:.1f}%",
        f"Current drawdown: {r['current_drawdown_pct']:.2f}%",
        f"Max drawdown: {r['max_drawdown_pct']:.2f}%",
        f"Daily loss limit remaining: {_money(daily_loss_remaining)}",
        "",
        "MARKET",
        f"BTC regime: {btc_regime}",
        f"Market breadth: {breadth:.0f}%",
        f"Strategy status: {strategy_status}",
        "",
        "COSTS",
        f"Fees: -${abs(a['total_fees']):,.2f}",
        f"Estimated slippage: -${abs(a['estimated_slippage_cost']):,.2f}",
        f"Gross P&L: {_signed(a['gross_pnl'])} → Net P&L: {_signed(a['net_pnl'])}",
    ]

    if top_setups:
        lines.append("")
        lines.append("TOP SETUPS")
        for i, s_ in enumerate(top_setups[:5], 1):
            lines.append(f"{i}. {s_['symbol']} — score {s_['score']:.1f}")

    if events:
        lines.append("")
        lines.append("IMPORTANT EVENTS")
        lines.extend(events[:5])

    lines.append("")
    lines.append(f"SAMPLE: {s['closed_trades']} trades / {s['trading_days']} days")
    if not s["ratios_reliable"]:
        lines.append("⚠️ Sample too small — ratios not yet meaningful.")
    else:
        adv = rep["advanced"]
        if adv.get("sharpe_annualised") is not None:
            lines.append(f"Sharpe: {adv['sharpe_annualised']:.2f}")
        if adv.get("sortino_annualised") is not None:
            lines.append(f"Sortino: {adv['sortino_annualised']:.2f}")
    return "\n".join(lines)
