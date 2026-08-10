"""The terminal-log parser that turns an opaque -6 into an actionable cause."""

from __future__ import annotations

from quantbot.connectors.mt5_diag import diagnose

LOG = """MetaTrader 5 x64 build 5836 started for MetaQuotes Ltd.
Windows 11 build 26100, GMT+2
'41139617': authorization on MetaQuotes-Demo failed (Invalid account)
Experts	automated trading is enabled
"""


def _write(tmp_path, text):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "20260809.log").write_text(text, encoding="utf-16")
    return tmp_path


def test_detects_failed_account_authorization(tmp_path):
    diag = diagnose(_write(tmp_path, LOG))
    assert diag.build == 5836
    assert diag.algo_trading is True
    assert diag.last_auth_failure == ("41139617", "MetaQuotes-Demo", "Invalid account")
    joined = " ".join(diag.explain())
    assert "NOT logged in" in joined
    assert "expired" in joined, "an invalid account should suggest making a new demo"


def test_later_success_supersedes_an_earlier_failure(tmp_path):
    text = LOG + "'41139617': authorized on MetaQuotes-Demo\n"
    diag = diagnose(_write(tmp_path, text))
    assert diag.last_auth_failure is None
    assert diag.last_auth_success == ("41139617", "MetaQuotes-Demo")


def test_reports_algo_trading_disabled(tmp_path):
    text = (
        "MetaTrader 5 x64 build 5836 started\n"
        "'41139617': authorized on MetaQuotes-Demo\n"
        "Experts\tautomated trading is disabled\n"
    )
    diag = diagnose(_write(tmp_path, text))
    assert diag.algo_trading is False
    assert any("Algo Trading is OFF" in s for s in diag.explain())


def test_missing_log_directory_is_not_an_error(tmp_path):
    diag = diagnose(tmp_path)
    assert diag.log_file is None
    assert diag.explain(), "must still return guidance"
