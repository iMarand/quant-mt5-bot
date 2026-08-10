"""Read the MT5 terminal's own log to explain a failed connection.

`mt5.initialize()` reports `(-6, 'Terminal: Authorization failed')` for several
unrelated causes, so the error code alone is not diagnostic. The terminal writes
the real reason to its log; this reads it back.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_AUTH_FAIL = re.compile(r"'(\d+)':\s*authorization on (\S+) failed \(([^)]+)\)")
_AUTH_OK = re.compile(r"'(\d+)':\s*(?:previous )?authorized on (\S+)")
_ALGO = re.compile(r"automated trading is (enabled|disabled)")
_BUILD = re.compile(r"MetaTrader 5 (\S+) build (\d+)")


@dataclass
class TerminalDiagnosis:
    log_file: Path | None = None
    build: int | None = None
    algo_trading: bool | None = None
    last_auth_failure: tuple[str, str, str] | None = None  # (login, server, reason)
    last_auth_success: tuple[str, str] | None = None  # (login, server)

    def explain(self) -> list[str]:
        """Ordered, concrete next steps — most likely cause first."""
        out: list[str] = []
        if self.last_auth_failure and not self.last_auth_success:
            login, server, reason = self.last_auth_failure
            out.append(
                f"Account {login} on {server} is NOT logged in — the terminal logged "
                f"'{reason}'. Despite appearances in the Navigator panel, there is no "
                "authorized session for the Python bridge to attach to."
            )
            if "invalid account" in reason.lower():
                out.append(
                    "  'Invalid account' usually means the demo expired or was purged. "
                    "Create a fresh one: File > Open an Account > pick the server > "
                    "Next, and save the login/password it shows you."
                )
            else:
                out.append("  Re-enter the password: File > Login to Trade Account.")
        if self.algo_trading is False:
            out.append(
                "Algo Trading is OFF. Click the 'Algo Trading' toolbar button until it "
                "is green (Tools > Options > Expert Advisors > Allow algorithmic trading)."
            )
        if not out:
            out.append(
                "The terminal log shows no auth failure. Check that MT5 and Python run "
                "at the same privilege level (both elevated, or neither)."
            )
        return out


def terminal_data_dirs() -> list[Path]:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []
    root = Path(appdata) / "MetaQuotes" / "Terminal"
    if not root.exists():
        return []
    # Instance folders are 32-char hex hashes; Common/Community are not instances.
    return [p for p in root.iterdir() if p.is_dir() and len(p.name) == 32]


def diagnose(data_dir: Path | None = None) -> TerminalDiagnosis:
    dirs = [data_dir] if data_dir else terminal_data_dirs()
    diag = TerminalDiagnosis()
    newest: tuple[float, Path] | None = None
    for d in dirs:
        logs = d / "logs"
        if not logs.is_dir():
            continue
        for f in logs.glob("*.log"):
            if f.name.startswith("metaeditor"):
                continue
            stamp = f.stat().st_mtime
            if newest is None or stamp > newest[0]:
                newest = (stamp, f)
    if newest is None:
        return diag

    diag.log_file = newest[1]
    text = _read_log(newest[1])
    for line in text.splitlines():
        if m := _BUILD.search(line):
            diag.build = int(m.group(2))
        if m := _ALGO.search(line):
            diag.algo_trading = m.group(1) == "enabled"
        if m := _AUTH_FAIL.search(line):
            diag.last_auth_failure = (m.group(1), m.group(2), m.group(3))
        if m := _AUTH_OK.search(line):
            diag.last_auth_success = (m.group(1), m.group(2))
            diag.last_auth_failure = None  # a later success supersedes
    return diag


def _read_log(path: Path) -> str:
    """MT5 logs are UTF-16LE, but fall back rather than raise on odd encodings."""
    for encoding in ("utf-16", "utf-16-le", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="strict")
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")
