"""Compliance layer — enforced by connectors, not bolted on afterwards (§3.4).

Three things every outbound HTTP call goes through:
  1. a per-host minimum interval between requests (rate limiting),
  2. a robots.txt check for the fetched path,
  3. an on-disk response cache, so calendar data isn't re-fetched every minute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)


class ComplianceError(RuntimeError):
    """Raised when a fetch is refused by policy (robots.txt, ToS opt-in, ...)."""


@dataclass
class FetchPolicy:
    user_agent: str
    min_interval_s: float = 2.0
    cache_dir: Path = Path("artifacts/cache")
    cache_ttl_s: float = 1800.0
    respect_robots: bool = True
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()

    # -- robots ------------------------------------------------------------
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base in self._robots:
            return self._robots[base]
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            with httpx.Client(timeout=10.0, headers={"User-Agent": self.user_agent}) as client:
                resp = client.get(f"{base}/robots.txt")
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp = None  # no robots.txt published -> nothing to violate
        except Exception as exc:  # network failure: fail closed on robots only
            log.warning("robots.txt fetch failed for %s (%s); treating as disallowed", base, exc)
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(["User-agent: *", "Disallow: /"])
        self._robots[base] = rp
        return rp

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self._robots_for(url)
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    # -- rate limit --------------------------------------------------------
    def _throttle(self, host: str) -> None:
        with self._lock:
            last = self._last_call.get(host, 0.0)
            wait = self.min_interval_s - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.monotonic()

    # -- cache -------------------------------------------------------------
    def _cache_file(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")

    def cached(self, url: str, ttl_s: float | None = None) -> str | None:
        ttl = self.cache_ttl_s if ttl_s is None else ttl_s
        path = self._cache_file(url)
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - blob.get("fetched_at", 0) > ttl:
            return None
        return blob.get("body")

    def store(self, url: str, body: str) -> None:
        self._cache_file(url).write_text(
            json.dumps({"fetched_at": time.time(), "url": url, "body": body}), encoding="utf-8"
        )

    # -- the one entry point -----------------------------------------------
    def fetch(self, url: str, ttl_s: float | None = None, force: bool = False) -> str:
        if not force:
            hit = self.cached(url, ttl_s)
            if hit is not None:
                log.debug("cache hit %s", url)
                return hit
        if not self.allowed(url):
            raise ComplianceError(
                f"robots.txt disallows {url} for user-agent {self.user_agent!r}. "
                "Use an official API or a licensed feed instead."
            )
        self._throttle(urlparse(url).netloc)
        with httpx.Client(
            timeout=self.timeout_s, headers={"User-Agent": self.user_agent}, follow_redirects=True
        ) as client:
            resp = client.get(url)
        resp.raise_for_status()
        self.store(url, resp.text)
        return resp.text
