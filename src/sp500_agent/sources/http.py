from __future__ import annotations

import threading
import time

import requests

DEFAULT_USER_AGENT = "SP500-Research-Agent/0.4 (+https://github.com/axlezhao/SP500-Research-Agent)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpClient:
    """requests.Session with a minimum interval between calls (shared across threads) and retries."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, max_per_second: float = 5.0, retries: int = 3, timeout: float = 60.0, session=None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / max_per_second if max_per_second else 0.0
        self.retries = retries
        self.timeout = timeout
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        for attempt in range(self.retries + 1):
            self._wait_for_slot()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.retries:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES or attempt == self.retries:
                    response.raise_for_status()
                    return response
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("unreachable")

    def get_json(self, url: str, params: dict | None = None):
        return self.get(url, params).json()

    def get_text(self, url: str, params: dict | None = None) -> str:
        return self.get(url, params).text
