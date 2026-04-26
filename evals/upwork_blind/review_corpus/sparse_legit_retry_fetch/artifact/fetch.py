import time
import urllib.request
import json


def fetch_and_retry(url, max_retries=3, backoff_base=0.5):
    """GET url, retry with exponential backoff on 5xx. Return parsed JSON."""
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(url) as resp:
                status = getattr(resp, "status", 200)
                if 500 <= status < 600:
                    raise _RetryableStatus(status)
                body = resp.read()
                return json.loads(body)
        except _RetryableStatus as exc:
            if attempt >= max_retries:
                raise
            time.sleep(backoff_base * (2 ** attempt))
            attempt += 1


class _RetryableStatus(Exception):
    def __init__(self, status):
        super().__init__(f"5xx status {status}")
        self.status = status
