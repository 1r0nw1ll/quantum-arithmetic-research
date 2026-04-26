# fetch_and_retry

Example usage:

```python
from fetch import fetch_and_retry
data = fetch_and_retry("https://api.example.com/v1/items")
```

Retry policy: exponential backoff on 5xx, up to `max_retries=3` by default.
Non-5xx errors propagate. Returns parsed JSON.
