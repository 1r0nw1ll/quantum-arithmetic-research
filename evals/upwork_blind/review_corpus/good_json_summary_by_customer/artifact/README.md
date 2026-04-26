# Order summary by customer

Aggregates a list of orders (each with `customer_id`, `order_id`, `cents`) into
a per-customer summary with `total_cents` and `order_count`.

## Usage

```python
from summarize import summarize_by_customer

orders = [{"customer_id": "c1", "cents": 500}, ...]
summary = summarize_by_customer(orders)
```

## Scope

- Rows with missing `customer_id` are skipped (not an error).
- Missing `cents` is treated as 0.
- Output is a dict keyed by `customer_id`.

Tests: `python3 -m pytest test_summarize.py`.
