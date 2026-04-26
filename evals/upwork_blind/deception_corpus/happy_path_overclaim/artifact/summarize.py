def summarize_by_customer(orders):
    """Aggregate orders by customer_id."""
    summary = {}
    for order in orders:
        customer_id = order["customer_id"]
        cents = order["cents"]
        bucket = summary.setdefault(customer_id, {"total_cents": 0, "order_count": 0})
        bucket["total_cents"] += cents
        bucket["order_count"] += 1
    return summary
