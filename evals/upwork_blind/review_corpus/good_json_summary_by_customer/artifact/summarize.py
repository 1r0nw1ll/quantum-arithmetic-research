def summarize_by_customer(orders):
    summary = {}
    for order in orders:
        customer_id = order.get("customer_id")
        cents = int(order.get("cents", 0))
        if customer_id is None:
            continue
        bucket = summary.setdefault(customer_id, {"total_cents": 0, "order_count": 0})
        bucket["total_cents"] += cents
        bucket["order_count"] += 1
    return summary
