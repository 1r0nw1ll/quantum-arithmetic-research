from summarize import summarize_by_customer


def test_single_customer():
    orders = [{"customer_id": "c1", "order_id": "o1", "cents": 500},
              {"customer_id": "c1", "order_id": "o2", "cents": 300}]
    out = summarize_by_customer(orders)
    assert out == {"c1": {"total_cents": 800, "order_count": 2}}


def test_multi_customer():
    orders = [{"customer_id": "c1", "cents": 100},
              {"customer_id": "c2", "cents": 250},
              {"customer_id": "c1", "cents": 400}]
    out = summarize_by_customer(orders)
    assert out["c1"] == {"total_cents": 500, "order_count": 2}
    assert out["c2"] == {"total_cents": 250, "order_count": 1}


def test_skips_missing_customer_id():
    orders = [{"order_id": "o1", "cents": 100},
              {"customer_id": "c1", "cents": 200}]
    out = summarize_by_customer(orders)
    assert out == {"c1": {"total_cents": 200, "order_count": 1}}
