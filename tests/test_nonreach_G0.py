from bfs_verify import bfs_reachable


def test_nonreach_under_G0():
    # CapsPair 6x6
    b_max, e_max = 6, 6
    start = (1, 2)
    target = (2, 1)

    # Base generators only
    generators = ["sigma", "lambda2"]
    max_depth = 6

    ok, visited = bfs_reachable(start, target, generators, b_max, e_max, max_depth)

    assert ok is False, "Target should be NONREACHABLE under G0 within depth bound."
    assert start in visited, "Start must be visited."
    assert target not in visited, "Target must not appear in visited set."
