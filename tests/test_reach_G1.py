from bfs_verify import bfs_reachable


def test_reach_under_G1_in_1_step():
    b_max, e_max = 6, 6
    start = (1, 2)
    target = (2, 1)

    # Augmented generators includes mu
    generators = ["sigma", "lambda2", "mu"]
    max_depth = 1

    ok, visited = bfs_reachable(start, target, generators, b_max, e_max, max_depth)

    assert ok is True, "Target should be REACHABLE under G1 within depth 1."
    assert target in visited, "Target must appear in visited set when reachable."
