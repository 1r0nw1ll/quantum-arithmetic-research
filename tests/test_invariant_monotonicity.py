from bfs_verify import GEN_APPLIERS


def test_e_monotone_under_G0_for_all_inbounds_states():
    # This test enforces the stated obstruction witness:
    # Under G0 = {sigma, lambda2}, e never decreases (for any legal move).
    b_max, e_max = 6, 6
    generators = ["sigma", "lambda2"]

    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            s = (b, e)
            for g in generators:
                s2 = GEN_APPLIERS[g](s, b_max, e_max)
                if s2 is None:
                    continue  # illegal/out_of_bounds move is excluded by contract
                assert s2[1] >= s[1], f"Monotonicity violated: {g} maps {s} -> {s2}"
