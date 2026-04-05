QA_COMPLIANCE = "test_module — canonical orbit verification, not empirical script"
"""Tests for qa-arithmetic core primitives."""

from qa_arithmetic import (
    qa_step, qa_mod, orbit_family, orbit_period,
    norm_f, v3, qa_tuple, identities, self_test,
    IDENTITY_NAMES, KNOWN_MODULI,
)


def test_qa_mod_basic():
    assert qa_mod(1, 24) == 1
    assert qa_mod(24, 24) == 24
    assert qa_mod(25, 24) == 1
    assert qa_mod(0, 24) == 24  # A1: never 0


def test_qa_step_fibonacci():
    b, e = 1, 1
    b, e = qa_step(b, e, 24)
    assert (b, e) == (1, 2)
    b, e = qa_step(b, e, 24)
    assert (b, e) == (2, 3)
    b, e = qa_step(b, e, 24)
    assert (b, e) == (3, 5)


def test_qa_step_wraparound():
    # 13 + 21 = 34, mod 24 A1 = 10
    b, e = qa_step(21, 13, 24)
    assert b == 13
    assert e == qa_mod(21 + 13, 24)
    assert e == 10


def test_orbit_singularity():
    assert orbit_family(24, 24, 24) == "singularity"
    assert orbit_family(9, 9, 9) == "singularity"
    assert orbit_period(24, 24, 24) == 1
    assert orbit_period(9, 9, 9) == 1


def test_orbit_satellite():
    assert orbit_family(8, 8, 24) == "satellite"
    assert orbit_family(8, 16, 24) == "satellite"
    assert orbit_period(8, 8, 24) == 8


def test_orbit_cosmos():
    assert orbit_family(1, 1, 24) == "cosmos"
    assert orbit_period(1, 1, 24) == 24


def test_orbit_counts():
    counts = {"singularity": 0, "satellite": 0, "cosmos": 0}
    for b in range(1, 25):
        for e in range(1, 25):
            counts[orbit_family(b, e, 24)] += 1
    assert counts["singularity"] == 1
    assert counts["satellite"] == 8
    assert counts["cosmos"] == 567


def test_norm_f():
    # f(1,1) = 1+1-1 = 1
    assert norm_f(1, 1) == 1
    # f(2,1) = 4+2-1 = 5
    assert norm_f(2, 1) == 5


def test_v3():
    assert v3(1) == 0
    assert v3(3) == 1  # noqa: ORBIT-1 — testing v3 arithmetic, not orbit classification
    assert v3(9) == 2
    assert v3(27) == 3
    assert v3(5) == 0
    assert v3(0) == 9999


def test_qa_tuple():
    b, e, d, a = qa_tuple(1, 1, 24)
    assert d == qa_mod(1 + 1, 24)  # 2
    assert a == qa_mod(1 + 2, 24)  # 3


def test_identities_chromogeometry():
    """C*C + F*F == G*G for all Fibonacci directions."""
    fibs = [(2, 1), (3, 2), (5, 3), (8, 5), (13, 8), (21, 13)]
    for b, e in fibs:
        ids = identities(b, e)
        assert ids["C"] * ids["C"] + ids["F"] * ids["F"] == ids["G"] * ids["G"]


def test_identities_relationships():
    ids = identities(2, 1)
    assert ids["A"] - ids["B"] == 2 * ids["C"]
    assert ids["G"] == ids["C"] + ids["B"]
    assert ids["G"] == ids["F"] + 2 * ids["E"]
    assert ids["H"] == ids["C"] + ids["F"]
    assert ids["X"] == ids["C"] // 2


def test_identity_names():
    assert len(IDENTITY_NAMES) == 16
    ids = identities(2, 1)
    for name in IDENTITY_NAMES:
        assert name in ids


def test_self_test():
    assert self_test(verbose=False)


def test_known_moduli():
    assert 9 in KNOWN_MODULI
    assert 24 in KNOWN_MODULI
