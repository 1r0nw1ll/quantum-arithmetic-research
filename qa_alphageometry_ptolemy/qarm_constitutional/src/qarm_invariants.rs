// QARM v0.2 Constitutional Invariants
// Mirrors QARM_v02_Failures.tla invariants exactly
// DO NOT modify without updating TLA+ spec first

use crate::qarm_state::*;

/// Inv_TupleClosed - mirrors TLA+ invariant
/// Checks that d = b+e and a = d+e
pub fn inv_tuple_closed(s: &State) -> bool {
    tuple_closed(s.b, s.e, s.d, s.a)
}

/// Inv_InBounds - mirrors TLA+ invariant
/// Checks that all values are within [0, CAP]
pub fn inv_in_bounds(s: &State) -> bool {
    in_bounds(s.b, s.e, s.d, s.a)
}

/// Inv_QDef - mirrors TLA+ invariant
/// Checks that qtag = qdef(a)
pub fn inv_qdef(s: &State) -> bool {
    s.qtag == qdef(s.a)
}

/// Inv_FailDomain - mirrors TLA+ invariant
/// Checks that fail is in valid domain
pub fn inv_fail_domain(s: &State) -> bool {
    matches!(
        s.fail,
        FailType::Ok | FailType::OutOfBounds | FailType::FixedQViolation | FailType::Illegal
    )
}

/// Inv_MoveDomain - mirrors TLA+ invariant
/// Checks that last_move is in valid domain
pub fn inv_move_domain(s: &State) -> bool {
    matches!(s.last_move, Move::None | Move::Sigma | Move::Mu | Move::Lambda)
}

/// Check all invariants at once
pub fn check_all_invariants(s: &State) -> bool {
    inv_tuple_closed(s)
        && inv_in_bounds(s)
        && inv_qdef(s)
        && inv_fail_domain(s)
        && inv_move_domain(s)
}

/// Verify invariants hold for a collection of states
pub fn verify_invariants(states: &[State]) -> Result<(), String> {
    for (i, s) in states.iter().enumerate() {
        if !inv_tuple_closed(s) {
            return Err(format!("State {}: Inv_TupleClosed violated: d={}, b+e={}, a={}, d+e={}",
                i, s.d, s.b + s.e, s.a, s.d + s.e));
        }
        if !inv_in_bounds(s) {
            return Err(format!("State {}: Inv_InBounds violated: b={}, e={}, d={}, a={}, CAP={}",
                i, s.b, s.e, s.d, s.a, CAP));
        }
        if !inv_qdef(s) {
            return Err(format!("State {}: Inv_QDef violated: qtag={}, qdef(a)={}",
                i, s.qtag, qdef(s.a)));
        }
        if !inv_fail_domain(s) {
            return Err(format!("State {}: Inv_FailDomain violated: fail={:?}",
                i, s.fail));
        }
        if !inv_move_domain(s) {
            return Err(format!("State {}: Inv_MoveDomain violated: last_move={:?}",
                i, s.last_move));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariants_on_initial_states() {
        let states = initial_states();
        assert!(verify_invariants(&states).is_ok(),
            "All initial states must satisfy invariants");
    }

    #[test]
    fn test_tuple_closed_invariant() {
        let good = State {
            b: 1,
            e: 2,
            d: 3,
            a: 5,
            qtag: qdef(5),
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(inv_tuple_closed(&good));

        let bad = State {
            b: 1,
            e: 2,
            d: 4, // Wrong: should be 3
            a: 5,
            qtag: qdef(5),
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(!inv_tuple_closed(&bad));
    }

    #[test]
    fn test_qdef_invariant() {
        let good = State {
            b: 0,
            e: 1,
            d: 1,
            a: 2,
            qtag: qdef(2),
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(inv_qdef(&good));

        let bad = State {
            b: 0,
            e: 1,
            d: 1,
            a: 2,
            qtag: 999, // Wrong qtag
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(!inv_qdef(&bad));
    }

    #[test]
    fn test_bounds_invariant() {
        let good = State {
            b: 5,
            e: 5,
            d: 10,
            a: 15,
            qtag: qdef(15),
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(inv_in_bounds(&good));

        let bad = State {
            b: 21, // Exceeds CAP=20
            e: 0,
            d: 21,
            a: 21,
            qtag: qdef(21),
            fail: FailType::Ok,
            last_move: Move::None,
        };
        assert!(!inv_in_bounds(&bad));
    }

    #[test]
    fn test_domain_invariants() {
        let s = State {
            b: 0,
            e: 0,
            d: 0,
            a: 0,
            qtag: 0,
            fail: FailType::Ok,
            last_move: Move::None,
        };

        assert!(inv_fail_domain(&s));
        assert!(inv_move_domain(&s));
    }
}
