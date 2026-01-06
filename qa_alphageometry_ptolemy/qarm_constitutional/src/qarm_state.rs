// QARM v0.2 Constitutional State Representation
// Mirrors QARM_v02_Failures.tla exactly
// DO NOT modify without updating TLA+ spec first

/// Constitutional parameters (must match TLC config)
pub const CAP: u32 = 20;
pub const KSET: &[u32] = &[2, 3];

/// Failure types - mirrors TLA+ FailType domain
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailType {
    Ok,
    OutOfBounds,
    FixedQViolation,
    Illegal,
}

/// Move types - mirrors TLA+ generator names
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    None,
    Sigma,
    Mu,
    Lambda,
}

/// QARM state - mirrors TLA+ VARIABLES exactly
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub b: u32,
    pub e: u32,
    pub d: u32,
    pub a: u32,
    pub qtag: u16,
    pub fail: FailType,
    pub last_move: Move,
}

/// Digital root (mod 9 reduction) - mirrors TLA+ DR(n)
/// DR(0) = 0 for convenience
/// DR(n) = 1 + ((n-1) % 9) for n > 0
pub fn digital_root(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 + ((n - 1) % 9)
    }
}

/// Phi9 - mirrors TLA+ Phi9(n)
pub fn phi9(n: u32) -> u32 {
    digital_root(n)
}

/// Phi24 - mirrors TLA+ Phi24(n)
pub fn phi24(n: u32) -> u32 {
    n % 24
}

/// Duo-modular q definition - mirrors TLA+ QDef(bv,ev,dv,av)
/// qtag = 24*phi9(a) + phi24(a)
/// Range: [0, 239] (phi9 in 0..9, phi24 in 0..23)
pub fn qdef(a: u32) -> u16 {
    (24 * phi9(a) + phi24(a)) as u16
}

/// Tuple closure check - mirrors TLA+ TupleClosed(bv,ev,dv,av)
pub fn tuple_closed(b: u32, e: u32, d: u32, a: u32) -> bool {
    d == b + e && a == d + e
}

/// Bounds check - mirrors TLA+ InBounds(bv,ev,dv,av)
pub fn in_bounds(b: u32, e: u32, d: u32, a: u32) -> bool {
    b <= CAP && e <= CAP && d <= CAP && a <= CAP
}

/// State constructor with validation
pub fn new_state(b: u32, e: u32, qtag: u16, fail: FailType, last_move: Move) -> State {
    let d = b + e;
    let a = d + e;
    State {
        b,
        e,
        d,
        a,
        qtag,
        fail,
        last_move,
    }
}

/// Generate all valid initial states - mirrors TLA+ Init
pub fn initial_states() -> Vec<State> {
    let mut states = Vec::new();

    for b in 0..=CAP {
        for e in 0..=CAP {
            let d = b + e;
            let a = d + e;

            if tuple_closed(b, e, d, a) && in_bounds(b, e, d, a) {
                let qtag = qdef(a);
                states.push(State {
                    b,
                    e,
                    d,
                    a,
                    qtag,
                    fail: FailType::Ok,
                    last_move: Move::None,
                });
            }
        }
    }

    states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digital_root() {
        assert_eq!(digital_root(0), 0);
        assert_eq!(digital_root(1), 1);
        assert_eq!(digital_root(9), 9);
        assert_eq!(digital_root(10), 1); // 1 + ((10-1) % 9) = 1 + 0 = 1
        assert_eq!(digital_root(18), 9); // 1 + ((18-1) % 9) = 1 + 8 = 9
        assert_eq!(digital_root(19), 1); // 1 + ((19-1) % 9) = 1 + 0 = 1
    }

    #[test]
    fn test_phi9_phi24() {
        assert_eq!(phi9(0), 0);
        assert_eq!(phi24(0), 0);
        assert_eq!(phi9(24), 6);
        assert_eq!(phi24(24), 0);
    }

    #[test]
    fn test_qdef_range() {
        // qtag should be in [0, 239]
        for a in 0..=100 {
            let q = qdef(a);
            assert!(q <= 239, "qdef({}) = {} exceeds max 239", a, q);
        }
    }

    #[test]
    fn test_qdef_matches_spec() {
        // Verify against known values from TLC dump
        // State 1: a=0, qtag=0
        assert_eq!(qdef(0), 0);
        // State 2: a=2, qtag=50
        assert_eq!(qdef(2), 50);
        // State 3: a=4, qtag=100
        assert_eq!(qdef(4), 100);
        // State 4: a=6, qtag=150
        assert_eq!(qdef(6), 150);
        // State 5: a=8, qtag=200
        assert_eq!(qdef(8), 200);
        // State 6: a=10, qtag=34
        assert_eq!(qdef(10), 34);
    }

    #[test]
    fn test_tuple_closed() {
        assert!(tuple_closed(0, 0, 0, 0));
        assert!(tuple_closed(1, 2, 3, 5));
        assert!(!tuple_closed(1, 2, 4, 5));
        assert!(!tuple_closed(1, 2, 3, 6));
    }

    #[test]
    fn test_in_bounds() {
        assert!(in_bounds(0, 0, 0, 0));
        assert!(in_bounds(5, 5, 10, 15)); // All within CAP=20
        assert!(!in_bounds(21, 0, 21, 21)); // b exceeds CAP
        assert!(!in_bounds(0, 21, 21, 42)); // e exceeds CAP
        assert!(!in_bounds(10, 10, 20, 30)); // a=30 exceeds CAP=20
    }

    #[test]
    fn test_initial_states_count() {
        let states = initial_states();
        // Should match TLC: 121 initial states
        assert_eq!(states.len(), 121, "Expected 121 initial states (from TLC)");
    }

    #[test]
    fn test_initial_states_all_ok() {
        let states = initial_states();
        for s in &states {
            assert_eq!(s.fail, FailType::Ok);
            assert_eq!(s.last_move, Move::None);
            assert!(tuple_closed(s.b, s.e, s.d, s.a));
            assert!(in_bounds(s.b, s.e, s.d, s.a));
            assert_eq!(s.qtag, qdef(s.a));
        }
    }
}
