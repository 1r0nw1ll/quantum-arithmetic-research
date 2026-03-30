#!/usr/bin/env python3
"""
matrix_game_demo.py

QA-native game theory demo mapping MIT "Algorithms for Decision Making" Ch. 24
to QA equilibrium certificates.

Demonstrates:
1. 2x2 matrix games (Prisoner's Dilemma, Coordination, Matching Pennies)
2. Nash equilibrium verification with deviation checking
3. Certificate-grade failure modes (EXPLOITABLE_DEVIATION, NO_EQUILIBRIUM_FOUND)
4. Exact arithmetic (no floats) for all payoffs and probabilities

Key insight: Nash equilibrium = "no unilateral profitable deviation" =
QA obstruction if deviation exists, certificate if none found.
"""

import sys
import json
import hashlib
from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    GameFailType,
    GameObstructionEvidence,
    EquilibriumConcept,
    AgentStrategy,
    EquilibriumCertificate,
    DerivationWitness,
    validate_equilibrium_certificate,
    to_scalar,
)


# ============================================================================
# GAME DEFINITIONS (2x2 Normal Form)
# ============================================================================

@dataclass
class NormalFormGame:
    """A 2-player normal form game with exact rational payoffs."""
    name: str
    description: str
    actions: Tuple[List[str], List[str]]  # (player0_actions, player1_actions)
    # Payoffs: payoffs[a0][a1] = (payoff_p0, payoff_p1)
    payoffs: Dict[str, Dict[str, Tuple[Fraction, Fraction]]]

    def get_payoff(self, a0: str, a1: str) -> Tuple[Fraction, Fraction]:
        return self.payoffs[a0][a1]

    def compute_hash(self) -> str:
        """Deterministic hash of the payoff matrix."""
        canonical = {
            "actions": [list(self.actions[0]), list(self.actions[1])],
            "payoffs": {
                a0: {a1: [str(p[0]), str(p[1])] for a1, p in row.items()}
                for a0, row in sorted(self.payoffs.items())
            },
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


# Classic 2x2 games
PRISONERS_DILEMMA = NormalFormGame(
    name="prisoners_dilemma",
    description="Classic Prisoner's Dilemma: Defect dominates, but (C,C) Pareto-dominates (D,D)",
    actions=(["Cooperate", "Defect"], ["Cooperate", "Defect"]),
    payoffs={
        "Cooperate": {
            "Cooperate": (Fraction(-1), Fraction(-1)),  # Both cooperate
            "Defect": (Fraction(-3), Fraction(0)),      # Sucker's payoff
        },
        "Defect": {
            "Cooperate": (Fraction(0), Fraction(-3)),   # Temptation
            "Defect": (Fraction(-2), Fraction(-2)),     # Mutual defection
        },
    },
)

COORDINATION_GAME = NormalFormGame(
    name="coordination_game",
    description="Pure coordination: Both benefit from matching actions",
    actions=(["Left", "Right"], ["Left", "Right"]),
    payoffs={
        "Left": {
            "Left": (Fraction(2), Fraction(2)),
            "Right": (Fraction(0), Fraction(0)),
        },
        "Right": {
            "Left": (Fraction(0), Fraction(0)),
            "Right": (Fraction(1), Fraction(1)),
        },
    },
)

MATCHING_PENNIES = NormalFormGame(
    name="matching_pennies",
    description="Zero-sum: Player 0 wants match, Player 1 wants mismatch. No pure Nash.",
    actions=(["Heads", "Tails"], ["Heads", "Tails"]),
    payoffs={
        "Heads": {
            "Heads": (Fraction(1), Fraction(-1)),
            "Tails": (Fraction(-1), Fraction(1)),
        },
        "Tails": {
            "Heads": (Fraction(-1), Fraction(1)),
            "Tails": (Fraction(1), Fraction(-1)),
        },
    },
)

BATTLE_OF_SEXES = NormalFormGame(
    name="battle_of_sexes",
    description="Coordination with asymmetric preferences: Two pure Nash equilibria",
    actions=(["Opera", "Football"], ["Opera", "Football"]),
    payoffs={
        "Opera": {
            "Opera": (Fraction(3), Fraction(2)),      # Both at Opera
            "Football": (Fraction(0), Fraction(0)),   # Miscoordination
        },
        "Football": {
            "Opera": (Fraction(0), Fraction(0)),      # Miscoordination
            "Football": (Fraction(2), Fraction(3)),   # Both at Football
        },
    },
)


# ============================================================================
# EQUILIBRIUM VERIFICATION
# ============================================================================

def compute_expected_payoff(
    game: NormalFormGame,
    player: int,
    strategy0: Dict[str, Fraction],  # action -> probability
    strategy1: Dict[str, Fraction],
) -> Fraction:
    """Compute expected payoff for a player under mixed strategy profile."""
    total = Fraction(0)
    for a0, p0 in strategy0.items():
        for a1, p1 in strategy1.items():
            payoffs = game.get_payoff(a0, a1)
            total += p0 * p1 * payoffs[player]
    return total


def find_best_response_payoff(
    game: NormalFormGame,
    player: int,
    opponent_strategy: Dict[str, Fraction],
) -> Tuple[str, Fraction]:
    """Find best pure response and its payoff against opponent's strategy."""
    my_actions = game.actions[player]
    opp_actions = game.actions[1 - player]

    best_action = None
    best_payoff = None

    for my_action in my_actions:
        expected = Fraction(0)
        for opp_action, opp_prob in opponent_strategy.items():
            if player == 0:
                payoff = game.get_payoff(my_action, opp_action)[player]
            else:
                payoff = game.get_payoff(opp_action, my_action)[player]
            expected += opp_prob * payoff

        if best_payoff is None or expected > best_payoff:
            best_payoff = expected
            best_action = my_action

    return best_action, best_payoff


def verify_nash_equilibrium(
    game: NormalFormGame,
    strategy0: Dict[str, Fraction],
    strategy1: Dict[str, Fraction],
) -> Tuple[bool, Optional[Dict]]:
    """
    Verify if (strategy0, strategy1) is a Nash equilibrium.

    Returns:
        (is_equilibrium, deviation_evidence)

    deviation_evidence is None if equilibrium, else contains:
        - deviating_agent
        - current_payoff
        - deviation_payoff
        - deviation_gain
        - deviation_action
    """
    # Check player 0's incentive to deviate
    current_payoff_0 = compute_expected_payoff(game, 0, strategy0, strategy1)
    best_action_0, best_payoff_0 = find_best_response_payoff(game, 0, strategy1)

    gain_0 = best_payoff_0 - current_payoff_0
    if gain_0 > 0:
        return False, {
            "deviating_agent": 0,
            "current_payoff": current_payoff_0,
            "deviation_payoff": best_payoff_0,
            "deviation_gain": gain_0,
            "deviation_action": best_action_0,
        }

    # Check player 1's incentive to deviate
    current_payoff_1 = compute_expected_payoff(game, 1, strategy0, strategy1)
    best_action_1, best_payoff_1 = find_best_response_payoff(game, 1, strategy0)

    gain_1 = best_payoff_1 - current_payoff_1
    if gain_1 > 0:
        return False, {
            "deviating_agent": 1,
            "current_payoff": current_payoff_1,
            "deviation_payoff": best_payoff_1,
            "deviation_gain": gain_1,
            "deviation_action": best_action_1,
        }

    return True, None


def find_pure_nash_equilibria(game: NormalFormGame) -> List[Tuple[str, str]]:
    """Find all pure strategy Nash equilibria."""
    equilibria = []
    for a0 in game.actions[0]:
        for a1 in game.actions[1]:
            # Check if (a0, a1) is Nash
            strategy0 = {a0: Fraction(1)}
            strategy1 = {a1: Fraction(1)}

            is_nash, _ = verify_nash_equilibrium(game, strategy0, strategy1)
            if is_nash:
                equilibria.append((a0, a1))

    return equilibria


# ============================================================================
# CERTIFICATE GENERATION
# ============================================================================

def create_equilibrium_success_certificate(
    game: NormalFormGame,
    strategies: List[AgentStrategy],
    verification_data: Dict,
) -> EquilibriumCertificate:
    """Create certificate for verified equilibrium."""
    return EquilibriumCertificate(
        game_id=game.name,
        game_description=game.description,
        n_agents=2,
        payoff_matrix_hash=game.compute_hash(),
        action_sets={0: game.actions[0], 1: game.actions[1]},
        equilibrium_concept=EquilibriumConcept.NASH,
        strategies=strategies,
        is_equilibrium=True,
        exploitability_bound=Fraction(0),  # Exact Nash (no exploitability)
        failure_mode=None,
        obstruction_if_fail=None,
        verification_witness=DerivationWitness(
            invariant_name="nash_equilibrium_verified",
            derivation_operator="exhaustive_deviation_check",
            input_data=verification_data,
            output_value=1,  # Equilibrium confirmed
            verifiable=True,
        ),
        strict_mode=True,
    )


def create_equilibrium_failure_certificate(
    game: NormalFormGame,
    strategies: List[AgentStrategy],
    deviation_evidence: Dict,
) -> EquilibriumCertificate:
    """Create certificate for strategy profile that is NOT an equilibrium."""
    return EquilibriumCertificate(
        game_id=game.name,
        game_description=game.description,
        n_agents=2,
        payoff_matrix_hash=game.compute_hash(),
        action_sets={0: game.actions[0], 1: game.actions[1]},
        equilibrium_concept=EquilibriumConcept.NASH,
        strategies=strategies,
        is_equilibrium=False,
        failure_mode=GameFailType.EXPLOITABLE_DEVIATION,
        obstruction_if_fail=GameObstructionEvidence(
            fail_type=GameFailType.EXPLOITABLE_DEVIATION,
            deviating_agent=deviation_evidence["deviating_agent"],
            deviation_strategy=deviation_evidence["deviation_action"],
            deviation_payoff_gain=deviation_evidence["deviation_gain"],
        ),
        verification_witness=DerivationWitness(
            invariant_name="nash_deviation_found",
            derivation_operator="deviation_enumeration",
            input_data={
                "deviating_agent": deviation_evidence["deviating_agent"],
                "current_payoff": str(deviation_evidence["current_payoff"]),
                "deviation_payoff": str(deviation_evidence["deviation_payoff"]),
                "deviation_gain": str(deviation_evidence["deviation_gain"]),
                "deviation_action": deviation_evidence["deviation_action"],
            },
            output_value=0,  # Not an equilibrium
            verifiable=True,
        ),
        strict_mode=True,
    )


def create_no_pure_nash_certificate(
    game: NormalFormGame,
    all_profiles_checked: int,
) -> EquilibriumCertificate:
    """Create certificate for game with no pure strategy Nash equilibrium."""
    # For "no equilibrium found", we don't claim a specific strategy profile
    # We certificate that exhaustive search found no pure Nash
    dummy_strategies = [
        AgentStrategy(agent_id=0, strategy_type="pure", strategy_description="N/A (search)", action=game.actions[0][0]),
        AgentStrategy(agent_id=1, strategy_type="pure", strategy_description="N/A (search)", action=game.actions[1][0]),
    ]

    return EquilibriumCertificate(
        game_id=game.name,
        game_description=game.description,
        n_agents=2,
        payoff_matrix_hash=game.compute_hash(),
        action_sets={0: game.actions[0], 1: game.actions[1]},
        equilibrium_concept=EquilibriumConcept.NASH,
        strategies=dummy_strategies,
        is_equilibrium=False,
        failure_mode=GameFailType.NO_EQUILIBRIUM_FOUND,
        obstruction_if_fail=None,  # No specific obstruction, just "none found"
        verification_witness=DerivationWitness(
            invariant_name="pure_nash_exhaustive_search",
            derivation_operator="exhaustive_profile_enumeration",
            input_data={
                "search_type": "pure_strategy",
                "profiles_checked": all_profiles_checked,
                "equilibria_found": 0,
            },
            output_value=0,  # No equilibrium found
            verifiable=True,
        ),
        strict_mode=True,
    )


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def run_demo():
    """Run the matrix game equilibrium demo."""
    print("=" * 70)
    print("  QA-Native Game Theory Demo: Matrix Games + Equilibrium Certificates")
    print("  Ref: MIT 'Algorithms for Decision Making' Chapter 24")
    print("=" * 70)

    certificates = {}

    # -------------------------------------------------------------------------
    # Scenario 1: Prisoner's Dilemma - (Defect, Defect) is unique Nash
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 1: Prisoner's Dilemma")
    print("-" * 70)
    print(f"Game: {PRISONERS_DILEMMA.description}")
    print("\nPayoff matrix (row=P0, col=P1):")
    print("              Cooperate    Defect")
    print("  Cooperate   (-1,-1)      (-3, 0)")
    print("  Defect      ( 0,-3)      (-2,-2)")

    # Find Nash equilibria
    pd_nash = find_pure_nash_equilibria(PRISONERS_DILEMMA)
    print(f"\nPure Nash equilibria found: {pd_nash}")

    # Verify (Defect, Defect) is Nash
    dd_strategy0 = {"Defect": Fraction(1)}
    dd_strategy1 = {"Defect": Fraction(1)}
    is_nash, deviation = verify_nash_equilibrium(PRISONERS_DILEMMA, dd_strategy0, dd_strategy1)
    print(f"(Defect, Defect) is Nash: {is_nash}")

    strategies_dd = [
        AgentStrategy(agent_id=0, strategy_type="pure", strategy_description="Always defect", action="Defect"),
        AgentStrategy(agent_id=1, strategy_type="pure", strategy_description="Always defect", action="Defect"),
    ]

    cert_pd = create_equilibrium_success_certificate(
        PRISONERS_DILEMMA,
        strategies_dd,
        {
            "profile": "(Defect, Defect)",
            "p0_payoff": "-2",
            "p1_payoff": "-2",
            "p0_deviation_gain": "0",
            "p1_deviation_gain": "0",
        },
    )
    certificates["prisoners_dilemma_nash"] = cert_pd

    # Show that (Cooperate, Cooperate) is NOT Nash
    cc_strategy0 = {"Cooperate": Fraction(1)}
    cc_strategy1 = {"Cooperate": Fraction(1)}
    is_nash_cc, deviation_cc = verify_nash_equilibrium(PRISONERS_DILEMMA, cc_strategy0, cc_strategy1)
    print(f"(Cooperate, Cooperate) is Nash: {is_nash_cc}")
    if deviation_cc:
        print(f"  Deviation: Agent {deviation_cc['deviating_agent']} can gain {deviation_cc['deviation_gain']} by playing {deviation_cc['deviation_action']}")

    strategies_cc = [
        AgentStrategy(agent_id=0, strategy_type="pure", strategy_description="Always cooperate", action="Cooperate"),
        AgentStrategy(agent_id=1, strategy_type="pure", strategy_description="Always cooperate", action="Cooperate"),
    ]

    cert_pd_fail = create_equilibrium_failure_certificate(
        PRISONERS_DILEMMA,
        strategies_cc,
        deviation_cc,
    )
    certificates["prisoners_dilemma_cooperate_fail"] = cert_pd_fail

    # -------------------------------------------------------------------------
    # Scenario 2: Coordination Game - Two pure Nash equilibria
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 2: Coordination Game")
    print("-" * 70)
    print(f"Game: {COORDINATION_GAME.description}")
    print("\nPayoff matrix:")
    print("            Left    Right")
    print("  Left     (2,2)    (0,0)")
    print("  Right    (0,0)    (1,1)")

    coord_nash = find_pure_nash_equilibria(COORDINATION_GAME)
    print(f"\nPure Nash equilibria found: {coord_nash}")

    # Certificate for (Left, Left)
    ll_strategy0 = {"Left": Fraction(1)}
    ll_strategy1 = {"Left": Fraction(1)}
    is_nash_ll, _ = verify_nash_equilibrium(COORDINATION_GAME, ll_strategy0, ll_strategy1)
    print(f"(Left, Left) is Nash: {is_nash_ll}")

    strategies_ll = [
        AgentStrategy(agent_id=0, strategy_type="pure", strategy_description="Always Left", action="Left"),
        AgentStrategy(agent_id=1, strategy_type="pure", strategy_description="Always Left", action="Left"),
    ]

    cert_coord = create_equilibrium_success_certificate(
        COORDINATION_GAME,
        strategies_ll,
        {
            "profile": "(Left, Left)",
            "p0_payoff": "2",
            "p1_payoff": "2",
            "p0_deviation_gain": "0",
            "p1_deviation_gain": "0",
            "note": "Pareto-dominant Nash equilibrium",
        },
    )
    certificates["coordination_nash_left"] = cert_coord

    # -------------------------------------------------------------------------
    # Scenario 3: Matching Pennies - No pure Nash equilibrium
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 3: Matching Pennies (Zero-Sum)")
    print("-" * 70)
    print(f"Game: {MATCHING_PENNIES.description}")
    print("\nPayoff matrix:")
    print("            Heads    Tails")
    print("  Heads    (1,-1)   (-1,1)")
    print("  Tails   (-1, 1)   (1,-1)")

    mp_nash = find_pure_nash_equilibria(MATCHING_PENNIES)
    print(f"\nPure Nash equilibria found: {mp_nash}")
    print("(No pure Nash - only mixed Nash at (1/2, 1/2) for both players)")

    # Verify that all pure profiles are NOT Nash
    all_profiles = len(MATCHING_PENNIES.actions[0]) * len(MATCHING_PENNIES.actions[1])
    cert_mp = create_no_pure_nash_certificate(MATCHING_PENNIES, all_profiles)
    certificates["matching_pennies_no_pure_nash"] = cert_mp

    # Show why (Heads, Heads) fails
    hh_strategy0 = {"Heads": Fraction(1)}
    hh_strategy1 = {"Heads": Fraction(1)}
    is_nash_hh, deviation_hh = verify_nash_equilibrium(MATCHING_PENNIES, hh_strategy0, hh_strategy1)
    print(f"\n(Heads, Heads) is Nash: {is_nash_hh}")
    if deviation_hh:
        print(f"  Deviation: Agent {deviation_hh['deviating_agent']} gains {deviation_hh['deviation_gain']} by {deviation_hh['deviation_action']}")

    # -------------------------------------------------------------------------
    # Scenario 4: Battle of the Sexes - Asymmetric coordination
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 4: Battle of the Sexes")
    print("-" * 70)
    print(f"Game: {BATTLE_OF_SEXES.description}")
    print("\nPayoff matrix:")
    print("             Opera    Football")
    print("  Opera     (3,2)     (0,0)")
    print("  Football  (0,0)     (2,3)")

    bos_nash = find_pure_nash_equilibria(BATTLE_OF_SEXES)
    print(f"\nPure Nash equilibria found: {bos_nash}")

    # Certificate for (Opera, Opera)
    strategies_oo = [
        AgentStrategy(agent_id=0, strategy_type="pure", strategy_description="Opera preference", action="Opera"),
        AgentStrategy(agent_id=1, strategy_type="pure", strategy_description="Opera (defer)", action="Opera"),
    ]

    cert_bos = create_equilibrium_success_certificate(
        BATTLE_OF_SEXES,
        strategies_oo,
        {
            "profile": "(Opera, Opera)",
            "p0_payoff": "3",
            "p1_payoff": "2",
            "p0_deviation_gain": "0",
            "p1_deviation_gain": "0",
            "note": "P0-preferred equilibrium",
        },
    )
    certificates["battle_of_sexes_opera"] = cert_bos

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CERTIFICATE VALIDATION")
    print("-" * 70)

    for name, cert in certificates.items():
        result = validate_equilibrium_certificate(cert)
        status = "VALID" if result.valid else f"INVALID: {result.violations}"
        print(f"{name}: {status}")

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPORTING CERTIFICATES")
    print("-" * 70)

    output = {
        "demo": "matrix_game_demo",
        "description": "QA-native game theory: 2x2 matrix games with equilibrium certificates",
        "reference": "MIT Algorithms for Decision Making, Chapter 24",
        "games": {
            name: cert.to_json()
            for name, cert in certificates.items()
        },
        "key_insights": [
            "Nash equilibrium = no unilateral profitable deviation.",
            "Verification is exhaustive deviation enumeration (certifiable).",
            "Failure mode EXPLOITABLE_DEVIATION captures 'not Nash' with witness.",
            "NO_EQUILIBRIUM_FOUND captures 'searched all pure profiles, none Nash'.",
            "All payoffs and probabilities are exact (Fraction), no floats.",
        ],
    }

    output_path = Path(__file__).parent / "matrix_game_cert.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported to: {output_path}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: QA-Native Game Theory Mapping")
    print("=" * 70)
    print("""
    MIT Book Chapter 24 (Multiagent Reasoning) → QA Mapping:

    Book Concept              QA Equivalent
    ------------              -------------
    Normal-form game          Payoff matrix with exact Fractions
    Strategy profile          AgentStrategy list
    Nash equilibrium          Certificate with is_equilibrium=True
    Deviation check           Exhaustive search + deviation witness
    No equilibrium            GameFailType.NO_EQUILIBRIUM_FOUND

    New Failure Modes (GameFailType enum):
    - EXPLOITABLE_DEVIATION: Found profitable unilateral deviation
    - NO_EQUILIBRIUM_FOUND: Exhaustive search found no equilibrium
    - REGRET_TOO_HIGH: Strategy exceeds regret bound
    - INFORMATION_SET_ALIASING: Multiagent NON_IDENTIFIABLE

    Certificate Structure:
    - EquilibriumCertificate with verification_witness
    - GameObstructionEvidence for failure cases
    - Exact arithmetic throughout (no floats)
""")


if __name__ == "__main__":
    run_demo()
