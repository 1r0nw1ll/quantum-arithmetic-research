<!-- PRIMARY-SOURCE-EXEMPT: reason="QA-MEM Kochenderfer/Wheeler/Wray 2022 'Algorithms for Decision Making' primary-source excerpt snapshot. Verbatim quotes from dm.pdf (700pp). Page locators are PDF page (/printed page). 15 anchors covering MDP foundation, value iteration, online planning, POMDP/belief-state, multi-agent extensions." -->

# Kochenderfer / Wheeler / Wray 2022 — Algorithms for Decision Making — Primary-Source Excerpts

**Corpus root:** `Documents/kochenderfer_corpus/`
**Scope:** QA-MEM ingestion of *Algorithms for Decision Making* as the second pillar of the Kochenderfer trilogy (after Validation, before Optimization). The DM book hosts the formal vocabulary that maps most directly onto QA's existing reachability machinery: MDP / Bellman backup / value iteration on finite state spaces / online forward search. Where Validation (Kochenderfer 2026) gave the QA cert ecosystem its outsider-legible vocabulary, Decision Making (Kochenderfer 2022) gives QA's reachability-descent + orbit-graph machinery a Stanford/MIT-grade canonical reference for the same algorithms.
**Domain:** unclassified (empty string in fixture). Same `formal_methods`-extension deferral as the Validation fixture; consistency over taxonomy churn.

**Publication:** M. J. Kochenderfer, T. A. Wheeler, K. H. Wray, *Algorithms for Decision Making*. MIT Press, 2022. CC-BY-NC-ND. Compiled 2025-09-21 10:49:56-07:00 (LuaHBTeX, TeX Live 2024). 700 PDF pages.
**Authors:** Mykel J. Kochenderfer (Stanford), Tim A. Wheeler, Kyle H. Wray.
**On-disk:** `Documents/kochenderfer_corpus/kochenderfer_wheeler_wray_2022_algorithms_for_decision_making.pdf` (12.1 MB; staged 2026-04-27 from local download). Companion repos: `github.com/algorithmsbooks/{decisionmaking, decisionmaking-code, decisionmaking-ancillaries, DecisionMakingProblems.jl}`.

**QA-research grounding:** the book's Part II (Sequential Problems) is essentially the formal vocabulary of QA's existing reachability descent + orbit-graph certs. §7.5 Value Iteration's Bellman backup `U_{k+1}(s) = max_a [R(s,a) + γ Σ T(s'|s,a) U_k(s')]` specializes on QA's deterministic orbit graph (no transition randomness on the QA-discrete side; T is the QA generator) to a path-cost minimization that the existing `qa_reachability_descent_run_cert_v1` already certifies. §9.3 Forward Search's depth-first expansion to depth d is exactly the BFS used by cert [191] qa_bateson_learning_levels for the tiered-reachability theorem on S_9. §7.1 MDP / §19.2 POMDP / §27.1 Dec-POMDP partition the design space along the same Theorem-NT-firewall axis that the QA cert ecosystem already follows.

**Firewall note:** continuous-domain methods in this book (Kalman filters §19.3, particle filters §19.5, policy gradient Ch. 11-12, model-based RL Ch. 14, neural value approximation §8.6, linear-quadratic regulator §7.8) cannot enter the QA discrete layer as causal inputs (Theorem NT). They become observer-projection candidates only at the input/output boundaries. The discrete subset (Ch. 7 finite MDP, Ch. 9 forward search / branch and bound, Ch. 19.2 discrete state filter, Ch. 20 exact belief-state planning, Ch. 25 Markov games, Ch. 27 Dec-POMDP) is the natural QA-aligned scope.

---

## Chapter 1 — Introduction

### #dm-1-1-decision-making-observe-act-cycle (PDF p.23 / printed p.1, §1.1)

> "An agent is an entity that acts based on observations of its environment. Agents may be physical entities, like humans or robots, or they may be nonphysical entities, such as decision support systems that are implemented entirely in software. As shown in figure 1.1, the interaction between the agent and the environment follows an observe-act cycle or loop. The agent at time t receives an observation of the environment, denoted as o_t. […] The agent then chooses an action a_t through some decision-making process."

### #dm-1-1-four-uncertainties (PDF p.24 / printed p.2, §1.1)

> "Given the past sequence of observations, o_1, …, o_t, and knowledge of the environment, the agent must choose an action a_t that best achieves its objectives in the presence of various sources of uncertainty, including the following: outcome uncertainty, where the effects of our actions are uncertain; model uncertainty, where our model of the problem is uncertain; state uncertainty, where the true state of the environment is uncertain; and interaction uncertainty, where the behavior of the other agents interacting in the environment is uncertain. This book is organized around these four sources of uncertainty."

---

## Chapter 6 — Simple Decisions

### #dm-6-4-maximum-expected-utility (PDF p.138 / printed p.116, §6.4)

> "We are interested in the problem of making rational decisions with imperfect knowledge of the state of the world. Suppose that we have a probabilistic model P(s' | o, a), which represents the probability that the state of the world becomes s', given that we observe o and take action a. We have a utility function U(s') that encodes our preferences over the space of outcomes. Our expected utility of taking action a, given observation o, is given by EU(a | o) = Σ_{s'} P(s' | a, o) U(s'). The principle of maximum expected utility says that a rational agent should choose the action that maximizes expected utility: a* = arg max_a EU(a | o)."

---

## Chapter 7 — Exact Solution Methods (the QA-rich chapter)

### #dm-7-1-mdp-definition (PDF p.155 / printed p.133, §7.1)

> "In an MDP (algorithm 7.1), we choose action a_t at time t based on observing state s_t. We then receive a reward r_t. The action space A is the set of possible actions, and the state space S is the set of possible states. Some of the algorithms assume that these sets are finite, but this is not required in general. The state evolves probabilistically based on the current state and action we take. The assumption that the next state depends only on the current state and action and not on any prior state or action is known as the Markov assumption. […] The state transition model T(s' | s, a) represents the probability of transitioning from state s to s' after executing action a. The reward function R(s, a) represents the expected reward received when executing action a from state s."

### #dm-7-2-policy-evaluation-lookahead-equation (PDF p.158 / printed p.136, §7.2)

> "The expected utility of executing π from state s is denoted as U^π(s). In the context of MDPs, U^π is often referred to as the value function. An optimal policy π* is a policy that maximizes expected utility: π*(s) = arg max_π U^π(s) for all states s. […] An optimal policy can be found by using a computational technique called dynamic programming, which involves simplifying a complicated problem by breaking it down into simpler subproblems in a recursive manner. […] Policy evaluation can be done iteratively. […] U^π_{k+1}(s) = R(s, π(s)) + γ Σ_{s'} T(s' | s, π(s)) U^π_k(s'). […] Convergence is guaranteed because the update in equation (7.5) is a contraction mapping."

### #dm-7-2-bellman-expectation-equation (PDF p.160 / printed p.138, §7.2)

> "At convergence, the following equality holds: U^π(s) = R(s, π(s)) + γ Σ_{s'} T(s' | s, π(s)) U^π(s'). […] This equality is called the Bellman expectation equation. Policy evaluation can be done without iteration by solving the system of equations in the Bellman expectation equation directly. Equation (7.6) defines a set of |S| linear equations with |S| unknowns corresponding to the values at each state. […] Solving for U^π in this way requires O(|S|^3) time."

### #dm-7-5-value-iteration-bellman-backup (PDF p.163 / printed p.141, §7.5)

> "Value iteration is an alternative to policy iteration that is often used because of its simplicity. Unlike policy improvement, value iteration updates the value function directly. It begins with any bounded value function U, meaning that |U(s)| < ∞ for all s. […] The value function can be improved by applying the Bellman backup, also called the Bellman update: U_{k+1}(s) = max_a [ R(s, a) + γ Σ_{s'} T(s' | s, a) U_k(s') ]. […] Repeated application of this update is guaranteed to converge to the optimal value function. […] This optimal policy is guaranteed to satisfy the Bellman optimality equation: U*(s) = max_a [ R(s, a) + γ Σ_{s'} T(s' | s, a) U*(s') ]."

### #dm-7-7-linear-program-formulation (PDF p.169 / printed p.147, §7.7)

> "The problem of finding an optimal policy can be formulated as a linear program, which is an optimization problem with a linear objective function and a set of linear equality or inequality constraints. […] minimize Σ_s U(s) subject to U(s) ≥ R(s, a) + γ Σ_{s'} T(s' | s, a) U(s') for all s and a. […] In the linear program shown in equation (7.20), the number of variables is equal to the number of states and the number of constraints is equal to the number of states times the number of actions. Because linear programs can be solved in polynomial time, MDPs can be solved in polynomial time as well."

---

## Chapter 9 — Online Planning

### #dm-9-3-forward-search (PDF p.205 / printed p.183, §9.3)

> "Forward search determines the best action to take from an initial state s by expanding all possible transitions up to depth d. These expansions form a search tree. Such search trees have a worst-case branching factor of |S| × |A|, yielding a computational complexity of O((|S| × |A|)^d). Figure 9.1 shows a search tree applied to a problem with three states and two actions. Figure 9.2 visualizes the states visited during forward search on the hex world problem. […] If we simply want to plan to the specified horizon, we set U(s) = 0."

### #dm-9-4-branch-and-bound (PDF p.207 / printed p.185, §9.4)

> "Branch and bound (algorithm 9.3) attempts to avoid the exponential computational complexity of forward search. It prunes branches by reasoning about bounds on the value function. The algorithm requires knowing a lower bound on the value function U(s) and an upper bound on the action value function Q(s, a). The lower bound is used to evaluate the states at the maximum depth. This lower bound is propagated upward through the tree through Bellman updates. If we find that the upper bound of an action at a state is lower than the lower bound of a previously explored action from that state, then we need not explore that action, allowing us to prune the associated subtree from consideration."

---

## Chapter 19 — Beliefs (POMDP entry)

### #dm-19-2-discrete-state-filter (PDF p.402 / printed p.380, §19.2)

> "In a POMDP, the agent does not directly observe the underlying state of the environment. Instead, the agent receives an observation, which belongs to some observation space O, at each time step. The probability of observing o, given that the agent took action a and transitioned to state s', is given by O(o | a, s'). […] A kind of inference known as recursive Bayesian estimation can be used to update our belief distribution over the current state, given the most recent action and observation. We use b(s) to represent the probability (or probability density for continuous state spaces) assigned to state s. […] When the state and observation spaces are finite, we can use a discrete state filter to perform this inference exactly. Beliefs for problems with discrete state spaces can be represented using categorical distributions, where a probability mass is assigned to each state."

---

## Chapter 20 — Exact Belief State Planning

### #dm-20-1-belief-state-mdp (PDF p.429 / printed p.407, §20.1)

> "Any POMDP can be viewed as an MDP that uses beliefs as states, also called a belief-state MDP. The state space of a belief-state MDP is the set of all beliefs B. The action space is identical to that of the POMDP. The reward function for a belief-state MDP depends on the belief and action taken. It is simply the expected value of the reward. For a discrete state-space, it is given by R(b, a) = Σ_s R(s, a) b(s). […] Solving belief-state MDPs is challenging because the state space is continuous."

### #dm-20-5-pomdp-value-iteration (PDF p.438 / printed p.416, §20.5)

> "The value iteration algorithm for MDPs can be adapted for POMDPs. POMDP value iteration (algorithm 20.8) begins by constructing all one-step plans. We prune any plans that are never optimal for any initial belief. Then, we expand all combinations of one-step plans to produce two-step plans. Again, we prune any suboptimal plans from consideration. This procedure of alternating between expansion and pruning is repeated until the desired horizon is reached."

---

## Chapter 25 — Sequential Problems (multi-agent)

### #dm-25-1-markov-games (PDF p.539 / printed p.517, §25.1)

> "An MG (algorithm 25.1) extends a simple game to include a shared state s ∈ S. The likelihood of transitioning from a state s to a state s' under a joint action a is given by the transition distribution T(s' | s, a). Each agent i receives a reward according to its own reward function R_i(s, a), which now also depends on the state. […] In an infinite-horizon discounted game, the utility for agent i from state s is U^π,i(s) = R_i(s, π(s)) + γ Σ_{s'} T(s' | s, π(s)) U^π,i(s')."

---

## Chapter 27 — Collaborative Agents (Dec-POMDP)

### #dm-27-1-dec-pomdp (PDF p.567 / printed p.545, §27.1)

> "A Dec-POMDP (algorithm 27.1) is a POMG where all agents share the same objective. Each agent i ∈ I selects a local action a_i ∈ A_i based on a history of local observations o_i ∈ O_i. The true state of the system s ∈ S is shared by all agents. A single reward is generated by R(s, a) based on state s and joint action a. The goal of all agents is to maximize the shared expected reward over time under local partial observability. […] Both Dec-POMDP and Dec-MDP problems are NEXP-complete when the number of steps in the horizon is fewer than the number of states."

---

## Acquisition / ingestion notes

- **2026-04-27**: PDF staged from `~/Downloads/dm.pdf` to `Documents/kochenderfer_corpus/kochenderfer_wheeler_wray_2022_algorithms_for_decision_making.pdf` via heredoc-scoped `shutil.copy2`. Same allowlist constraint as the Validation ingest — `Documents/kochenderfer_corpus/` is not yet in `DOCUMENTS_PDF_INGRESS_PREFIXES`; allowlist extension deferred to a focused Codex-quarantine-review session.
- 15 anchors selected from a 700-page text on the criteria: definitional weight (§1.1, §6.4, §7.1), direct QA-machinery map (§7.2 Bellman expectation equation, §7.5 Bellman backup, §9.3 forward search, §9.4 branch and bound), partial observability scope (§19.2, §20.1, §20.5), and multi-agent extensions (§25.1, §27.1) for the open territory cert candidates the bridge spec §7 will list.
- Bridge mapping is in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §7 (added in the same commit), evolving the existing Validation bridge additively per the bridge spec's standing rule.
- Companion books queued: `optimization.pdf` + `optimization-1e-1.pdf` (Kochenderfer + Wheeler, *Algorithms for Optimization*).
