<!-- PRIMARY-SOURCE-EXEMPT: reason="QA-MEM Kochenderfer/Wheeler 2026 'Algorithms for Optimization' (2nd edition) primary-source excerpt snapshot. Verbatim quotes from optimization.pdf 2e (631pp). Page locators are PDF page (/printed page). 15 anchors covering optimization formulation, descent methods, line search, hypergradient descent (Kochenderfer's named algorithm — distinct from any QA HGD), simulated annealing, multiobjective, surrogate / Bayesian, uncertainty-aware, discrete branch and bound." -->

# Kochenderfer / Wheeler 2026 (2nd ed.) — Algorithms for Optimization — Primary-Source Excerpts

**Corpus root:** `Documents/kochenderfer_corpus/`
**Scope:** QA-MEM ingestion of *Algorithms for Optimization* (2nd edition) as the third pillar of the Kochenderfer trilogy after Validation (2026) and Decision Making (2022). The Optimization book hosts the formal vocabulary for descent methods, line search, second-order methods, derivative-free / direct methods, stochastic optimization, simulated annealing, surrogate / Bayesian optimization, multiobjective optimization, optimization under uncertainty, and discrete / integer programming — many of which the QA cert ecosystem already has counterparts for under different names (HGD, reachability descent, integer-only state evolution, surrogate-as-observer-projection).

**Two editions on disk** (per QA-MEM Phase 4.x discipline: do not collapse editions):
- **2nd edition** (canonical anchor for this fixture): *Algorithms for Optimization*, MIT Press 2026, CC-BY-NC-ND. 631 pp. Compiled 2026-04-22 09:01:45-07:00 (LuaHBTeX, TeX Live 2024). Title page says "Algorithms for Optimization | second edition". On disk: `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_optimization_2e.pdf` (18.9 MB; staged 2026-04-27).
- **1st edition** (kept as separate witness): *Algorithms for Optimization*, MIT Press 2019. 520 pp. PDF compiled 2022-05-22 (GPL Ghostscript 9.53.3). On disk: `Documents/kochenderfer_corpus/kochenderfer_wheeler_2019_algorithms_for_optimization_1e.pdf` (8.3 MB; staged 2026-04-27). Anchored separately as a SourceWork in the fixture; the 2nd edition `supersedes` it. The 1st edition is the version the QA cert ecosystem was originally cross-referenced against (e.g., the historic `(Kochenderfer + Wheeler 2019)` citations in earlier bridge docs).

**Domain:** unclassified (empty string in fixture). Same `formal_methods`-extension deferral as the Validation and Decision Making fixtures.

**Authors:** Mykel J. Kochenderfer (Stanford), Tim A. Wheeler. CC-BY-NC-ND. Companion repos: `github.com/algorithmsbooks/{algforopt-notebooks, optimization, optimization-ancillaries}`.

**QA-research grounding:** Optimization book content has the most overlap with QA's existing cert ecosystem: §4 Local Descent / §5 First-Order Methods (descent direction iteration `x^(k+1) = x^(k) + α^(k) d^(k)`) is the canonical form of QA reachability descent (`qa_reachability_descent_run_cert_v1`); §5.9 Hypergradient Descent (Baydin et al. 2018, Kochenderfer's specific named algorithm — derivative of step factor `α^(k) = α^(k-1) + μ(g^(k))^T g^(k-1)`) shares the name "HGD" with QA's harmonic gradient descent but is a **different** mathematical object (Kochenderfer's HGD is meta-learning the float step-factor of a continuous gradient method; QA's HGD operates on integer orbit graphs). This bridge does NOT claim QA HGD = Kochenderfer HGD. §22.4 Branch and Bound (LP-relaxation + integer branch + bound-pruning) maps onto QA reachability-descent + cert [191] tier filtering on the discrete side. §17 Surrogate Models and §19.5 Expected Improvement (Bayesian optimization) are observer-projection candidates — they belong at the input boundary if used at all, never as causal QA dynamics.

**Theorem NT directionality (per the bridge spec's standing rule):** Optimization vocabulary represents QA where appropriate; it does not redefine QA. QA remains primitively a control / reachability theory; the algorithms in this book provide an external interpretive vocabulary for the descent / search / pruning machinery QA already uses on integer state spaces.

---

## Chapter 1 — Introduction (foundations)

### #opt-1-3-mathematical-formulation (PDF p.25 / printed p.5, §1.3)

> "An optimization problem can be formulated mathematically as follows: minimize_x f(x) subject to x ∈ X (1.1). Here, x is a design point, which can be represented as a vector of values corresponding to different design variables. […] The elements in this vector can be adjusted to minimize the objective function f. Any value of x in the feasible set X that minimizes the objective function is called a solution or minimizer. […] The feasible set X may be defined in terms of a set of constraints. Each constraint limits the set of possible solutions, and together the constraints define the feasible set X."

### #opt-1-5-minima-local-vs-global (PDF p.30 / printed p.10, §1.5)

> "When minimizing f, we wish to find a global minimizer, a value of x for which f(x) is minimized. A function may have at most one global minimum, but it may have multiple global minimizers. Unfortunately, it is generally difficult to prove that a given candidate point is at a global minimum. Often, the best we can do is check whether it is at a local minimum. […] A strong local minimizer, also known as a strict local minimizer, is a point that uniquely minimizes f within a neighborhood. […] A weak local minimizer is a local minimizer that is not a strong local minimizer."

### #opt-1-6-optimality-conditions (PDF p.30-32 / printed p.10-12, §1.6)

> "In cases where the objective function is twice differentiable (at least at the minimizer), we can establish conditions for optimality. […] Univariate Conditions: A design point is guaranteed to be at a strong local minimum if the local derivative is zero and the second derivative is positive: 1. f'(x*) = 0, 2. f''(x*) > 0. […] Multivariate Conditions: 1. ∇f(x) = 0, the first-order necessary condition. 2. ∇²f(x) is positive semidefinite, the second-order necessary condition. The first-order necessary condition tells us that the function is not changing at x. […] The second-order necessary condition tells us that x is in a bowl."

---

## Chapter 4 — Local Descent

### #opt-4-1-descent-direction-iteration (PDF p.81-82 / printed p.61-62, §4.1)

> "A common approach to optimization is to incrementally improve a design point x by taking a step that attempts to minimize the objective value based on a local model. The local model may be obtained, for example, from a first- or second-order Taylor approximation. Optimization algorithms that follow this general approach are referred to as descent direction methods. They start with a design point x^(1) and then generate a sequence of points, sometimes called iterates, to converge to a local minimum. The iterative descent direction procedure involves the following steps: 1. Check whether x^(k) satisfies the termination conditions. […] 2. Determine the descent direction d^(k) using local information such as the gradient or Hessian. […] 3. Determine the step factor α^(k). […] 4. Compute the next design point according to: x^(k+1) ← x^(k) + α^(k) d^(k) (4.1)."

### #opt-4-3-line-search (PDF p.84 / printed p.64, §4.3)

> "Instead of using a fixed or decaying step factor, we can use line search to directly optimize the step factor to minimize the objective function given a descent direction d: minimize_α f(x + αd) (4.6). Line search is a univariate optimization problem, a class of problems covered in chapter 3. To inform the search, we can use the derivative of the line search objective, which is simply the directional derivative along d at x + αd. […] One disadvantage of conducting a line search at each step is the computational cost of optimizing α to a high degree of precision. Instead, it is common to quickly find a reasonable value and then move on, selecting x^(k+1), and then picking a new direction d^(k+1)."

---

## Chapter 5 — First-Order Methods

### #opt-5-1-gradient-descent-steepest-direction (PDF p.97-98 / printed p.77-78, §5.1)

> "The gradient descent method uses the gradient to select the next descent direction d. […] The motivation for gradient descent comes from the first-order Taylor series approximation about our current iterate x^(k): f(x^(k) + αd) ≈ f(x^(k)) + αd^T g^(k) (5.2). We can choose the direction d that minimizes this first-order approximation subject to the constraint that ‖d‖ = 1. […] The direction that minimizes this first-order approximation is the direction of steepest descent, which is simply the direction opposite the gradient: d^(k) = -g^(k)/‖g^(k)‖ (5.3). […] Following the direction of steepest descent is guaranteed to lead to improvement, provided that the objective function is smooth, the step factor α is sufficiently small, and we are not already at a point where the gradient is zero."

### #opt-5-9-hypergradient-descent-baydin-2018 (PDF p.106-108 / printed p.86-88, §5.9)

> "Hypergradient descent was developed with the understanding that the derivative of the step factor should be useful for improving optimizer performance. A hypergradient is a derivative taken with respect to a hyperparameter. Hypergradient algorithms reduce the sensitivity to the hyperparameter, allowing it to adapt more quickly. Hypergradient descent applies gradient descent to the step factor of an underlying descent method. The method requires the partial derivative of the objective function with respect to the step factor. […] The resulting update rule is: α^(k) = α^(k-1) - μ ∂f(x^(k))/∂α^(k-1) = α^(k-1) + μ(g^(k))^T g^(k-1) (5.38), where μ is the hypergradient step factor. […] [Baydin, Cornish, Rubio, Schmidt, Wood, 'Online Learning Rate Adaptation with Hypergradient Descent', ICLR 2018.]"

---

## Chapter 6 — Second-Order Methods

### #opt-6-1-newtons-method (PDF p.115-116 / printed p.95-96, §6.1)

> "Knowing the function value and gradient for a design point can help determine the direction to travel, but this first-order information does not directly help determine how far to step to reach a local minimum. Second-order information, on the other hand, allows us to make a quadratic approximation of the objective function and approximate the right step size to reach a local minimum. […] In univariate optimization, the quadratic approximation about a point x^(k) comes from the second-order Taylor expansion: q(x) = f(x^(k)) + (x - x^(k)) f'(x^(k)) + ((x - x^(k))² / 2) f''(x^(k)) (6.1). Setting the derivative to zero and solving for the root yields the update equation for Newton's method: x^(k+1) = x^(k) - f'(x^(k))/f''(x^(k)) (6.3). […] Newton's method does tend to converge quickly when in a bowl-like region that is sufficiently close to a local minimum. It has quadratic convergence."

---

## Chapter 7 — Direct Methods (derivative-free)

### #opt-7-1-cyclic-coordinate-search (PDF p.133-134 / printed p.113-114, §7.1)

> "Direct methods rely solely on the objective function f. These methods are also called zero-order, black box, pattern search, or derivative-free methods. Direct methods do not rely on derivative information to guide them toward a local minimum or identify when they have reached a local minimum. They use other criteria to choose the next search direction and to judge when they have converged. Cyclic coordinate search, also known as coordinate descent or taxicab search, simply alternates between coordinate directions for its line search. […] This process is equivalent to doing a sequence of line searches along the set of n basis vectors. […] Like steepest descent, cyclic coordinate search is guaranteed either to improve or to remain the same with each iteration."

---

## Chapter 8 — Stochastic Methods

### #opt-8-4-simulated-annealing (PDF p.160-162 / printed p.140-142, §8.4)

> "Simulated annealing borrows inspiration from metallurgy. Temperature is used to control the degree of stochasticity during the randomized search. The temperature starts high, allowing the process to freely move about the search space, with the hope that in this phase the process will find a good region with the best local minimum. The temperature is then slowly brought down, reducing the stochasticity and forcing the search to converge to a minimum. […] At every iteration, a candidate transition from x to x' is sampled from a transition distribution T and is accepted with probability { 1 if Δy ≤ 0; e^{-Δy/t} if Δy > 0 (8.6) where Δy = f(x') - f(x) is the difference in the objective and t > 0 is the temperature. This acceptance probability, known as the Metropolis criterion, allows the algorithm to escape from local minima when the temperature is high. […] An exponential annealing schedule, which is more common, uses a simple decay factor t^(k+1) = γt^(k) for some γ ∈ (0, 1)."

---

## Chapter 15 — Multiobjective Optimization

### #opt-15-1-pareto-optimality (PDF p.345-346 / printed p.325-326, §15.1)

> "The notion of Pareto optimality is useful when discussing problems where there are multiple objectives. A design is Pareto optimal if it is impossible to improve in one objective without worsening at least one other objective. In multiobjective design optimization, we can generally focus our efforts on designs that are Pareto optimal without having to commit to a particular trade-off between objectives. […] In multiobjective optimization, our objective function f returns an m-dimensional vector of values y when evaluated at a design point x. The different dimensions of y correspond to different objectives, sometimes also referred to as metrics or criteria. We can objectively rank two design points x and x' only when one is better in at least one objective and no worse in any other. That is, x dominates x' if and only if f_i(x) ≤ f_i(x') for i in 1:m and f_i(x) < f_i(x') for some i (15.1)."

---

## Chapter 17 — Surrogate Models

### #opt-17-1-fitting-surrogate-models (PDF p.385-386 / printed p.365-366, §17.1)

> "A surrogate model f̂ parameterized by θ is designed to mimic the true objective function f. The parameters θ can be adjusted to fit the model based on samples collected from f. […] Fitting a model to a set of points requires tuning the parameters to minimize the difference between the true evaluations and those predicted by the model, typically according to an L_p norm: minimize_θ ‖y - ŷ‖_p (17.4). […] Equation (17.4) penalizes the deviation of the model only at the data points. There is no guarantee that the model will continue to fit well away from observed data, and model accuracy typically decreases the farther we go from the sampled points. This form of model fitting is called regression."

---

## Chapter 19 — Surrogate Optimization (Bayesian optimization)

### #opt-19-5-expected-improvement (PDF p.434-435 / printed p.414-415, §19.5)

> "Optimization is concerned with finding the minimum of the objective function. While maximizing the probability of improvement will tend to decrease the objective function over time, it does not improve very much with each iteration. We can focus our exploration of points that maximize our expected improvement over the current best function value. […] We can calculate the expected improvement using the distribution predicted by the Gaussian process: E[I(y)] = (y_min - μ̂) P(y ≤ y_min) + σ̂² N(y_min | μ̂, σ̂²) (19.12)."

---

## Chapter 20 — Optimization Under Uncertainty

### #opt-20-2-set-based-uncertainty-minimax (PDF p.449-451 / printed p.429-431, §20.2)

> "Set-based uncertainty approaches assume that z belongs to a set Z, but these approaches make no assumptions about the relative likelihood of different points within that set. The set Z can be defined in different ways. […] In problems with set-based uncertainty, we often want to minimize the maximum possible value of the objective function. Such a minimax approach solves the optimization problem: minimize_{x ∈ X} maximize_{z ∈ Z} f(x, z) (20.1). In other words, we want to find an x that minimizes f, assuming the worst-case value for z. […] The minimax problem preserves convexity. In other words, if f(x, z) is convex in x for each fixed z, then f_mod(x) = maximize_{z ∈ Z} f(x, z) is also convex in x, allowing us to use convex optimization methods to solve the minimax problem."

---

## Chapter 22 — Discrete Optimization

### #opt-22-4-branch-and-bound-integer (PDF p.491-492 / printed p.471-472, §22.4)

> "One method for finding the global optimum of a discrete problem, such as an integer program, is to enumerate all possible solutions. The branch and bound method guarantees that an optimal solution is found without having to evaluate all possible solutions. Many commercial integer program solvers use ideas from both the cutting plane method and branch and bound. The method gets its name from the branch operation that partitions the solution space and the bound operation that computes a lower bound for a partition. […] We branch by considering two new LPs, each one created by adding one of the following constraints to the dequeued LP: x_i ≤ ⌊x*_{i,c}⌋ or x_i ≥ ⌈x*_{i,c}⌉ (22.12). […] If either solution lowers the objective value when compared to the best integral solution seen so far, it is placed into the priority queue. Not placing solutions already known to be inferior to the best integral solution seen thus far allows branch and bound to prune the search space."

---

## Acquisition / ingestion notes

- **2026-04-27**: Both PDFs staged from `~/Downloads/optimization.pdf` and `~/Downloads/optimization-1e-1.pdf` to `Documents/kochenderfer_corpus/` via heredoc-scoped `shutil.copy2`. Allowlist constraint unchanged.
- 15 anchors selected from a 631-page text (2nd edition, canonical) on the criteria: foundational definitions (§1.3, §1.5, §1.6), descent-method scaffolding (§4.1, §4.3, §5.1, §5.9, §6.1, §7.1), stochastic-combinatorial baseline (§8.4), multiobjective and surrogate / uncertainty / discrete extensions (§15.1, §17.1, §19.5, §20.2, §22.4).
- Bridge mapping is in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §8 (added in the same commit), evolving the existing bridge additively per the standing rule.
- Hypergradient Descent anchor (§5.9) deliberately preserves the Kochenderfer-specific definition (Baydin et al. 2018, ICLR — derivative of step factor for a continuous gradient method) and does **not** claim it equals QA's harmonic gradient descent. The bridge §8.2 will list this as `candidate` only after the QA-side HGD definition has been pinned to a canonical reference.
- 1st edition kept as separate SourceWork with `supersedes` edge from the 2nd edition. The 1st edition is the version most pre-2026 QA cert citations refer to (`Kochenderfer + Wheeler 2019`).
- Trilogy ingestion now complete: Validation (2026, 441pp, 15 claims), Decision Making (2022, 700pp, 15 claims), Optimization (2026 2e, 631pp, 15 claims; 1st ed kept). Bridge spec evolves to §8.
