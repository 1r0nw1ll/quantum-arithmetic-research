# Paper 3: RML - Experimental Design (LaTeX-Ready)
## Meta-Policy Learning Over QA Manifolds

**Date**: 2025-12-29
**Status**: DESIGN LOCKED - READY TO IMPLEMENT

---

## Core Thesis (Paper 3)

> **"Learning can be done by querying structure, not optimizing loss."**

**What this means**:
- Paper 2 proved: QAWM learns **which worlds are possible** (topology)
- Paper 3 proves: Policies can **control using structural queries**, not reward maximization

**Paradigm shift**: Planning guided by learned impossibility, not gradient descent.

---

## The Object: Policy Over Generators

### Definition

A policy over generator alphabet:
$$\pi_\theta(g \mid s, \tau)$$

Where:
- $s \in \mathcal{S}_q$ : current QA state
- $\tau$ : task specification (target SCC / resonance class / return-in-k goal)
- $g \in \{\sigma, \mu, \lambda_2, \nu\}$ : generator choice

### Critical Constraint (This Makes It Not RL)

> **The policy NEVER predicts next states.**
> It only queries QAWM's structural predictions.

**What the policy can query**:
1. $P_{\text{QAWM}}(\text{legal} \mid s, g)$ : predicted legality
2. $P_{\text{QAWM}}(\text{return-in-k} \mid s, \tau)$ : predicted reachability
3. $P_{\text{QAWM}}(\text{fail-type} \mid s, g)$ : predicted obstruction (if illegal)

**What the policy CANNOT query**:
- $P(s' \mid s, g)$ : next state distribution (this would be model-based RL)
- $V(s)$ : value function over states
- $Q(s, g)$ : action-value function

**Key distinction**: Policy uses **structural predicates**, not dynamics.

---

## Experimental Design (Minimal, Tight)

### Task Definition

**Standard task** (used across all baselines):
- **Start state**: Random $(b, e)$ from $\text{Caps}(30,30)$
- **Target class**: Diagonal states $\mathcal{R}^* = \{(b,b) : 1 \le b \le 30\}$
- **Horizon**: $k = 10$ steps
- **Generators**: $\Sigma = \{\sigma, \mu, \lambda_2, \nu\}$

**Success criterion**: $s_T \in \mathcal{R}^*$ within $T \le k$ steps

**Primary metric**: **Oracle calls to success** (not cumulative reward)

---

## Baselines (Implement in This Order)

### Baseline 1: Random-Legal

**Description**: Choose uniformly among legal generators.

**Pseudocode**:
```python
def random_legal_policy(s, oracle):
    legal_gens = [g for g in generators if oracle.is_legal(s, g)]
    if len(legal_gens) == 0:
        return None  # Stuck
    return random.choice(legal_gens)
```

**Oracle calls per step**: 4 (one legality check per generator)

**Expected performance**: Poor (no guidance toward target)

**Purpose**: Lower bound baseline

---

### Baseline 2: Oracle-Greedy (Upper Bound)

**Description**: Use **true oracle** to compute return-in-k for each legal generator, pick best.

**Pseudocode**:
```python
def oracle_greedy_policy(s, target_class, k, oracle):
    best_g = None
    best_reachable = False

    for g in generators:
        if oracle.is_legal(s, g):
            next_s = oracle.step(s, g)
            reachable = oracle.return_in_k(next_s, target_class, k-1, generators)

            if reachable:
                best_g = g
                best_reachable = True
                break  # Found a path, take it

    if best_g is None:
        # No reachable move, pick random legal
        return random_legal_policy(s, oracle)

    return best_g
```

**Oracle calls per step**:
- Legality checks: 4
- Return-in-k queries: up to 4 (expensive BFS, but upper bound)
- **Total**: ~8-12 per step (worst case)

**Expected performance**: Near-optimal (uses ground truth)

**Purpose**: Ceiling / upper bound on achievable performance

**Note**: This is NOT cheating - it defines the theoretical maximum given the oracle.

---

### Baseline 3: QAWM-Greedy (Key Result, No Learning Yet)

**Description**: Score generators using **learned QAWM predictions**, pick best.

**Scoring function**:
$$\text{score}(g \mid s, \tau) = P_{\text{QAWM}}(\text{legal} \mid s, g) \times P_{\text{QAWM}}(\text{return-in-k} \mid s, g)$$

**Pseudocode**:
```python
def qawm_greedy_policy(s, target_class, k, qawm_model):
    scores = {}

    for g in generators:
        # Get QAWM predictions (no oracle calls!)
        state_feat = extract_state_features(s)
        gen_idx = generator_to_index(g)

        # Forward pass through QAWM
        outputs = qawm_model(state_feat, gen_idx)

        p_legal = outputs['legal_logits']  # Predicted legality
        p_return = outputs['return_logits']  # Predicted return-in-k

        # Combined score
        scores[g] = p_legal * p_return

    # Pick highest scoring generator
    best_g = max(scores, key=scores.get)

    # Verify legality with oracle (required for execution)
    if not oracle.is_legal(s, best_g):
        # QAWM was wrong, fall back to random-legal
        return random_legal_policy(s, oracle)

    return best_g
```

**Oracle calls per step**:
- QAWM predictions: 0 oracle calls (pure model inference)
- Legality verification: 1 (only for chosen generator)
- **Total**: ~1-4 per step (if QAWM prediction is wrong, try others)

**Expected performance**:
- Better than random-legal (has structural guidance)
- Worse than oracle-greedy (uses learned predictions, not ground truth)
- **If close to oracle-greedy**: Paper 3 is already 50% done!

**Critical insight**:
> "QAWM-Greedy uses learned structure to reduce oracle calls from ~10 (oracle-greedy) to ~1-2 (QAWM-greedy) per step."

**Purpose**:
- Proves QAWM enables efficient control
- No learning needed yet - just structural prediction
- If this alone works well, it validates the entire thesis

---

### Baseline 4: RML Policy (Learning, Lightweight)

**Description**: Learn generator preferences using **bandit REINFORCE** over QAWM features.

**Architecture**:
- Input: State features (same 128-dim as QAWM)
- Output: Logits over 4 generators
- Hidden: 256 → 256 MLP (same as QAWM)

**Update rule**: REINFORCE (policy gradient)

**Pseudocode**:
```python
class RMLPolicy:
    def __init__(self):
        self.model = MLPClassifier(hidden=(256, 256), output=4)

    def select_generator(self, s, qawm_model):
        # Get state features
        state_feat = extract_state_features(s)

        # Get QAWM structural hints (optional augmentation)
        qawm_scores = []
        for g in generators:
            outputs = qawm_model(state_feat, generator_to_index(g))
            qawm_scores.append(outputs['legal_logits'] * outputs['return_logits'])

        # Policy logits
        logits = self.model.forward(state_feat)

        # Combine with QAWM hints (weighted)
        combined_logits = logits + alpha * np.array(qawm_scores)

        # Sample from distribution
        probs = softmax(combined_logits)
        g_idx = np.random.choice(4, p=probs)

        return generators[g_idx]

    def update(self, episode_trajectory, success):
        # REINFORCE update
        # Gradient: ∇_θ log π_θ(g|s) * R
        # R = 1 if success, 0 if failure

        states, actions = zip(*episode_trajectory)

        for s, g in zip(states, actions):
            state_feat = extract_state_features(s)
            g_idx = generator_to_index(g)

            # Compute gradient
            logits = self.model.forward(state_feat)
            log_prob = log_softmax(logits)[g_idx]

            # Update (simplified for bandit case)
            grad = log_prob * (success - baseline)
            self.model.update(grad)
```

**Training protocol**:
- Episodes: 1,000 episodes
- Batch size: 32 episodes per update
- Baseline: Moving average of success rate
- Exploration: ε-greedy with ε = 0.1

**Oracle calls per step**: ~1-2 (same as QAWM-greedy after training)

**Expected performance**:
- Similar or slightly better than QAWM-greedy
- **If marginal improvement**: Still valuable (shows learning adapts to QAWM predictions)
- **If large improvement**: Strong result (meta-learning over structure)

**Purpose**: Show that policy learning **over structural predictions** outperforms fixed heuristics

---

## Metrics (Report Only These Three)

### Metric 1: Success Rate

$$\text{Success Rate} = \frac{\text{\# episodes reaching target in } \le k \text{ steps}}{\text{total episodes}}$$

**Report**:
- Random-Legal: ~X%
- Oracle-Greedy: ~Y% (near 100%)
- QAWM-Greedy: ~Z% (between X and Y)
- RML: ~W% (≥ Z)

### Metric 2: Oracle Calls to Success (PRIMARY)

$$\text{Oracle Efficiency} = \frac{\text{average oracle calls per successful episode}}{\text{horizon } k}$$

**Report**:
- Random-Legal: ~40 calls (4 per step × 10 steps)
- Oracle-Greedy: ~100 calls (10 per step × 10 steps, expensive)
- **QAWM-Greedy: ~15 calls** (1.5 per step × 10 steps) ← **KEY RESULT**
- RML: ~12 calls (1.2 per step × 10 steps, if learning helps)

**Critical comparison**:
> "QAWM-Greedy achieves 85% success rate with 7× fewer oracle calls than Oracle-Greedy."

**This sentence proves the thesis**.

### Metric 3: Generalization (Train Caps30, Test Caps50)

Same as Paper 2:
- Train RML on Caps(30,30)
- Test on Caps(50,50) without retraining
- Report success rate + oracle efficiency

**Expected**: Modest degradation (similar to Paper 2's 0.836 → 0.816)

---

## Experimental Protocol (Reproducible)

### Setup
- Oracle: Caps(30,30), q_def="none"
- Generators: {σ, μ, λ₂, ν}
- Target class: Diagonal {(b,b)}
- Horizon: k = 10
- Episodes: 100 test episodes per baseline

### Evaluation
- Metrics: Success rate, oracle calls, generalization
- Random seed: 42 (reproducible)
- No early stopping (run all k steps unless target reached)

### Ablations (Optional)
1. QAWM-Greedy **without** return-in-k head (legality only)
2. RML **without** QAWM hints (pure policy gradient)
3. Different target classes (not just diagonal)

---

## LaTeX-Ready Content (Results Section Skeleton)

### Section 3: Control via Structural Prediction

```latex
\section{Meta-Policy Learning Over Structural Predictions}

We evaluate whether QAWM's learned topology enables efficient control.
We define a standard task: reaching the diagonal target class
$\mathcal{R}^* = \{(b,b) : 1 \le b \le 30\}$ from random initial states
within $k=10$ steps using generators $\Sigma = \{\sigma, \mu, \lambda_2, \nu\}$.

\subsection{Baselines}

We compare four policies:

\begin{enumerate}
\item \textbf{Random-Legal}: Uniform selection among legal generators (lower bound).
\item \textbf{Oracle-Greedy}: Uses true oracle return-in-$k$ to select generators (upper bound).
\item \textbf{QAWM-Greedy}: Scores generators using QAWM's learned structural predictions.
\item \textbf{RML}: Lightweight policy gradient learning over QAWM features.
\end{enumerate}

\subsection{Primary Result: Oracle Efficiency}

Table~\ref{tab:oracle_efficiency} shows that QAWM-Greedy achieves
comparable success rates to Oracle-Greedy while requiring \textbf{7× fewer
oracle calls} ($\sim$15 vs $\sim$100 per episode). This demonstrates that
learned structural predictions enable efficient control without exhaustive
ground-truth queries.

\subsection{Meta-Learning (RML)}

RML further improves oracle efficiency to $\sim$12 calls per episode,
demonstrating that policy learning over structural predictions outperforms
fixed heuristics (Table~\ref{tab:rml_results}).
```

### Table: Oracle Efficiency

```latex
\begin{table}[h]
\centering
\caption{Oracle efficiency: average calls per successful episode.}
\label{tab:oracle_efficiency}
\begin{tabular}{lccc}
\toprule
Policy & Success Rate & Oracle Calls & Relative \\
\midrule
Random-Legal & 35\% & 40 & 1.0× \\
Oracle-Greedy & 95\% & 100 & 2.5× \\
\textbf{QAWM-Greedy} & \textbf{85\%} & \textbf{15} & \textbf{0.375×} \\
RML & 88\% & 12 & 0.3× \\
\bottomrule
\end{tabular}
\end{table}
```

**Note**: Numbers are illustrative placeholders - will be filled after implementation.

---

## Key Claims (Paper 3)

### Claim 1: Structural Prediction Enables Control
**Evidence**: QAWM-Greedy achieves 85% success with 7× fewer oracle calls than Oracle-Greedy.

### Claim 2: Meta-Learning Improves Efficiency
**Evidence**: RML reduces calls from ~15 (QAWM-Greedy) to ~12, showing adaptive improvement.

### Claim 3: Generalizes Across Manifolds
**Evidence**: RML trained on Caps30 transfers to Caps50 with minimal degradation.

### Claim 4: This Is Not Reinforcement Learning
**Separation**:
- RL: Learns $Q(s,a)$ to maximize cumulative reward
- RML: Queries structural predicates to satisfy reachability constraints
- RML never predicts next states, only selects generators based on learned impossibility

---

## Implementation Checklist

- [ ] Implement Random-Legal baseline
- [ ] Implement Oracle-Greedy baseline
- [ ] Implement QAWM-Greedy (no learning)
- [ ] Run 100 test episodes for each baseline
- [ ] **If QAWM-Greedy ≈ Oracle-Greedy**: Paper 3 is 50% done!
- [ ] Implement RML policy (bandit REINFORCE)
- [ ] Train RML for 1,000 episodes
- [ ] Evaluate generalization (Caps30 → Caps50)
- [ ] Generate plots (success rate, oracle calls, generalization)
- [ ] Write Results section

---

## Success Criteria

### Minimal Success (Publishable)
- QAWM-Greedy success rate > Random-Legal + 20%
- QAWM-Greedy oracle calls < Oracle-Greedy / 2

### Strong Success (High-Impact)
- QAWM-Greedy success rate ≥ 80%
- QAWM-Greedy oracle calls ≤ Oracle-Greedy / 5
- RML improves over QAWM-Greedy by ≥ 5%

### Weak (Still Defensible)
- QAWM-Greedy improves over Random-Legal
- RML shows any improvement over QAWM-Greedy
- Generalization holds (Caps30 → Caps50)

---

## Timeline Estimate

**Phase 1** (Baselines, no learning): 2-3 hours
- Random-Legal: 30 min
- Oracle-Greedy: 1 hour
- QAWM-Greedy: 1 hour
- Evaluation: 30 min

**Phase 2** (RML learning): 2-4 hours
- RML policy class: 1 hour
- Training loop: 1 hour
- Evaluation: 1 hour
- Debugging/tuning: 1 hour

**Phase 3** (Results & writing): 1-2 hours
- Generate plots: 30 min
- Write Results section: 1 hour

**Total**: 5-9 hours for complete Paper 3 implementation + results.

---

## Critical Notes

### What Makes This NOT RL

**Key distinctions**:
1. **No value function**: RML doesn't learn $V(s)$ or $Q(s,a)$
2. **No next-state prediction**: RML queries structural predicates, not dynamics
3. **Binary reward**: Success/failure, not cumulative discounted returns
4. **Constraint satisfaction**: Reachability, not optimization

**Framing**:
> "RML is meta-learning over a structural oracle surrogate, not reinforcement learning over a Markov Decision Process."

### Why QAWM-Greedy Is the Key Result

**If QAWM-Greedy alone works well**:
- Proves learned structure enables control
- No learning needed (just query QAWM)
- Oracle efficiency gains immediate
- RML becomes "polish", not core thesis

**Strategy**: Lead with QAWM-Greedy results, RML as "further improvement"

---

## Next Step

**Implement baselines in order**:
1. Random-Legal (30 min)
2. Oracle-Greedy (1 hour)
3. **QAWM-Greedy (1 hour) ← Critical test**

After QAWM-Greedy results:
- **If strong**: Proceed to RML
- **If weak**: Debug QAWM predictions or adjust scoring function

---

**Design Status**: ✅ LOCKED
**Ready for**: Implementation (proceed to Option 2)

---

**End of Experimental Design**
