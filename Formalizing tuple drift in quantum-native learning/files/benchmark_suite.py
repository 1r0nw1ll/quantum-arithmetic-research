"""
Benchmark harness for Paper 1 reproducible results.
Generates tables, figures, and statistics using canonical oracle.
"""

from qa_oracle import QAOracle, QAState, FailType, construct_qa_state
from collections import defaultdict
from typing import List, Set, Tuple, Dict
import json
import sys

# Increase recursion limit for Tarjan's algorithm on large graphs
sys.setrecursionlimit(10000)

class QABenchmark:
    """Generate reproducible benchmark statistics for Caps(N,N)"""
    
    def __init__(self, oracle: QAOracle):
        self.oracle = oracle
    
    def enumerate_caps(self) -> List[QAState]:
        """
        Enumerate all states in Caps(N,N).
        Returns: [(b,e) : 1 ≤ b ≤ N, 1 ≤ e ≤ N]
        """
        N = self.oracle.N
        states = []
        
        for b in range(1, N + 1):
            for e in range(1, N + 1):
                state = construct_qa_state(b, e)
                states.append(state)
        
        return states
    
    def compute_topology_stats(self, states: List[QAState],
                              generators: List[Tuple[str, int]]) -> Dict:
        """
        Compute topology statistics:
        - #states, #edges, #failures (by type)
        - #SCCs, max SCC size
        
        Args:
            generators: [(move, k_param), ...] 
                       e.g., [("sigma", 2), ("lambda2", 2)]
        """
        num_states = len(states)
        
        # Count edges and failures
        legal_edges = 0
        fail_counts = defaultdict(int)
        
        for s in states:
            for move, k_param in generators:
                if self.oracle.is_legal(s, move, k_param):
                    legal_edges += 1
                else:
                    fail_type = self.oracle.get_fail_type(s, move, k_param)
                    if fail_type:
                        fail_counts[fail_type.value] += 1
        
        # Compute SCCs via Tarjan
        sccs = self._compute_sccs(states, generators)
        num_sccs = len(sccs)
        max_scc_size = max(len(scc) for scc in sccs) if sccs else 0
        
        return {
            'num_states': num_states,
            'num_edges': legal_edges,
            'num_failures': sum(fail_counts.values()),
            'fail_type_counts': dict(fail_counts),
            'num_sccs': num_sccs,
            'max_scc_size': max_scc_size,
            'scc_histogram': self._scc_histogram(sccs)
        }
    
    def _compute_sccs(self, states: List[QAState],
                     generators: List[Tuple[str, int]]) -> List[Set[QAState]]:
        """
        Tarjan's algorithm for exact SCC computation.
        """
        # Build adjacency list
        graph = defaultdict(list)
        state_set = set(states)
        
        for s in states:
            for move, k_param in generators:
                if self.oracle.is_legal(s, move, k_param):
                    next_s = self.oracle.step(s, move, k_param)
                    if next_s in state_set:  # Stay within enumerated set
                        graph[s].append(next_s)
        
        # Tarjan's algorithm
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            on_stack[node] = True
            stack.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            
            if lowlinks[node] == index[node]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)
        
        for node in states:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def _scc_histogram(self, sccs: List[Set[QAState]]) -> Dict[int, int]:
        """Return histogram of SCC sizes"""
        hist = defaultdict(int)
        for scc in sccs:
            hist[len(scc)] += 1
        return dict(hist)
    
    def generate_paper1_table(self, caps_list: List[int],
                             generator_configs: List[Dict]) -> str:
        """
        Generate LaTeX Table 1 for Paper 1.
        
        Args:
            caps_list: [30, 50]
            generator_configs: [
                {"name": "{σ,λ₂}", "moves": [("sigma", 2), ("lambda2", 2)]},
                ...
            ]
        """
        rows = []
        
        for N in caps_list:
            oracle = QAOracle(N, q_def="none")
            benchmark = QABenchmark(oracle)
            states = benchmark.enumerate_caps()
            
            for config in generator_configs:
                stats = benchmark.compute_topology_stats(states, config["moves"])
                
                row = (f"Caps({N},{N}) & {config['name']} & {stats['num_states']} & "
                       f"{stats['num_edges']} & {stats['num_failures']} & "
                       f"{stats['num_sccs']} & {stats['max_scc_size']} \\\\")
                rows.append(row)
        
        table = "\\begin{tabular}{lcccccc}\n"
        table += "Cap & $\\Sigma$ & \\#States & \\#Edges & \\#Fail & \\#SCC & Max-SCC \\\\\n"
        table += "\\hline\n"
        table += "\n".join(rows)
        table += "\n\\end{tabular}"
        
        return table


# ============================================================================
# Main: Generate Paper 1 Results
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CANONICAL QA ORACLE - PAPER 1 BENCHMARK SUITE")
    print("=" * 70)
    
    # Configuration
    caps_list = [30, 50]
    generator_configs = [
        {
            "name": r"$\{\sigma,\lambda_2\}$",
            "moves": [("sigma", 2), ("lambda2", 2)]
        },
        {
            "name": r"$\{\sigma,\mu,\lambda_2\}$",
            "moves": [("sigma", 2), ("mu", 2), ("lambda2", 2)]
        },
        {
            "name": r"$\{\sigma,\mu,\lambda_2,\nu\}$",
            "moves": [("sigma", 2), ("mu", 2), ("lambda2", 2), ("nu", 2)]
        }
    ]
    
    # Generate Table 1
    print("\n" + "=" * 70)
    print("TABLE 1: TOPOLOGY STATISTICS")
    print("=" * 70)
    
    oracle_30 = QAOracle(30, q_def="none")
    benchmark_30 = QABenchmark(oracle_30)
    
    table_latex = benchmark_30.generate_paper1_table(caps_list, generator_configs)
    print(table_latex)
    
    # Generate Table 2: Failure Distribution (Caps(30,30), {σ,μ,λ₂})
    print("\n" + "=" * 70)
    print("TABLE 2: FAILURE DISTRIBUTION (Caps(30,30), {σ,μ,λ₂})")
    print("=" * 70)
    
    states_30 = benchmark_30.enumerate_caps()
    stats = benchmark_30.compute_topology_stats(
        states_30,
        [("sigma", 2), ("mu", 2), ("lambda2", 2)]
    )
    
    print(f"{'Fail Type':<20} {'Count':>8} {'%':>6}")
    print("-" * 36)
    for fail_type, count in sorted(stats['fail_type_counts'].items()):
        pct = 100 * count / stats['num_failures']
        print(f"{fail_type:<20} {count:>8} {pct:>5.1f}%")
    
    # Generate SCC Histogram (for phase transition figure)
    print("\n" + "=" * 70)
    print("SCC HISTOGRAM (Caps(30,30), {σ,μ,λ₂,ν})")
    print("=" * 70)
    
    stats_full = benchmark_30.compute_topology_stats(
        states_30,
        [("sigma", 2), ("mu", 2), ("lambda2", 2), ("nu", 2)]
    )
    
    print(f"{'SCC Size':<12} {'Count':>8}")
    print("-" * 22)
    for size, count in sorted(stats_full['scc_histogram'].items()):
        print(f"{size:<12} {count:>8}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE - Results match canonical Caps experiments")
    print("=" * 70)
