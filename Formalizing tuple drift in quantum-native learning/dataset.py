"""
QAWM Dataset Generation
Collects training data from QA oracle via random exploration.

Modified to use numpy arrays (no PyTorch dependency).
"""

import numpy as np
from qa_oracle import QAOracle, QAState, construct_qa_state, FailType
from qawm import extract_state_features, generator_to_index
import random
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# Dataset Generation
# =============================================================================

class QADatasetGenerator:
    """
    Generate training data from oracle via fixed-budget exploration.
    """

    def __init__(self, oracle: QAOracle, generators: List[str]):
        """
        Args:
            oracle: QA ground truth oracle
            generators: list of generator names to probe
        """
        self.oracle = oracle
        self.generators = generators
        self.N = oracle.N

    def sample_random_state(self) -> QAState:
        """Sample a random state from Caps(N,N)"""
        b = random.randint(1, self.N)
        e = random.randint(1, self.N)
        return construct_qa_state(b, e)

    def probe_transition(self, state: QAState, gen: str) -> Dict:
        """
        Query oracle for single transition.

        Returns:
            record: dict with all transition information
        """
        is_legal = self.oracle.is_legal(state, gen)

        if is_legal:
            next_state = self.oracle.step(state, gen)
            fail_type = None
        else:
            next_state = None
            fail_type_enum = self.oracle.get_fail_type(state, gen)
            fail_type = fail_type_enum.value if fail_type_enum else None

        return {
            'state': state,
            'generator': gen,
            'legal': is_legal,
            'fail_type': fail_type,
            'next_state': next_state
        }

    def generate_dataset(self, budget: int,
                        return_in_k_budget: int = 100) -> 'QATransitionDataset':
        """
        Generate dataset via random exploration.

        Args:
            budget: number of (state, generator) pairs to probe
            return_in_k_budget: number of return-in-k queries (expensive)

        Returns:
            dataset: QATransitionDataset
        """
        records = []

        print(f"Generating dataset with budget {budget}...")

        # Random exploration
        for i in range(budget):
            state = self.sample_random_state()
            gen = random.choice(self.generators)

            record = self.probe_transition(state, gen)
            records.append(record)

            if (i + 1) % 1000 == 0:
                print(f"  Collected {i+1}/{budget} transitions")

        # Compute return-in-k labels for subset (expensive)
        print(f"Computing return-in-k labels for {return_in_k_budget} samples...")
        return_in_k_labels = self._compute_return_in_k_subset(
            records,
            return_in_k_budget
        )

        # Build dataset
        dataset = QATransitionDataset(records, return_in_k_labels)

        print(f"✅ Dataset generated: {len(dataset)} samples")
        print(f"   Legal: {sum(1 for r in records if r['legal'])}")
        print(f"   Illegal: {sum(1 for r in records if not r['legal'])}")
        print(f"   Return-in-k labels: {len(return_in_k_labels)}")

        return dataset

    def _compute_return_in_k_subset(self, records: List[Dict],
                                    budget: int) -> Dict[int, bool]:
        """
        Compute return-in-k for a subset of states.

        This is expensive, so we only label a subset.
        """
        # Sample indices
        indices = random.sample(range(len(records)), min(budget, len(records)))

        # Define simple target class (e.g., states with b=e diagonal)
        target_class = set()
        for b in range(1, self.N + 1):
            target_class.add(construct_qa_state(b, b))

        k = 10  # Return-in-k horizon
        gen_list = [(g, 2) for g in self.generators]

        labels = {}
        for idx in indices:
            state = records[idx]['state']
            reachable = self.oracle.return_in_k(state, target_class, k, gen_list)
            labels[idx] = reachable

        return labels


# =============================================================================
# Dataset Class (Numpy Implementation)
# =============================================================================

class QATransitionDataset:
    """
    Dataset for QAWM training (numpy arrays, no PyTorch).
    """

    def __init__(self, records: List[Dict], return_in_k_labels: Dict[int, bool]):
        """
        Args:
            records: list of transition records from oracle
            return_in_k_labels: dict mapping record index -> return-in-k label
        """
        self.records = records
        self.return_in_k_labels = return_in_k_labels

        # Build fail type mapping
        self.fail_type_to_idx = {
            'OUT_OF_BOUNDS': 0,
            'PARITY': 1,
            'PHASE_VIOLATION': 2,
            'INVARIANT': 3,
            'REDUCTION': 4
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
        """
        Get single training example.

        Returns:
            state_features: (128,) array
            gen_idx: scalar int
            labels: dict of label arrays
        """
        record = self.records[idx]

        # Extract features
        state_features = extract_state_features(record['state'])
        gen_idx = generator_to_index(record['generator'])

        # Extract labels
        legal_label = 1 if record['legal'] else 0

        # Fail type (only meaningful for illegal moves)
        if record['fail_type'] is not None:
            fail_type_idx = self.fail_type_to_idx.get(record['fail_type'], 0)
        else:
            fail_type_idx = 0  # Dummy (will be masked in loss)

        # Return-in-k (only available for subset)
        if idx in self.return_in_k_labels:
            return_label = 1 if self.return_in_k_labels[idx] else 0
        else:
            return_label = -1  # Missing label

        labels = {
            'legal': legal_label,
            'fail_type': fail_type_idx,
            'return': return_label,
            'illegal_mask': not record['legal']
        }

        return state_features, gen_idx, labels

    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Get a batch of samples.

        Args:
            indices: list of sample indices

        Returns:
            state_features: (batch_size, 128)
            gen_indices: (batch_size,)
            labels: dict of (batch_size,) arrays
        """
        state_features_list = []
        gen_indices_list = []
        labels_list = {
            'legal': [],
            'fail_type': [],
            'return': [],
            'illegal_mask': []
        }

        for idx in indices:
            state_feat, gen_idx, labels = self[idx]
            state_features_list.append(state_feat)
            gen_indices_list.append(gen_idx)

            for key in labels_list:
                labels_list[key].append(labels[key])

        # Stack into arrays
        state_features = np.stack(state_features_list)
        gen_indices = np.array(gen_indices_list, dtype=np.int64)

        labels_batch = {
            'legal': np.array(labels_list['legal'], dtype=np.int64),
            'fail_type': np.array(labels_list['fail_type'], dtype=np.int64),
            'return': np.array(labels_list['return'], dtype=np.int64),
            'illegal_mask': np.array(labels_list['illegal_mask'], dtype=np.bool_)
        }

        return state_features, gen_indices, labels_batch

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate over dataset in batches.

        Args:
            batch_size: number of samples per batch
            shuffle: whether to shuffle before iterating

        Yields:
            (state_features, gen_indices, labels) tuples
        """
        indices = list(range(len(self)))

        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.get_batch(batch_indices)


# =============================================================================
# Dataset Statistics
# =============================================================================

def analyze_dataset(dataset: QATransitionDataset):
    """Print dataset statistics"""
    legal_count = sum(1 for r in dataset.records if r['legal'])
    illegal_count = len(dataset) - legal_count

    fail_type_counts = defaultdict(int)
    for r in dataset.records:
        if r['fail_type']:
            fail_type_counts[r['fail_type']] += 1

    return_true = sum(1 for label in dataset.return_in_k_labels.values() if label)
    return_false = len(dataset.return_in_k_labels) - return_true

    print("\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Legal: {legal_count} ({100*legal_count/len(dataset):.1f}%)")
    print(f"  Illegal: {illegal_count} ({100*illegal_count/len(dataset):.1f}%)")
    print(f"\n  Failure type distribution:")
    for fail_type, count in sorted(fail_type_counts.items()):
        print(f"    {fail_type}: {count}")
    print(f"\n  Return-in-k labels:")
    print(f"    Labeled: {len(dataset.return_in_k_labels)}")
    print(f"    Reachable: {return_true}")
    print(f"    Unreachable: {return_false}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from qa_oracle import QAOracle

    # Initialize oracle
    oracle = QAOracle(N=30, q_def="none")
    generators = ['sigma', 'mu', 'lambda2', 'nu']

    # Generate dataset
    gen = QADatasetGenerator(oracle, generators)
    dataset = gen.generate_dataset(budget=1000, return_in_k_budget=100)

    # Analyze
    analyze_dataset(dataset)

    # Test batching
    print("\nBatch test:")
    for batch_idx, (state_features, gen_indices, labels) in enumerate(dataset.iterate_batches(32)):
        print(f"  Batch {batch_idx}: state_features {state_features.shape}, gen_indices {gen_indices.shape}")
        if batch_idx >= 2:
            break

    print("\n✅ Dataset generation complete")
