#!/usr/bin/env python3
"""
QA Symbolic Conjecture Miner v2
Extracts mathematical conjectures from GNN embeddings using clustering and pattern matching
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import sympy as sp
import logging
import json
from pathlib import Path
from collections import Counter
import time
import torch_geometric
import torch.serialization
from torch_geometric.data.storage import GlobalStorage

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Allow torch.load to deserialize torch_geometric graph objects under the new weights_only default.
torch.serialization.add_safe_globals([
    torch_geometric.data.data.Data,
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    GlobalStorage,
])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SymbolicMiner")


class QASymbolicMiner:
    """
    Mines symbolic conjectures from QA tuple embeddings
    """

    def __init__(self, embeddings_path, graph_path, dataset_path):
        self.embeddings_path = embeddings_path
        self.graph_path = graph_path
        self.dataset_path = dataset_path

        self.embeddings = None
        self.graph = None
        self.df = None
        self.clusters = None
        self.conjectures = []

    def load_data(self):
        """Load all required data"""
        logger.info("Loading data...")

        # Load embeddings
        logger.info(f"Loading embeddings from {self.embeddings_path}")
        self.embeddings = np.load(self.embeddings_path)
        logger.info(f"✓ Embeddings shape: {self.embeddings.shape}")

        # Load graph
        logger.info(f"Loading graph from {self.graph_path}")
        self.graph = torch.load(self.graph_path)
        logger.info(f"✓ Graph: {self.graph.num_nodes} nodes")

        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"✓ Dataset: {len(self.df)} tuples")

    def cluster_embeddings(self, method='dbscan', **kwargs):
        """
        Cluster embeddings to find groups of similar tuples
        """
        logger.info(f"Clustering embeddings using {method.upper()}...")

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(self.embeddings)

        if method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)

            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = clusterer.fit_predict(embeddings_norm)

            n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            n_noise = list(self.clusters).count(-1)

            logger.info(f"✓ Found {n_clusters} clusters")
            logger.info(f"  Noise points: {n_noise}")

        elif method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 10)

            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = clusterer.fit_predict(embeddings_norm)

            logger.info(f"✓ Created {n_clusters} clusters")

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Add cluster labels to dataframe
        self.df['cluster'] = self.clusters

        # Show cluster distribution
        cluster_counts = Counter(self.clusters)
        logger.info("Cluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            if cluster_id != -1:  # Skip noise
                logger.info(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} tuples")

        return self.clusters

    def extract_cluster_patterns(self, cluster_id):
        """
        Extract mathematical patterns from a single cluster
        """
        # Get tuples in this cluster
        cluster_mask = self.df['cluster'] == cluster_id
        cluster_tuples = self.df[cluster_mask]

        if len(cluster_tuples) < 3:
            return []

        patterns = []

        # Pattern 1: Check modular relationships
        mod_patterns = self._check_modular_patterns(cluster_tuples)
        patterns.extend(mod_patterns)

        # Pattern 2: Check geometric relationships
        geom_patterns = self._check_geometric_patterns(cluster_tuples)
        patterns.extend(geom_patterns)

        # Pattern 3: Check algebraic identities
        alg_patterns = self._check_algebraic_patterns(cluster_tuples)
        patterns.extend(alg_patterns)

        return patterns

    def _check_modular_patterns(self, tuples):
        """Check for modular arithmetic patterns"""
        patterns = []

        # Check if all tuples share same mod-24 residue pattern
        for mod_col in ['b_mod24', 'e_mod24', 'd_mod24', 'a_mod24']:
            unique_vals = tuples[mod_col].unique()
            if len(unique_vals) == 1:
                patterns.append({
                    'type': 'modular_constant',
                    'column': mod_col,
                    'value': unique_vals[0],
                    'count': len(tuples)
                })

        # Check for modular relationships
        if 'a_mod24' in tuples.columns and 'b_mod24' in tuples.columns and 'e_mod24' in tuples.columns:
            # Check if a_mod24 = b_mod24 + 2*e_mod24 (mod 24)
            expected = (tuples['b_mod24'] + 2 * tuples['e_mod24']) % 24
            if (tuples['a_mod24'] == expected).all():
                patterns.append({
                    'type': 'modular_identity',
                    'formula': 'a ≡ b + 2e (mod 24)',
                    'count': len(tuples)
                })

        return patterns

    def _check_geometric_patterns(self, tuples):
        """Check for geometric patterns (Pythagorean, Eisenstein)"""
        patterns = []

        if 'geometry' in tuples.columns:
            geom_types = tuples['geometry'].value_counts()

            for geom_type, count in geom_types.items():
                if count / len(tuples) > 0.8:  # 80% threshold
                    patterns.append({
                        'type': 'geometric_class',
                        'geometry': geom_type,
                        'proportion': count / len(tuples),
                        'count': len(tuples)
                    })

        return patterns

    def _check_algebraic_patterns(self, tuples):
        """Check for algebraic identities"""
        patterns = []

        # Check basic QA identity: d = b + e
        if 'd' in tuples.columns and 'b' in tuples.columns and 'e' in tuples.columns:
            if (tuples['d'] == tuples['b'] + tuples['e']).all():
                patterns.append({
                    'type': 'algebraic_identity',
                    'formula': 'd = b + e',
                    'count': len(tuples)
                })

        # Check: a = b + 2e
        if 'a' in tuples.columns:
            expected_a = tuples['b'] + 2 * tuples['e']
            if (tuples['a'] == expected_a).all():
                patterns.append({
                    'type': 'algebraic_identity',
                    'formula': 'a = b + 2e',
                    'count': len(tuples)
                })

        return patterns

    def generate_conjectures(self):
        """
        Generate conjectures from all clusters
        """
        logger.info("Generating conjectures from clusters...")

        unique_clusters = [c for c in set(self.clusters) if c != -1]

        for cluster_id in unique_clusters:
            logger.info(f"Processing cluster {cluster_id}...")

            patterns = self.extract_cluster_patterns(cluster_id)

            if patterns:
                # Create conjecture from patterns
                conjecture = {
                    'cluster_id': cluster_id,
                    'patterns': patterns,
                    'tuple_count': int((self.clusters == cluster_id).sum()),
                    'rank_score': self._calculate_rank_score(patterns)
                }

                self.conjectures.append(conjecture)
                logger.info(f"  Found {len(patterns)} patterns")

        logger.info(f"✓ Generated {len(self.conjectures)} conjectures")

        # Sort by rank score
        self.conjectures.sort(key=lambda x: x['rank_score'], reverse=True)

        return self.conjectures

    def _calculate_rank_score(self, patterns):
        """
        Calculate importance rank for a conjecture
        """
        score = 0

        for pattern in patterns:
            # Weight by pattern type
            if pattern['type'] == 'modular_identity':
                score += 10
            elif pattern['type'] == 'algebraic_identity':
                score += 8
            elif pattern['type'] == 'geometric_class':
                score += pattern.get('proportion', 0) * 5
            elif pattern['type'] == 'modular_constant':
                score += 3

            # Weight by tuple count (more tuples = more significant)
            score += np.log(pattern.get('count', 1))

        return score

    def format_conjecture(self, conjecture):
        """
        Format conjecture as human-readable string
        """
        lines = []
        lines.append(f"Conjecture (Cluster {conjecture['cluster_id']}):")
        lines.append(f"  Rank Score: {conjecture['rank_score']:.2f}")
        lines.append(f"  Tuple Count: {conjecture['tuple_count']}")
        lines.append("  Patterns:")

        for pattern in conjecture['patterns']:
            if pattern['type'] == 'modular_identity':
                lines.append(f"    - {pattern['formula']}")
            elif pattern['type'] == 'algebraic_identity':
                lines.append(f"    - {pattern['formula']}")
            elif pattern['type'] == 'geometric_class':
                lines.append(f"    - Geometry: {pattern['geometry']} ({pattern['proportion']:.0%})")
            elif pattern['type'] == 'modular_constant':
                lines.append(f"    - {pattern['column']} = {pattern['value']} (constant)")

        return "\n".join(lines)

    def mine(self, clustering_method='dbscan', output_path='conjectures.json'):
        """
        Main mining pipeline
        """
        logger.info("="*60)
        logger.info("QA SYMBOLIC CONJECTURE MINER - Starting")
        logger.info("="*60)

        start_time = time.time()

        # Step 1: Load data
        self.load_data()

        # Step 2: Cluster embeddings
        self.cluster_embeddings(method=clustering_method)

        # Step 3: Generate conjectures
        self.generate_conjectures()

        # Statistics
        total_time = time.time() - start_time

        logger.info("="*60)
        logger.info("CONJECTURE MINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total conjectures: {len(self.conjectures)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("="*60)

        # Display top 10 conjectures
        logger.info("\nTop 10 Conjectures:")
        logger.info("-"*60)
        for i, conj in enumerate(self.conjectures[:10], 1):
            logger.info(f"\n{i}. {self.format_conjecture(conj)}")

        # Save to JSON
        if output_path:
            logger.info(f"\nSaving conjectures to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(self.conjectures, f, indent=2, cls=NpEncoder)
            logger.info(f"✓ Conjectures saved")

        return self.conjectures


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Mine symbolic conjectures from QA embeddings')
    parser.add_argument('--embeddings', default='qa_embeddings.pt',
                       help='Path to embeddings file')
    parser.add_argument('--graph', default='qa_graph.pt',
                       help='Path to graph file')
    parser.add_argument('--dataset', default='qa_10000_balanced_tuples.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--output', default='conjectures.json',
                       help='Output path for conjectures')
    parser.add_argument('--clustering', default='dbscan', choices=['dbscan', 'kmeans'],
                       help='Clustering method')

    args = parser.parse_args()

    try:
        miner = QASymbolicMiner(args.embeddings, args.graph, args.dataset)
        conjectures = miner.mine(clustering_method=args.clustering, output_path=args.output)

        logger.info("✓ SUCCESS: Mining completed")
        return 0

    except Exception as e:
        logger.error(f"✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
