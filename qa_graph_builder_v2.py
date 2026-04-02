#!/usr/bin/env python3
"""
QA Graph Builder v2 - Instrumented Version
Fixes the October 8th silent execution problem with real-time monitoring
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GraphBuilder")


class QAGraphBuilder:
    """
    Instrumented QA tuple graph builder with progress tracking
    """

    def __init__(self, csv_path, checkpoint_dir="./checkpoints", edge_mode="pipeline"):
        self.csv_path = csv_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.edge_mode = edge_mode
        self.df = None
        self.edge_stats = {
            'harmonic': 0,
            'modular': 0,
            'geometric': 0,
            'sigma': 0,
            'mu': 0,
            'lambda2': 0,
            'nu': 0,
        }

    def _build_tuple_index(self):
        tuple_to_idx = {}
        for idx, row in self.df.iterrows():
            key = (row['b'], row['e'])
            tuple_to_idx[key] = idx
        return tuple_to_idx

    def load_dataset(self):
        """Load and validate dataset"""
        logger.info(f"Loading dataset from {self.csv_path}")
        start_time = time.time()

        self.df = pd.read_csv(self.csv_path)
        load_time = time.time() - start_time

        logger.info(f"✓ Loaded {len(self.df)} tuples in {load_time:.2f}s")

        # Validate required columns
        required_cols = ['b', 'e', 'd', 'a', 'b_mod24', 'e_mod24', 'd_mod24', 'a_mod24']
        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"✓ Validated columns: {required_cols}")

        return self.df

    def create_node_features(self):
        """Extract node feature tensor"""
        logger.info("Creating node features...")

        feature_cols = ['b', 'e', 'd', 'a', 'b_mod24', 'e_mod24', 'd_mod24', 'a_mod24']
        x = torch.tensor(self.df[feature_cols].values, dtype=torch.float)

        logger.info(f"✓ Node features: shape={x.shape}, dtype={x.dtype}")
        return x

    def build_harmonic_edges(self):
        """Build edges based on harmonic transitions (b+1, e) or (b, e+1)"""
        logger.info("Pass 1/3: Building harmonic transition edges")
        edges = []

        # Create lookup dictionaries for fast matching
        tuple_to_idx = self._build_tuple_index()

        # Find harmonic transitions with progress bar
        for idx, row in tqdm(self.df.iterrows(),
                           total=len(self.df),
                           desc="Harmonic edges",
                           unit="nodes"):
            b, e = row['b'], row['e']

            # Check for (b+1, e) transition
            next_key_1 = (b + 1, e)
            if next_key_1 in tuple_to_idx:
                edges.append([idx, tuple_to_idx[next_key_1]])

            # Check for (b, e+1) transition
            next_key_2 = (b, e + 1)
            if next_key_2 in tuple_to_idx:
                edges.append([idx, tuple_to_idx[next_key_2]])

        self.edge_stats['harmonic'] = len(edges)
        logger.info(f"✓ Found {len(edges)} harmonic edges")

        return edges

    def build_canonical_edges(self):
        """Build edges based on canonical generators sigma, mu, lambda2, nu."""
        logger.info("Building canonical generator edges (sigma, mu, lambda2, nu)")
        edges = []
        edge_types = []
        tuple_to_idx = self._build_tuple_index()

        for idx, row in tqdm(self.df.iterrows(),
                           total=len(self.df),
                           desc="Canonical edges",
                           unit="nodes"):
            b, e = int(row['b']), int(row['e'])

            sigma_key = (b, e + 1)
            if sigma_key in tuple_to_idx:
                edges.append([idx, tuple_to_idx[sigma_key]])
                edge_types.append(1)

            mu_key = (e, b)
            if mu_key in tuple_to_idx:
                edges.append([idx, tuple_to_idx[mu_key]])
                edge_types.append(2)

            lambda2_key = (2 * b, 2 * e)
            if lambda2_key in tuple_to_idx:
                edges.append([idx, tuple_to_idx[lambda2_key]])
                edge_types.append(3)

            if b % 2 == 0 and e % 2 == 0:
                nu_key = (b // 2, e // 2)
                if nu_key in tuple_to_idx:
                    edges.append([idx, tuple_to_idx[nu_key]])
                    edge_types.append(4)

        self.edge_stats['sigma'] = edge_types.count(1)
        self.edge_stats['mu'] = edge_types.count(2)
        self.edge_stats['lambda2'] = edge_types.count(3)
        self.edge_stats['nu'] = edge_types.count(4)
        logger.info(
            "✓ Found %d sigma, %d mu, %d lambda2, %d nu edges",
            self.edge_stats['sigma'],
            self.edge_stats['mu'],
            self.edge_stats['lambda2'],
            self.edge_stats['nu'],
        )

        return edges, edge_types

    def build_modular_edges(self):
        """Build edges based on mod-24 symmetry"""
        logger.info("Pass 2/3: Building modular symmetry edges")
        edges = []

        # Group by a_mod24 for efficient matching
        groups = self.df.groupby('a_mod24').groups

        logger.info(f"Processing {len(groups)} modular groups")

        for mod_val, indices in tqdm(groups.items(),
                                     desc="Modular edges",
                                     unit="groups"):
            indices = list(indices)
            # Connect all nodes with same a_mod24 value
            for i, idx1 in enumerate(indices):
                for idx2 in indices[i+1:]:
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])  # Bidirectional

        self.edge_stats['modular'] = len(edges)
        logger.info(f"✓ Found {len(edges)} modular symmetry edges")

        return edges

    def build_geometric_edges(self):
        """Build edges based on geometric class matching"""
        logger.info("Pass 3/3: Building geometric class edges")
        edges = []

        # Check if geometry column exists
        if 'geometry' not in self.df.columns:
            logger.warning("No 'geometry' column found, skipping geometric edges")
            return edges

        # Group by geometry type
        groups = self.df.groupby('geometry').groups

        logger.info(f"Processing {len(groups)} geometry groups")

        for geom_type, indices in tqdm(groups.items(),
                                      desc="Geometric edges",
                                      unit="groups"):
            indices = list(indices)
            # Sample edges to avoid explosion (connect each node to 5 neighbors)
            for idx1 in indices:
                # Randomly sample up to 5 neighbors
                neighbors = np.random.choice(indices,
                                           min(5, len(indices)),
                                           replace=False)
                for idx2 in neighbors:
                    if idx1 != idx2:
                        edges.append([idx1, idx2])

        self.edge_stats['geometric'] = len(edges)
        logger.info(f"✓ Found {len(edges)} geometric edges")

        return edges

    def build_graph(self, save_path="qa_graph.pt"):
        """
        Main graph building pipeline with full instrumentation
        """
        logger.info("="*60)
        logger.info("QA GRAPH BUILDER V2 - Starting")
        logger.info("="*60)
        logger.info(f"Edge mode: {self.edge_mode}")

        start_time = time.time()

        # Step 1: Load dataset
        self.load_dataset()

        # Step 2: Create node features
        x = self.create_node_features()

        # Step 3: Build edges (mode-specific)
        all_edges = []
        edge_types = []

        if self.edge_mode == "canonical":
            canonical_edges, canonical_types = self.build_canonical_edges()
            all_edges.extend(canonical_edges)
            edge_types.extend(canonical_types)
        else:
            harmonic_edges = self.build_harmonic_edges()
            all_edges.extend(harmonic_edges)
            edge_types.extend([1] * len(harmonic_edges))  # Type 1 = harmonic

            modular_edges = self.build_modular_edges()
            all_edges.extend(modular_edges)
            edge_types.extend([2] * len(modular_edges))   # Type 2 = modular

            geometric_edges = self.build_geometric_edges()
            all_edges.extend(geometric_edges)
            edge_types.extend([3] * len(geometric_edges))  # Type 3 = geometric

        # Convert to tensors
        logger.info("Converting edges to tensor format...")
        if len(all_edges) == 0:
            logger.error("No edges found! Check your data.")
            raise ValueError("Graph has no edges")

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        # Create labels (if harmonic column exists)
        if 'harmonic' in self.df.columns:
            y = torch.tensor(self.df['harmonic'].astype('category').cat.codes.values,
                           dtype=torch.long)
        else:
            y = torch.zeros(len(self.df), dtype=torch.long)

        # Create PyG Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Statistics
        total_time = time.time() - start_time
        avg_degree = graph.num_edges / graph.num_nodes

        logger.info("="*60)
        logger.info("GRAPH CONSTRUCTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Nodes: {graph.num_nodes}")
        logger.info(f"Edges: {graph.num_edges}")
        if self.edge_mode == "canonical":
            logger.info(f"  - Sigma: {self.edge_stats['sigma']}")
            logger.info(f"  - Mu: {self.edge_stats['mu']}")
            logger.info(f"  - Lambda2: {self.edge_stats['lambda2']}")
            logger.info(f"  - Nu: {self.edge_stats['nu']}")
        else:
            logger.info(f"  - Harmonic: {self.edge_stats['harmonic']}")
            logger.info(f"  - Modular: {self.edge_stats['modular']}")
            logger.info(f"  - Geometric: {self.edge_stats['geometric']}")
        logger.info(f"Average degree: {avg_degree:.2f}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("="*60)

        # Save graph
        if save_path:
            logger.info(f"Saving graph to {save_path}")
            torch.save(graph, save_path)
            logger.info(f"✓ Graph saved successfully")

        return graph


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Build QA tuple graph with monitoring')
    parser.add_argument('--input', default='qa_10000_balanced_tuples.csv',
                       help='Input CSV file with QA tuples')
    parser.add_argument('--output', default='qa_graph.pt',
                       help='Output path for graph file')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--edge-mode', default='pipeline', choices=['pipeline', 'canonical'],
                       help='Edge construction mode')

    args = parser.parse_args()

    try:
        builder = QAGraphBuilder(args.input, args.checkpoint_dir, edge_mode=args.edge_mode)
        graph = builder.build_graph(args.output)

        logger.info("✓ SUCCESS: Graph building completed")
        return 0

    except Exception as e:
        logger.error(f"✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
