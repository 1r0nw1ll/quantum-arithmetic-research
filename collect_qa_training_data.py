#!/usr/bin/env python3
"""
T-006: QA Training Dataset Curation
Collect and curate comprehensive QA training dataset for QALM training.

Aggregates:
- QA theorems and proofs from research documents
- Synthetic QA arithmetic examples
- E8 geometry and signal processing data
- Question-answer pairs for QA reasoning
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

class QADatasetCurator:
    """Curator for comprehensive QA training dataset"""

    def __init__(self, output_path: str = "qa_training_dataset.jsonl"):
        self.output_path = Path(output_path)
        self.dataset = []

    def extract_theorems_from_vault(self, vault_path: str = "QAnotes/") -> List[Dict]:
        """
        Extract QA theorems and mathematical content from Obsidian vault.
        Target: 100+ theorem statements with formal proofs
        """
        print(f"[1/4] Extracting theorems from vault: {vault_path}")
        theorems = []
        vault = Path(vault_path)

        if not vault.exists():
            print(f"  ⚠️  Vault not found: {vault_path}")
            return theorems

        # Patterns to detect mathematical content
        theorem_patterns = [
            r'theorem[:\s]+(.+?)(?:\n|$)',
            r'conjecture[:\s]+(.+?)(?:\n|$)',
            r'proposition[:\s]+(.+?)(?:\n|$)',
            r'lemma[:\s]+(.+?)(?:\n|$)',
            r'invariant[:\s]+(.+?)(?:\n|$)',
        ]

        proof_pattern = r'proof[:\s]+(.+?)(?:QED|∎|$)'

        md_files = list(vault.rglob("*.md"))
        print(f"  Found {len(md_files)} markdown files")

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8', errors='ignore')

                # Extract theorems
                for pattern in theorem_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        statement = match.group(1).strip()
                        if len(statement) > 10:  # Filter trivial matches
                            theorems.append({
                                'type': 'theorem',
                                'statement': statement,
                                'source': str(md_file.relative_to(vault)),
                                'domain': 'qa_mathematics'
                            })

                # Look for QA-specific content
                if any(term in content.lower() for term in ['(b,e,d,a)', 'invariant', 'fibonacci', 'e8', 'harmonic']):
                    # Extract QA examples
                    qa_examples = self._extract_qa_examples(content, md_file)
                    theorems.extend(qa_examples)

            except Exception as e:
                print(f"  ⚠️  Error reading {md_file}: {e}")
                continue

        print(f"  ✓ Extracted {len(theorems)} theorems and examples")
        return theorems

    def _extract_qa_examples(self, content: str, source_file: Path) -> List[Dict]:
        """Extract QA tuple examples and invariant relations"""
        examples = []

        # Look for tuple patterns like (3,5,8,13)
        tuple_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        matches = re.finditer(tuple_pattern, content)

        for match in matches:
            b, e, d, a = map(int, match.groups())
            # Verify QA closure
            if b + e == d and e + d == a:
                examples.append({
                    'type': 'qa_example',
                    'tuple': {'b': b, 'e': e, 'd': d, 'a': a},
                    'invariants': {
                        'J': b * d,
                        'K': d * a,
                        'X': e * d
                    },
                    'source': str(source_file.name),
                    'domain': 'qa_tuples'
                })

        return examples

    def generate_synthetic_examples(self, count: int = 10000) -> List[Dict]:
        """
        Generate synthetic QA arithmetic examples.
        Target: 10K+ parameter tuples with computed invariants
        """
        print(f"[2/4] Generating {count} synthetic QA examples")
        examples = []

        # Modular bases as specified in QA_RULES.yaml
        modular_bases = [9, 24, 72, 288]

        for i in range(count):
            # Generate random (b,e) pairs
            b = np.random.randint(1, 100)
            e = np.random.randint(1, 100)

            # QA closure
            d = b + e
            a = b + 2 * e  # or equivalently e + d

            # Compute invariants
            J = b * d
            K = d * a
            X = e * d

            # Check ellipse constraint
            inner_ellipse_valid = (a**2 == d**2 + 2*d*e + e**2)

            example = {
                'type': 'synthetic_qa',
                'tuple': {'b': int(b), 'e': int(e), 'd': int(d), 'a': int(a)},
                'invariants': {'J': int(J), 'K': int(K), 'X': int(X)},
                'modular_residues': {
                    f'mod{mod}': {
                        'b': int(b % mod),
                        'e': int(e % mod),
                        'd': int(d % mod),
                        'a': int(a % mod)
                    } for mod in modular_bases
                },
                'properties': {
                    'inner_ellipse_valid': bool(inner_ellipse_valid),
                    'is_fibonacci': self._is_fibonacci_sequence(b, e, d, a)
                },
                'domain': 'qa_synthetic'
            }

            examples.append(example)

            if (i + 1) % 2000 == 0:
                print(f"  Generated {i + 1}/{count} examples...")

        print(f"  ✓ Generated {len(examples)} synthetic examples")
        return examples

    def _is_fibonacci_sequence(self, b: int, e: int, d: int, a: int) -> bool:
        """Check if tuple follows Fibonacci pattern"""
        return (d == b + e) and (a == d + e)

    def incorporate_e8_geometry(self, data_path: str = None) -> List[Dict]:
        """
        Incorporate E8 geometry and signal processing data.
        Target: Multi-modal QA examples with geometric interpretations
        """
        print(f"[3/4] Incorporating E8 geometry and signal processing data")
        examples = []

        # E8 root system data
        e8_examples = self._generate_e8_qa_mappings(count=1000)
        examples.extend(e8_examples)

        # Signal processing examples from experiments
        try:
            signal_examples = self._extract_signal_processing_data()
            examples.extend(signal_examples)
        except Exception as e:
            print(f"  ⚠️  Could not extract signal data: {e}")

        print(f"  ✓ Incorporated {len(examples)} geometry/signal examples")
        return examples

    def _generate_e8_qa_mappings(self, count: int = 1000) -> List[Dict]:
        """Generate QA tuples with E8 alignment scores"""
        examples = []

        for _ in range(count):
            b, e = np.random.randint(1, 50, 2)
            d = b + e
            a = b + 2 * e

            # Simulate E8 embedding (8D projection)
            qa_8d = np.array([b, e, d, a, b*d, d*a, e*d, (b+e+d+a)/4])
            qa_8d = qa_8d / np.linalg.norm(qa_8d)

            # Simplified E8 alignment (would use actual E8 roots in production)
            alignment_score = np.abs(np.mean(np.cos(qa_8d * np.pi)))

            examples.append({
                'type': 'e8_qa_mapping',
                'tuple': {'b': int(b), 'e': int(e), 'd': int(d), 'a': int(a)},
                'e8_embedding': qa_8d.tolist(),
                'e8_alignment': float(alignment_score),
                'domain': 'e8_geometry'
            })

        return examples

    def _extract_signal_processing_data(self) -> List[Dict]:
        """Extract data from signal processing experiments"""
        examples = []

        # Check if signal experiments have generated data
        signal_script = Path("run_signal_experiments_final.py")
        if signal_script.exists():
            examples.append({
                'type': 'signal_experiment',
                'description': 'Audio signal classification using QA Harmonic Index',
                'script': str(signal_script),
                'domain': 'signal_processing'
            })

        return examples

    def create_qa_qa_pairs(self, count: int = 5000) -> List[Dict]:
        """
        Create question-answer pairs for QA reasoning.
        Target: 5K+ QA-specific Q&A examples
        """
        print(f"[4/4] Creating {count} question-answer pairs")
        qa_pairs = []

        # Template-based QA generation
        templates = [
            {
                'q': 'Given QA tuple ({b}, {e}, {d}, {a}), compute invariant J',
                'a': 'J = b × d = {b} × {d} = {J}',
                'type': 'invariant_computation'
            },
            {
                'q': 'Verify QA closure: does b + e = d for tuple ({b}, {e}, {d}, {a})?',
                'a': 'b + e = {b} + {e} = {sum_be}, d = {d}. Closure is {valid}',
                'type': 'closure_verification'
            },
            {
                'q': 'What is the mod {mod} residue of invariant K for tuple ({b}, {e}, {d}, {a})?',
                'a': 'K = d × a = {d} × {a} = {K}. K mod {mod} = {K_mod}',
                'type': 'modular_arithmetic'
            },
            {
                'q': 'Is tuple ({b}, {e}, {d}, {a}) a valid QA tuple?',
                'a': 'Check: b+e={sum_be}, d={d} ({check1}); e+d={sum_ed}, a={a} ({check2}). Valid: {valid}',
                'type': 'validation'
            }
        ]

        for i in range(count):
            b, e = np.random.randint(1, 50, 2)
            d = b + e
            a = e + d
            J, K, X = b * d, d * a, e * d
            mod = np.random.choice([9, 24, 72])

            template = templates[i % len(templates)]

            q = template['q'].format(
                b=b, e=e, d=d, a=a, mod=mod
            )
            closure_valid = (d == b+e and a == e+d)
            a_text = template['a'].format(
                b=b, e=e, d=d, a=a,
                J=J, K=K, X=X,
                sum_be=b+e, sum_ed=e+d,
                valid=closure_valid,
                check1='✓' if d == b+e else '✗',
                check2='✓' if a == e+d else '✗',
                mod=mod, K_mod=K % mod
            )

            qa_pairs.append({
                'type': 'qa_reasoning',
                'question': q,
                'answer': a_text,
                'qa_type': template['type'],
                'tuple': {'b': int(b), 'e': int(e), 'd': int(d), 'a': int(a)},
                'domain': 'qa_qa_pairs'
            })

            if (i + 1) % 1000 == 0:
                print(f"  Created {i + 1}/{count} Q&A pairs...")

        print(f"  ✓ Created {len(qa_pairs)} Q&A pairs")
        return qa_pairs

    def compile_dataset(self, vault_path: str = "QAnotes/") -> None:
        """Compile complete dataset from all sources"""
        print("\n" + "="*70)
        print("QA Training Dataset Curation (T-006)")
        print("="*70 + "\n")

        # Step 1: Extract theorems
        theorems = self.extract_theorems_from_vault(vault_path)
        self.dataset.extend(theorems)

        # Step 2: Generate synthetic examples
        synthetic = self.generate_synthetic_examples(count=10000)
        self.dataset.extend(synthetic)

        # Step 3: Incorporate E8 geometry
        e8_data = self.incorporate_e8_geometry()
        self.dataset.extend(e8_data)

        # Step 4: Create Q&A pairs
        qa_pairs = self.create_qa_qa_pairs(count=5000)
        self.dataset.extend(qa_pairs)

        print("\n" + "-"*70)
        print(f"Total dataset size: {len(self.dataset)} examples")
        print("-"*70)

    def save_dataset(self) -> None:
        """Save dataset in JSONL format"""
        print(f"\nSaving dataset to {self.output_path}...")

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                # Add metadata
                example['created_at'] = datetime.now().isoformat()
                example['version'] = '1.0'

                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(self.dataset)} examples")

        # Generate summary statistics
        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print("-"*70)

        # Count by type
        type_counts = {}
        domain_counts = {}

        for example in self.dataset:
            ex_type = example.get('type', 'unknown')
            domain = example.get('domain', 'unknown')

            type_counts[ex_type] = type_counts.get(ex_type, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print("\nBy Type:")
        for ex_type, count in sorted(type_counts.items()):
            print(f"  {ex_type:30} {count:>8,} examples")

        print("\nBy Domain:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain:30} {count:>8,} examples")

        print("\n" + "="*70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='QA Training Dataset Curator (T-006)')
    parser.add_argument('--vault', default='QAnotes/',
                       help='Path to QAnotes Obsidian vault')
    parser.add_argument('--output', default='qa_training_dataset.jsonl',
                       help='Output path for dataset')
    parser.add_argument('--synthetic-count', type=int, default=10000,
                       help='Number of synthetic examples to generate')
    parser.add_argument('--qa-pairs-count', type=int, default=5000,
                       help='Number of Q&A pairs to create')

    args = parser.parse_args()

    curator = QADatasetCurator(output_path=args.output)
    curator.compile_dataset(vault_path=args.vault)
    curator.save_dataset()

    print("\n✅ T-006 Dataset Curation Complete!")
    print(f"   Output: {args.output}")
    print(f"   Total examples: {len(curator.dataset):,}")


if __name__ == '__main__':
    main()
