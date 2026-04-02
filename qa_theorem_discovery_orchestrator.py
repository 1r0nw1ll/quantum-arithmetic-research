#!/usr/bin/env python3
"""
QA Automated Theorem Discovery Orchestrator
Coordinates all agents for end-to-end theorem discovery pipeline
"""

import subprocess
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Orchestrator")


class QATheoremDiscoveryOrchestrator:
    """
    Main orchestrator for the 5-stage theorem discovery pipeline
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.workspace = Path(self.config.get('workspace', './qa_discovery_workspace'))
        self.workspace.mkdir(exist_ok=True)

        # Results tracking
        self.results = {
            'start_time': None,
            'end_time': None,
            'stages': {},
            'errors': [],
            'summary': {}
        }

        # Default configuration
        self.default_config = {
            'dataset': 'qa_10000_balanced_tuples.csv',
            'graph_output': self.workspace / 'qa_graph.pt',
            'embeddings_output': self.workspace / 'qa_embeddings.npy',
            'conjectures_output': self.workspace / 'conjectures.json',
            'lean_proofs_dir': self.workspace / 'lean_proofs',
            'checkpoint_dir': self.workspace / 'checkpoints',
            'edge_mode': 'pipeline',
            'gnn_epochs': 300,
            'gnn_checkpoint_interval': 50,
            'gnn_log_interval': 10,
            'clustering_method': 'dbscan',
            'max_conjectures_to_verify': 20,
            'canonical_validator': Path(
                "Formalizing tuple drift in quantum-native learning/files/files(1)/validate_canonical_v2.py"
            ),
            'run_preflight': True,
        }

        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

    def run_stage(self, stage_name, command, description):
        """
        Run a single stage of the pipeline
        """
        logger.info("="*70)
        logger.info(f"STAGE: {stage_name}")
        logger.info(f"{description}")
        logger.info("="*70)

        stage_start = time.time()

        try:
            # Run command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            stage_time = time.time() - stage_start

            # Record success
            self.results['stages'][stage_name] = {
                'status': 'success',
                'duration': stage_time,
                'output': result.stdout[-1000:],  # Last 1000 chars
                'command': ' '.join(command)
            }

            logger.info(f"\n✓ Stage completed successfully in {stage_time:.2f}s")

            return True

        except subprocess.CalledProcessError as e:
            stage_time = time.time() - stage_start

            # Record failure
            self.results['stages'][stage_name] = {
                'status': 'failed',
                'duration': stage_time,
                'error': e.stderr,
                'output': e.stdout, # Add stdout to error log
                'command': ' '.join(command)
            }

            logger.error(f"\n✗ Stage failed after {stage_time:.2f}s")
            logger.error(f"Error: {e.stderr}")
            logger.error(f"Output: {e.stdout}") # Print stdout as well

            self.results['errors'].append({
                'stage': stage_name,
                'error': str(e),
                'stderr': e.stderr
            })

            return False

    def stage_0_preflight(self):
        """
        Stage 0: Canonical validator preflight
        """
        validator_path = Path(self.config['canonical_validator'])
        command = ['python', str(validator_path)]

        logger.info("="*70)
        logger.info("STAGE: Stage 0: Canonical Preflight")
        logger.info("Validating canonical QA invariants and checksums")
        logger.info("="*70)

        stage_start = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        stage_time = time.time() - stage_start
        output = (result.stdout or "") + (result.stderr or "")
        failed = result.returncode != 0 or "VALIDATION FAILED" in output

        self.results['stages']['Stage 0: Canonical Preflight'] = {
            'status': 'failed' if failed else 'success',
            'duration': stage_time,
            'output': output[-1000:],
            'command': ' '.join(command),
        }

        if failed:
            logger.error(f"\n✗ Stage failed after {stage_time:.2f}s")
            logger.error(output[-1000:])
            self.results['errors'].append({
                'stage': 'Stage 0: Canonical Preflight',
                'error': 'Canonical validation failed',
                'stderr': result.stderr,
            })
            return False

        logger.info(f"\n✓ Stage completed successfully in {stage_time:.2f}s")
        return True

    def stage_1_build_graph(self):
        """
        Stage 1: Build QA tuple graph
        """
        command = [
            'python', 'qa_graph_builder_v2.py',
            '--input', str(self.config['dataset']),
            '--output', str(self.config['graph_output']),
            '--checkpoint-dir', str(self.config['checkpoint_dir']),
            '--edge-mode', str(self.config['edge_mode'])
        ]

        return self.run_stage(
            stage_name="Stage 1: Graph Construction",
            command=command,
            description="Building QA tuple graph with harmonic, modular, and geometric edges"
        )

    def stage_2_train_gnn(self):
        """
        Stage 2: Train GNN
        """
        command = [
            'python', 'qa_gnn_trainer_v2.py',
            '--graph', str(self.config['graph_output']),
            '--epochs', str(self.config['gnn_epochs']),
            '--checkpoint-interval', str(self.config['gnn_checkpoint_interval']),
            '--log-interval', str(self.config['gnn_log_interval']),
            '--output-dir', str(self.config['checkpoint_dir']),
            '--embeddings-output', str(self.config['embeddings_output'])
        ]

        return self.run_stage(
            stage_name="Stage 2: GNN Training",
            command=command,
            description="Training Graph Neural Network to learn tuple patterns"
        )

    def stage_3_mine_conjectures(self):
        """
        Stage 3: Mine symbolic conjectures
        """
        command = [
            'python', 'qa_symbolic_miner_v2.py',
            '--embeddings', str(self.config['embeddings_output']),
            '--graph', str(self.config['graph_output']),
            '--dataset', str(self.config['dataset']),
            '--output', str(self.config['conjectures_output']),
            '--clustering', str(self.config['clustering_method'])
        ]

        return self.run_stage(
            stage_name="Stage 3: Conjecture Mining",
            command=command,
            description="Extracting mathematical conjectures from GNN embeddings"
        )

    def stage_4_verify_lean(self):
        """
        Stage 4: Verify with Lean
        """
        command = [
            'python', 'qa_lean_verifier_v2.py',
            '--conjectures', str(self.config['conjectures_output']),
            '--output-dir', str(self.config['lean_proofs_dir']),
            '--max-conjectures', str(self.config['max_conjectures_to_verify'])
        ]

        return self.run_stage(
            stage_name="Stage 4: Formal Verification",
            command=command,
            description="Verifying conjectures with Lean 4 theorem prover"
        )

    def stage_5_export_results(self):
        """
        Stage 5: Export results
        """
        logger.info("="*70)
        logger.info("STAGE: Stage 5: Results Export")
        logger.info("Generating final report and visualizations")
        logger.info("="*70)

        stage_start = time.time()

        try:
            # Load results from previous stages
            conjectures_path = self.config['conjectures_output']
            proofs_path = self.config['lean_proofs_dir'] / 'proof_records.json'

            # Generate summary report
            summary = self.generate_summary_report(conjectures_path, proofs_path)

            # Save summary
            summary_path = self.workspace / 'discovery_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"✓ Summary report saved to {summary_path}")

            # Generate human-readable report
            self.generate_text_report(summary)

            stage_time = time.time() - stage_start

            self.results['stages']['Stage 5: Export'] = {
                'status': 'success',
                'duration': stage_time
            }

            return True

        except Exception as e:
            logger.error(f"✗ Export failed: {str(e)}")
            return False

    def generate_summary_report(self, conjectures_path, proofs_path):
        """
        Generate numerical summary of discoveries
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {k: str(v) for k, v in self.config.items()},
            'statistics': {}
        }

        # Load conjectures
        if Path(conjectures_path).exists():
            with open(conjectures_path, 'r') as f:
                conjectures = json.load(f)
                summary['statistics']['total_conjectures'] = len(conjectures)
                summary['statistics']['top_conjecture_score'] = conjectures[0]['rank_score'] if conjectures else 0

        # Load proofs
        if Path(proofs_path).exists():
            with open(proofs_path, 'r') as f:
                proofs = json.load(f)
                summary['statistics']['verified_proofs'] = len(proofs.get('verified', []))
                summary['statistics']['failed_proofs'] = len(proofs.get('failed', []))

        return summary

    def generate_text_report(self, summary):
        """
        Generate human-readable text report
        """
        report_path = self.workspace / 'DISCOVERY_REPORT.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("QA AUTOMATED THEOREM DISCOVERY - FINAL REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Workspace: {self.workspace}\n\n")

            f.write("RESULTS SUMMARY:\n")
            f.write("-"*70 + "\n")

            stats = summary.get('statistics', {})
            f.write(f"Total Conjectures Discovered: {stats.get('total_conjectures', 0)}\n")
            f.write(f"Verified Proofs: {stats.get('verified_proofs', 0)}\n")
            f.write(f"Failed Verifications: {stats.get('failed_proofs', 0)}\n")
            f.write(f"Top Conjecture Score: {stats.get('top_conjecture_score', 0):.2f}\n\n")

            f.write("STAGE EXECUTION:\n")
            f.write("-"*70 + "\n")

            for stage_name, stage_data in self.results['stages'].items():
                status = stage_data['status']
                duration = stage_data.get('duration', 0)
                f.write(f"{stage_name}: {status.upper()} ({duration:.2f}s)\n")

            f.write("\n")

            total_time = self.results.get('end_time', 0) - self.results.get('start_time', 0)
            f.write(f"Total Pipeline Duration: {total_time:.2f}s ({total_time/60:.2f}m)\n")

            f.write("\n" + "="*70 + "\n")

        logger.info(f"✓ Text report saved to {report_path}")

    def run_full_pipeline(self):
        """
        Execute the complete 5-stage pipeline
        """
        logger.info("╔"+"="*68+"╗")
        logger.info("║" + " "*68 + "║")
        logger.info("║" + "  QA AUTOMATED THEOREM DISCOVERY PIPELINE".center(68) + "║")
        logger.info("║" + " "*68 + "║")
        logger.info("╚"+"="*68+"╝")

        self.results['start_time'] = time.time()

        # Stage 0: Canonical preflight
        if self.config.get('run_preflight', True):
            if not self.stage_0_preflight():
                logger.error("Pipeline aborted: Stage 0 failed")
                return False

        # Stage 1: Build Graph
        if not self.stage_1_build_graph():
            logger.error("Pipeline aborted: Stage 1 failed")
            return False

        # Stage 2: Train GNN
        if not self.stage_2_train_gnn():
            logger.error("Pipeline aborted: Stage 2 failed")
            return False

        # Stage 3: Mine Conjectures
        if not self.stage_3_mine_conjectures():
            logger.error("Pipeline aborted: Stage 3 failed")
            return False

        # Stage 4: Verify with Lean
        if not self.stage_4_verify_lean():
            logger.warning("Stage 4 had issues, but continuing...")

        # Stage 5: Export Results
        self.results['end_time'] = time.time() # Moved this line here
        self.stage_5_export_results()
        total_time = self.results['end_time'] - self.results['start_time']

        # Final summary
        logger.info("\n" + "╔"+"="*68+"╗")
        logger.info("║" + " "*68 + "║")
        logger.info("║" + "  PIPELINE COMPLETE!".center(68) + "║")
        logger.info("║" + " "*68 + "║")
        logger.info("╚"+"="*68+"╝")

        logger.info(f"\nTotal Duration: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info(f"Workspace: {self.workspace}")
        logger.info(f"\nCheck {self.workspace}/DISCOVERY_REPORT.txt for full results")

        return True


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Run complete QA theorem discovery pipeline'
    )
    parser.add_argument('--dataset', default='qa_10000_balanced_tuples.csv',
                       help='Input dataset CSV')
    parser.add_argument('--workspace', default='./qa_discovery_workspace',
                       help='Workspace directory')
    parser.add_argument('--epochs', type=int, default=300,
                       help='GNN training epochs')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs, fewer verifications)')
    parser.add_argument('--edge-mode', default='pipeline', choices=['pipeline', 'canonical'],
                       help='Edge construction mode for graph builder')
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip canonical validator preflight')

    args = parser.parse_args()

    # Build config
    config = {
        'dataset': args.dataset,
        'workspace': args.workspace,
        'gnn_epochs': 50 if args.quick else args.epochs,
        'gnn_checkpoint_interval': 10 if args.quick else 50,
        'max_conjectures_to_verify': 5 if args.quick else 20,
        'edge_mode': args.edge_mode,
        'run_preflight': not args.skip_preflight,
    }

    try:
        orchestrator = QATheoremDiscoveryOrchestrator(config)
        success = orchestrator.run_full_pipeline()

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
