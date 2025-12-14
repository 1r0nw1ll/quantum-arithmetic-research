# GEMINI.md: Project Overview

## Directory Overview

This directory contains a multi-faceted research project centered around a novel mathematical framework called the **Quantum Arithmetic (QA) System**. The project explores the theoretical foundations of the QA system, its applications in signal processing and finance, and the use of machine learning to automate the discovery of related geometric theorems.

The research is highly computational, with a strong emphasis on Python for simulation, analysis, and visualization. The project is well-documented, with a formal LaTeX report detailing the mathematical properties of the QA system.

## Key Files

*   `qa_formal_report.tex`: A LaTeX document that provides a formal description of the Quantum Arithmetic (QA) system. It details the mathematical and geometric properties of the system, including its multi-orbit structure and its alignment with the E8 exceptional Lie algebra.
*   `run_signal_experiments_final.py`: A Python script that applies the QA system to the classification of audio signals. It implements the `QASystem` class and uses the "Harmonic Index" to distinguish between different types of signals.
*   `geometrist_v4_gnn.py`: A Python script that uses a Graph Neural Network (GNN) to learn and generate "valid geometric theorems" based on concepts from Universal Hyperbolic Geometry. This represents a line of research into automated theorem proving.
*   `backtest_advanced_strategy.py`: A Python script that backtests a financial trading strategy on the S&P 500 (SPY) using the "Harmonic Index" as a key signal.

## Usage

This directory is a research environment, not a traditional software project. The primary way to interact with the project is by running the Python scripts to reproduce the experiments and generate the results.

### Running the Experiments

The main experiments can be run directly from the command line:

```bash
# To run the signal classification experiment:
python run_signal_experiments_final.py

# To run the geometric theorem generation experiment:
python geometrist_v4_gnn.py

# To run the financial backtesting experiment:
python backtest_advanced_strategy.py
```

**Note:** You may need to install the required Python libraries (e.g., `numpy`, `matplotlib`, `pandas`, `torch`, `scikit-learn`, `tqdm`) before running the scripts. You can typically install them using `pip`:

```bash
pip install numpy matplotlib pandas torch scikit-learn tqdm
```
