#!/bin/bash
echo "--- Running Benchmark Query: 'What is QA?' ---"
python qa_graph_query.py "What is QA?" --method ppr_hybrid --top-k 7

echo -e "\n--- Running Benchmark Query: 'football network experiment' ---"
python qa_graph_query.py "football network experiment" --method ppr_hybrid --top-k 7
