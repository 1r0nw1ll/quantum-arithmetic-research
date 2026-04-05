#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-qa_alphageometry_ptolemy/qa_ledger/conjectures/QA_CONJ__SUBSTRATE_INVARIANCE__v1.json}"
CLAIMED_TYPE="${2:-QA_CONJECTURE}"

echo "{\"target\":\"${TARGET}\",\"claimed_type\":\"${CLAIMED_TYPE}\",\"publish\":false}" \
  | python3 qa_moltbook/qa_moltbook_skill_validate.py \
  | python3 -m json.tool
