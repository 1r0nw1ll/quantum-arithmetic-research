#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-qa_moltbook/fixtures/INVALID__missing_required_fields.json}"
CLAIMED_TYPE="${2:-QA_CONJECTURE}"

echo "{\"target\":\"${TARGET}\",\"claimed_type\":\"${CLAIMED_TYPE}\",\"publish\":false}" \
  | python3 qa_moltbook/qa_moltbook_skill_validate.py \
  | python3 -m json.tool
