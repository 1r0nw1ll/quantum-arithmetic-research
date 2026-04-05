#!/usr/bin/env bash
set -euo pipefail

run_attest() {
  local target="$1"
  local claimed_type="$2"
  local source_url="$3"
  echo
  echo "=== ${claimed_type} ==="
  echo "{\"mode\":\"attest\",\"target\":\"${target}\",\"claimed_type\":\"${claimed_type}\",\"source_url\":\"${source_url}\",\"publish\":false}" \
    | python3 qa_moltbook/qa_moltbook_skill_validate.py \
    | python3 -m json.tool
}

run_attest \
  "qa_moltbook/fixtures/moltbook_live_snapshots/moltbook_skill_json__2026-02-08.snapshot.json" \
  "MOLTBOOK_SKILL_JSON.v1" \
  "https://www.moltbook.com/skill.json"

run_attest \
  "qa_moltbook/fixtures/moltbook_live_snapshots/moltbook_developers_excerpt__2026-02-08.txt" \
  "MOLTBOOK_DEVELOPERS_EXCERPT.v1" \
  "https://moltbook.com/developers"

run_attest \
  "qa_moltbook/fixtures/moltbook_live_snapshots/moltbook_post_shell__2026-02-08.txt" \
  "MOLTBOOK_POST_SHELL.v1" \
  "https://www.moltbook.com/post/cbd6474f-8478-4894-95f1-7b104a73bcd5"
