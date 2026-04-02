---
name: ob-status
description: Open Brain status — recent thoughts, stats, and any items needing action
user_invocable: true
---

Run the Open Brain session-start protocol:

1. `mcp__open-brain__recent_thoughts` with since_days=3, limit=10
2. `mcp__open-brain__thought_stats`

Summarize: how many thoughts in last 3 days, breakdown by type (observation/task/reference), and flag any tasks that look unresolved or stale. Keep it to ~10 lines.
