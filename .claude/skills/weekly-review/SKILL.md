---
name: weekly-review
description: Run the weekly review pipeline — activity log + scorecard + review draft
user_invocable: true
---

Run the weekly review pipeline:

1. `python build_activity_log.py 7` — generate activity log for last 7 days
2. `python weekly_review.py` — generate automated scorecard
3. Check if `reviews/` directory exists, list existing reviews
4. Draft a review summary covering:
   - Key accomplishments this week
   - Cert families added/modified
   - Open Brain capture rate
   - Git commit frequency (check against 7-day hygiene rule)
   - Blockers or stale items
   - Priorities for next week

Save draft to `reviews/review_YYYY-MM-DD.md` using today's date.
