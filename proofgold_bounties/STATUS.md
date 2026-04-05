# Proofgold Bounty Hunting - Current Status

**Last Updated:** 2026-01-24 16:30 EST

## ✅ Infrastructure Complete

### Litecoin Node
- **Status:** Fully synced and operational
- **Blocks:** 3,044,013+
- **RPC:** Configured with txindex=1
- **Location:** `/home/player2/.litecoin/`

### Proofgold Lava Client
- **Status:** Running in daemon mode (PID: 787714)
- **Version:** Built from source 2026-01-11
- **Data Directory:** `/home/player2/.proofgold/`
- **RPC:** Enabled with credentials in proofgold.conf
- **Logs:** `/home/player2/.proofgold/debug.log`

### Wallet
- **LTC Balance:** 0.074 LTC (~14 publications worth)
- **Receive Address:** ltc1qevdacgxeqcq8amf3857xzz5gknhj3kvlrgs35a

## 🔄 In Progress: Blockchain Sync

### Current Activity
Proofgold daemon is downloading and validating the blockchain:

```
[16:29:23] Got delta 7281faa3bdf457d9c0b9360705dfb7a6b435806d5bb2d2a5978951b8f749faab
[16:29:23] Got delta fafc5800d8524318fd65a6009fa84adf5fbd3d9ec367543b0afee292673caed3
[16:29:23] Got delta 33a5872fd7c2a69f2cd842002e6d32524f692dc98318d852232ede4f1eedb620
```

### Status Check
```bash
proofgoldcli getinfo
# Output: Exception: Failure("cannot find best validated header; probably out of sync")
```

This is **expected** during initial sync. The error will disappear once sync completes.

### Monitoring
Sync monitor running in background (PID: 788409)
- **Monitor log:** `/home/player2/signal_experiments/proofgold_bounties/data/logs/sync_monitor.log`
- **Check status:** `tail -f /home/player2/signal_experiments/proofgold_bounties/data/logs/sync_monitor.log`

### Estimated Completion
Unknown - depends on blockchain size. Typical range: **1-3 hours** for initial sync.

## 📝 Draft Files Ready

### Target: MetaCat Pullback Constructor
- **Proposition ID:** `c2d12cb7804aee9d668c3f2a183da3607108fd8fa11165ff4009eb5cc39862bc`
- **Theory:** HotG (Higher-Order Tarski-Grothendieck)
- **Theory Asset ID:** `205bb94e1cb9e6a6e7c01dd6fadd85a66ace11d84d498db4dcc608011825bdc1`
- **Bounty:** 250 bars
- **Draft File:** `/home/player2/signal_experiments/proofgold_bounties/drafts/refute_pullback_constr.pfg`

### Fixes Applied
1. ✅ **Document header:** Changed from base58 address to hex theory asset ID
2. ✅ **Daemon mode:** RPC enabled with proper credentials

### Known Issues to Address (After Sync)
1. **Proof syntax:** Current draft uses invalid `Proof/Qed/admit` syntax
   - Need to use IHOL proof terms: `PrAp`, `PrLa`, `TmAp`, `TmLa`, `Hyp`, `Known`
2. **Refutation strategy:** Proposition is existential (∃ x0 x2 x4 ... x6)
   - Refutation requires proof of `(∃ ...) -> False`
   - May need MetaCat lemmas to make this tractable
   - Consider pivoting to different bounty if not refutable with on-chain library

## 🎯 Next Steps (Once Sync Completes)

### 1. Test Draft Validation
```bash
proofgoldcli "readdraft /home/player2/signal_experiments/proofgold_bounties/drafts/refute_pullback_constr.pfg"
```

Expected outcome:
- If successful: Output showing document structure and any errors in proof terms
- This will reveal exactly what Known declarations are available from HotG
- Will show expected proof term format

### 2. Analyze Refutability
Based on readdraft output, determine:
- What MetaCat/HotG theorems are available for refutation
- Whether the existential claim is actually refutable
- If not, pivot to a different bounty target

### 3. Construct Proof or Pivot
- **If refutable:** Build IHOL proof term for `(∃ ...) -> False`
- **If not refutable:** Select different target from bounty list

### 4. Publish
Once proof validates:
```bash
./scripts/80_publish.sh
```

## 📊 Quick Reference

### Check Sync Status
```bash
proofgoldcli getinfo
```

### Monitor Sync Progress
```bash
tail -f /home/player2/.proofgold/debug.log | grep -E "Got delta|finished|bestblock"
```

### Validate Draft
```bash
proofgoldcli "readdraft /path/to/draft.pfg"
```

### View Wallet
```bash
proofgoldcli printassets
```

## 🚨 Important Notes

1. **Systemd service not configured:** Daemon running manually to avoid Type=simple + -daemon mismatch
2. **Sync must complete:** Cannot validate drafts until blockchain fully synced
3. **Existential claims tricky:** This particular bounty may not be ideal first target
4. **No rush:** Better to wait for full sync than force incomplete validation

## 📚 ChatGPT Guidance Summary

From 2026-01-24 handoff:

1. **Theory ID fix was correct:** HotG hex asset ID eliminated `c_hexstring_bebits` error
2. **Daemon/systemd issue understood:** Running manually is fine for now
3. **Proof syntax needs work:** `admit` not valid, must use actual IHOL terms
4. **Refutation reality check:** Existential claims need existing lemmas to refute
5. **Wait for sync, then test:** `readdraft` output will show exactly what's needed

## 🔗 Useful Links

- Proposition on explorer: https://formalweb3.uibk.ac.at/pgbce/q.php?b=c2d12cb7...
- HotG theory page: https://formalweb3.uibk.ac.at/pgbce/As.php?b=205bb94e1...
- Proofgold main site: http://proofgold.net (currently timing out)

---

**Status:** Ready to proceed once blockchain sync completes. Monitor `/home/player2/signal_experiments/proofgold_bounties/data/logs/sync_monitor.log` for completion notification.
