# Proofgold Bounty System - Quickstart

**Goal:** Get from zero to publishing proofs in 4 commands.

---

## Prerequisites

- Ubuntu/Debian Linux
- 50GB+ free disk space
- 4GB+ RAM
- Stable internet
- Sudo access

---

## 4-Step Setup

```bash
cd /home/player2/signal_experiments/proofgold_bounties

# Step 1: Verify system
./scripts/00_host_check.sh

# Step 2: Install dependencies (~10 min)
./scripts/10_install_deps_debian.sh

# Step 3: Setup blockchain infrastructure (~10 min)
./scripts/20_setup_litecoin.sh    # Litecoin node
./scripts/30_build_lava.sh        # Proofgold Lava
./scripts/40_config_proofgold.sh  # Configuration

# Step 4: Start and verify (~5 min)
./scripts/50_start_stack.sh
./scripts/60_healthcheck.sh
```

**Total time:** ~30 minutes + 1-2 hours blockchain sync

---

## Wait for Sync

Monitor sync progress:

```bash
watch -n 60 ./scripts/60_healthcheck.sh
```

Wait until:
- Litecoin: "Fully synced"
- Proofgold: Database populated

This takes **1-2 hours** on first run.

---

## Create Your First Proof

```bash
# Interactive workflow
./scripts/70_bounty_workflow.sh

# Select option 2: Create new draft
# Enter theory ID: HOHF
# Enter proposition ID: (from bounty list)

# Edit the draft:
nano drafts/draft_HOHF_*.pfg

# Validate before publishing:
./scripts/70_bounty_workflow.sh  # Select option 3
```

---

## Publish and Collect Bounty

```bash
./scripts/80_publish.sh

# Follow the prompts:
# 1. Select your draft
# 2. Confirm publication
# 3. Wait ~30 minutes for confirmations
# 4. Automatic publish and bounty collection
```

---

## Key Files

- **Scripts:** `scripts/*.sh`
- **Config:** `env/.env` (SECRET - RPC credentials)
- **Drafts:** `drafts/*.pfg`
- **Template:** `drafts/templates/hohf_document.template`
- **Full docs:** `README.md`

---

## Troubleshooting

**Sync stuck?**
```bash
journalctl -u litecoind -f
tail -f ~/.proofgold/proofgold.log
```

**RPC errors?**
```bash
./scripts/litecoin_rpc_test.sh
systemctl status litecoind
```

**Build failed?**
```bash
cat data/lava_build.log
```

---

## Next Steps

1. Read full README: `cat README.md`
2. Study IHOL grammar: https://prfgld.github.io/ihol.html
3. Browse bounties: https://proofgold.org
4. Start proving!

---

**Full documentation:** README.md
**Implementation details:** IMPLEMENTATION_SUMMARY.md
**Help:** https://prfgld.github.io/publishing.html
