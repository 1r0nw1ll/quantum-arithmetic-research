# Proofgold Lava Bounty System

Complete local setup for proving mathematical theorems and collecting bounties on the Proofgold blockchain.

**Status:** Production-ready infrastructure
**Created:** 2026-01-10
**Blockchain:** Proofgold (backed by Litecoin)

---

## Table of Contents

1. [What is Proofgold?](#what-is-proofgold)
2. [Quickstart (4 Commands)](#quickstart-4-commands)
3. [Detailed Setup](#detailed-setup)
4. [Bounty Workflow](#bounty-workflow)
5. [Publishing Proofs](#publishing-proofs)
6. [Troubleshooting](#troubleshooting)
7. [Architecture](#architecture)
8. [Security Notes](#security-notes)
9. [Resources](#resources)

---

## What is Proofgold?

**Proofgold** is a peer-to-peer cryptocurrency for trading formal mathematical proofs. Key features:

- **Blockchain-backed:** Proofs are published to the Proofgold blockchain (anchored to Litecoin)
- **Bounty system:** Propositions can have bounties; prove them and collect the reward
- **Formal verification:** All proofs are machine-checked using a higher-order logic system
- **IHOL language:** Proofs written in an assembly-style language based on Church's Simple Type Theory

**Proofgold Lava** is a high-performance client implementation (faster than the original Proofgold Core).

### How Bounties Work

1. Someone posts a bounty on an unproven proposition
2. You write a formal proof in Proofgold's IHOL language
3. You **commit** your proof (anti-frontrunning mechanism)
4. After confirmations, you **publish** your proof
5. If valid, you become the owner and can **collect the bounty**

---

## Quickstart (4 Commands)

```bash
# 1. Verify your system meets requirements
./scripts/00_host_check.sh

# 2. Install all dependencies (OCaml, Zarith, Litecoin, etc.)
./scripts/10_install_deps_debian.sh

# 3. Set up Litecoin node and Proofgold Lava
./scripts/20_setup_litecoin.sh
./scripts/30_build_lava.sh
./scripts/40_config_proofgold.sh

# 4. Start the stack and verify health
./scripts/50_start_stack.sh
./scripts/60_healthcheck.sh

# Now wait for sync (1-2 hours), then start proving!
./scripts/70_bounty_workflow.sh  # Create and validate proofs
./scripts/80_publish.sh          # Publish and collect bounties
```

**Important:** After starting the stack, Litecoin and Proofgold must sync with their blockchains. This can take 1-2 hours or more depending on network speed.

---

## Detailed Setup

### Prerequisites

- **OS:** Debian/Ubuntu Linux (tested on Ubuntu)
- **Disk:** 50GB+ free space (for blockchain sync)
- **RAM:** 4GB+ recommended
- **Internet:** Stable connection for syncing
- **Sudo access:** Required for installing packages

### Step-by-Step Installation

#### 1. System Verification

```bash
cd proofgold_bounties
./scripts/00_host_check.sh
```

Checks:
- OS compatibility
- Disk space
- Memory
- Internet connectivity
- Sudo access

#### 2. Install Dependencies

```bash
./scripts/10_install_deps_debian.sh
```

Installs:
- Build tools (gcc, make, m4, pkg-config)
- OCaml + opam (package manager)
- Zarith library (arbitrary precision integers)
- GMP and DBM libraries
- Git, curl, wget

**Time:** 5-10 minutes
**Log:** `data/install_deps.log`

#### 3. Setup Litecoin Node

```bash
./scripts/20_setup_litecoin.sh
```

- Downloads and installs Litecoin daemon
- Generates strong RPC credentials
- Creates `litecoin.conf` with security best practices
- Sets up systemd service
- Locks RPC to localhost only (127.0.0.1)

**Security:** RPC credentials saved to `env/.env` (mode 600, keep secret!)

#### 4. Build Proofgold Lava

```bash
./scripts/30_build_lava.sh
```

- Clones `github.com/ckaliszyk/proofgold-lava`
- Builds from source using OCaml
- Creates binary symlink at `scripts/proofgoldlava`
- Records build version and commit hash

**Time:** 5-10 minutes
**Log:** `data/lava_build.log`

#### 5. Configure Proofgold

```bash
./scripts/40_config_proofgold.sh
```

Interactive configuration:
- Choose bootstrap mode (default or independent)
- Optionally enable peer listening
- Creates `~/.proofgold/proofgold.conf`
- Links Litecoin RPC credentials

#### 6. Start the Stack

```bash
./scripts/50_start_stack.sh
```

Options:
- **Daemon mode** (recommended): Runs as systemd user service
- **Foreground mode**: Runs in terminal (debugging)
- **tmux mode**: Background session with easy reattachment

Starts:
1. Litecoin daemon (via systemd)
2. Proofgold Lava client (your choice of mode)

#### 7. Verify Health

```bash
./scripts/60_healthcheck.sh
```

Checks:
- Litecoin service status
- Litecoin RPC connectivity
- Litecoin sync progress
- Proofgold Lava process
- Proofgold data directory
- Disk and memory usage

**Expected:** Initial run will show "still syncing". Wait for full sync before bounty work.

---

## Bounty Workflow

Once synced, use the interactive bounty workflow:

```bash
./scripts/70_bounty_workflow.sh
```

### Menu Options

1. **List available bounties**
   - Queries Proofgold blockchain for open bounties
   - Shows HOHF reward bounties (legacy theory with prizes)

2. **Create new draft from template**
   - Scaffolds a new Proofgold document
   - Uses IHOL grammar templates
   - Saves to `drafts/` directory

3. **Validate existing draft (readdraft)**
   - Locally checks proof syntax
   - Verifies against Proofgold type system
   - Must pass before publishing

4. **Show bounty statistics**
   - Your current assets
   - Available bounties
   - Recent claims

5. **Help: Proofgold document format**
   - IHOL grammar reference
   - Proof term constructors
   - Example structures

### Document Structure

A Proofgold document has this format:

```proofgold
Document THEORY_ID

(* Known declarations - imports *)
Known and : prop -> prop -> prop

(* Definitions *)
Def myFunc : set -> set := fun x => x

(* Main theorem *)
Thm target : statement
Proof:
  proof_term
Qed.
```

See `drafts/templates/hohf_document.template` for a full annotated template.

---

## Publishing Proofs

### The Publishing Pipeline

Proofgold uses a two-stage commit-publish protocol to prevent frontrunning:

```bash
./scripts/80_publish.sh
```

**Steps:**

1. **Add nonce** - Randomizes commitment
2. **Add publisher** - Signs with your key
3. **Commit draft** - Publishes commitment transaction to Litecoin
4. **Wait 12 confirmations** - Anti-frontrunning delay (~30 minutes)
5. **Publish draft** - Reveals actual proof
6. **Collect bounties** - Claims any eligible rewards

**Costs:** Publishing requires Litecoin for transaction fees. Ensure your wallet has sufficient LTC.

### Confirmation Wait

Litecoin block time is ~2.5 minutes. Waiting for 12 confirmations takes approximately **30 minutes**.

The script polls automatically and shows progress:
```
Confirmations: 8 / 12 [66%]
```

### After Publishing

Check results:
```bash
# View your assets
./scripts/proofgoldlava printassets

# Check publication log
cat drafts/publications.log

# Proofgold explorer (web)
https://proofgold.org
```

---

## Troubleshooting

### Sync Issues

**Symptom:** Litecoin or Proofgold not syncing

**Solutions:**

1. **Check internet connection:**
   ```bash
   ping 8.8.8.8
   ```

2. **Verify Litecoin is running:**
   ```bash
   systemctl status litecoind
   journalctl -u litecoind -f
   ```

3. **Check Proofgold logs:**
   ```bash
   tail -f ~/.proofgold/proofgold.log
   ```

4. **Wipe and re-sync (last resort):**
   ```bash
   systemctl --user stop proofgoldlava
   rm -rf ~/.proofgold/db
   systemctl --user start proofgoldlava
   ```

### Build Failures

**Symptom:** Lava build fails

**Check:**
1. OCaml version: `ocaml -version` (need 4.10+)
2. Zarith installed: `opam list | grep zarith`
3. Build log: `cat data/lava_build.log`

**Fix:**
```bash
# Reinstall opam environment
opam update
opam upgrade
opam install zarith num cryptokit
./scripts/30_build_lava.sh
```

### RPC Connection Errors

**Symptom:** "Connection refused" when checking Litecoin

**Check:**
1. Service running: `systemctl status litecoind`
2. RPC credentials: `cat ~/.litecoin/litecoin.conf`
3. Port listening: `netstat -tlnp | grep 9332`

**Fix:**
```bash
# Restart Litecoin
sudo systemctl restart litecoind

# Check startup errors
journalctl -u litecoind --since "5 minutes ago"
```

### Publishing Failures

**Symptom:** "Insufficient funds" or publish fails

**Solutions:**

1. **Check wallet balance:**
   ```bash
   ./scripts/proofgoldlava printassets
   ```

2. **Ensure Litecoin synced:**
   ```bash
   litecoin-cli -rpcuser=$LITECOIN_RPC_USER -rpcpassword=$LITECOIN_RPC_PASS getblockchaininfo
   ```

3. **Verify draft passes readdraft:**
   ```bash
   ./scripts/70_bounty_workflow.sh  # Option 3
   ```

---

## Architecture

### Directory Structure

```
proofgold_bounties/
├── README.md                    # This file
├── env/
│   ├── .env                    # RPC credentials (generated, SECRET)
│   └── .env.example            # Template
├── scripts/
│   ├── 00_host_check.sh       # System verification
│   ├── 10_install_deps_debian.sh
│   ├── 20_setup_litecoin.sh
│   ├── 30_build_lava.sh
│   ├── 40_config_proofgold.sh
│   ├── 50_start_stack.sh
│   ├── 60_healthcheck.sh
│   ├── 70_bounty_workflow.sh
│   ├── 80_publish.sh
│   ├── proofgoldlava -> ../data/proofgold-lava/binary
│   ├── litecoin_rpc_test.sh
│   └── proofgold_status.sh
├── data/
│   ├── litecoin/              # Optional: Litecoin data (or uses ~/.litecoin)
│   ├── proofgold/             # Proofgold data (or uses ~/.proofgold)
│   ├── proofgold-lava/        # Lava source code
│   ├── logs/                  # Runtime logs
│   ├── build_versions.txt     # Dependency versions
│   └── lava_build_version.txt # Lava build info
└── drafts/
    ├── templates/
    │   └── hohf_document.template
    ├── draft_*.pfg            # Your proofs
    └── publications.log       # Publication history
```

### External Directories

- `~/.litecoin/` - Litecoin blockchain data (~5GB+)
- `~/.proofgold/` - Proofgold blockchain data (~500MB+)

### Services

- **litecoind:** System service (`/etc/systemd/system/litecoind.service`)
- **proofgoldlava:** User service (`~/.config/systemd/user/proofgoldlava.service`)

### Network Ports

- Litecoin RPC: `127.0.0.1:9332` (localhost only, secured)
- Litecoin P2P: `9333` (default, outbound)
- Proofgold P2P: `21805` (optional, if listening enabled)

---

## Security Notes

### Critical Security Practices

1. **RPC Credentials:**
   - Stored in `env/.env` (mode 600)
   - **Never** commit to git
   - **Never** share publicly
   - Generated with strong randomness (32-byte base64)

2. **RPC Binding:**
   - Locked to `127.0.0.1` (localhost only)
   - Firewall blocks external access
   - No remote RPC (prevents fund theft)

3. **Wallet Security:**
   - Proofgold wallet stored in `~/.proofgold/wallet.dat`
   - Back up wallet before major operations
   - Consider encrypting wallet

4. **System Security:**
   - Run as non-root user
   - Use systemd for process isolation
   - Keep system updated: `sudo apt-get update && sudo apt-get upgrade`

### Best Practices

- **Testnet first:** Use `LITECOIN_NETWORK="testnet"` for testing
- **Backups:** Regularly backup `~/.proofgold/wallet.dat`
- **Firewall:** Ensure inbound 9332 is blocked
- **Monitoring:** Check logs regularly for anomalies

---

## Resources

### Official Proofgold Documentation

- **Main site:** https://proofgold.org
- **Installation (Core):** https://prfgld.github.io/installation.html
- **Publishing guide:** https://prfgld.github.io/publishing.html
- **IHOL grammar:** https://prfgld.github.io/ihol.html
- **Explorer:** https://proofgold.org (view published proofs)

### Academic Papers

- **FMBC'22:** "Proofgold: Blockchain for Formal Methods"
  - https://d-nb.info/1365349454/34
  - Describes architecture, commitment protocol, HOL4 integration

- **Exploring Formal Math on Blockchain (2025):**
  - https://arxiv.org/html/2509.08267v1
  - Proofgold explorer and analysis

### Community

- **GitHub:** https://github.com/ckaliszyk/proofgold-lava
- **Litecoin:** https://litecoin.org
- **HOL4:** https://hol-theorem-prover.org (for automated proving)

### Bounty Strategies

1. **Start with HOHF:** Legacy theory with existing bounties
2. **Automated proving:** Use HOL4 + export to Proofgold format
3. **Proof compression:** Keep proofs small (block size limits)
4. **Check explorer:** See what's already proven to avoid duplication

---

## Quick Reference

### Service Management

```bash
# Litecoin
sudo systemctl start litecoind
sudo systemctl stop litecoind
sudo systemctl status litecoind
journalctl -u litecoind -f

# Proofgold Lava (if using systemd user service)
systemctl --user start proofgoldlava
systemctl --user stop proofgoldlava
systemctl --user status proofgoldlava
journalctl --user -u proofgoldlava -f
```

### Health Checks

```bash
# Full health check
./scripts/60_healthcheck.sh

# Litecoin RPC test
./scripts/litecoin_rpc_test.sh

# Proofgold status
./scripts/proofgold_status.sh

# Check sync
litecoin-cli -rpcuser=$LITECOIN_RPC_USER -rpcpassword=$LITECOIN_RPC_PASS getblockchaininfo
```

### Common Tasks

```bash
# Create proof draft
./scripts/70_bounty_workflow.sh  # Option 2

# Validate draft
./scripts/70_bounty_workflow.sh  # Option 3

# Publish proof
./scripts/80_publish.sh

# Check assets
./scripts/proofgoldlava printassets
```

---

## Development Roadmap

### Future Enhancements

- [ ] HOL4 integration for automated proving
- [ ] Batch certificate generation from AlphaGeometry
- [ ] Web UI for bounty browsing
- [ ] Automated bounty targeting (priority queue)
- [ ] Proof compression utilities
- [ ] Multi-signature publishing

---

## License

This infrastructure toolkit is released under the MIT License.

Proofgold and Proofgold Lava are subject to their own licenses. See:
- https://github.com/ckaliszyk/proofgold-lava

---

## Support

For issues with this toolkit:
- Open an issue on GitHub (if available)
- Check troubleshooting section above
- Review logs in `data/` directory

For Proofgold protocol questions:
- See official documentation: https://prfgld.github.io
- Check Proofgold community channels

---

**Last Updated:** 2026-01-10
**Version:** 1.0.0
**Status:** Production Ready
