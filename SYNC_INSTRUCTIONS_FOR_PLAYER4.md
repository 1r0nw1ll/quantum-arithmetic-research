# Project Synchronization Instructions for player4

## Objective
This document provides instructions for `player4` (as Gemini CLI) to synchronize the `signal_experiments` project directory from `player2` across the local area network. Once synchronized, `player4` can proceed with the QALM production training.

## Prerequisites for player4
1.  **Network Connectivity:** Ensure `player4` can reach `player2` over the local area network.
2.  **`rsync` Installed:** Verify `rsync` is installed on `player4` (usually pre-installed on Kali Linux).
3.  **`player2`'s IP Address/Hostname:** You will need the IP address or hostname of `player2`.
4.  **`player2`'s Username:** The username on `player2` (e.g., `player2`).

## Instructions for player4 (as Gemini CLI)

**Step 1: Obtain player2's IP Address (if unknown)**
If you don't know `player2`'s IP address, you can try to find it. On `player2`'s side, the user can run:
```bash
ip a | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1
```
Assume for these instructions that `player2`'s IP is `[PLAYER2_IP_ADDRESS]` and the username is `[PLAYER2_USERNAME]`. **You will need to replace these placeholders.**

**Step 2: Initiate Synchronization using `rsync`**
Execute the following command in your shell on `player4` to pull the `signal_experiments` project from `player2`. This command will create the `signal_experiments` directory in your current working directory on `player4`.

**IMPORTANT:** Replace `[PLAYER2_USERNAME]` with the actual username on `player2` and `[PLAYER2_IP_ADDRESS]` with `player2`'s IP address or hostname.

```bash
rsync -avz --progress [PLAYER2_USERNAME]@[PLAYER2_IP_ADDRESS]:/home/player2/signal_experiments/ ./signal_experiments/
```

**Explanation of `rsync` flags:**
*   `-a`: Archive mode; equals `-rlptgoD` (recursive, links, perms, times, group, owner, devices). This is generally what you want for backups and mirroring.
*   `-v`: Verbose; shows what files are being transferred.
*   `-z`: Compress file data during the transfer.
*   `--progress`: Shows progress during transfer.
*   `[PLAYER2_USERNAME]@[PLAYER2_IP_ADDRESS]:/home/player2/signal_experiments/`: This is the source path on `player2`.
*   `./signal_experiments/`: This is the destination path on `player4`.

**Step 3: Verify Synchronization**
After the `rsync` command completes, navigate into the new `signal_experiments` directory and list its contents to ensure the project files are present:
```bash
cd signal_experiments/
ls -F
```

**Step 4: Install Python Dependencies on player4**
Ensure all necessary Python libraries are installed on `player4`. Navigate to the `signal_experiments` directory and run:
```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

**Step 5: Start QALM Production Training on player4**
Once the project is synced and dependencies are installed, you can initiate the QALM production training. It is recommended to start with the "Medium Model" configuration to balance performance and resource usage.

```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --hidden-size 512 \
    --num-layers 8 \
    --num-heads 8 \
    --checkpoint-dir checkpoints/qalm_v1_medium
```

**Step 6: Monitor Training on player4**
You can monitor the training progress by tailing the log file:
```bash
tail -f qalm_training.log
```

This completes the synchronization and setup for QALM training on `player4`.
