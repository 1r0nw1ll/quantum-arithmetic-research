# Docker Quick Start - Get Running in 10 Minutes

**Goal**: Run your first QA experiment in a Docker container.

---

## Prerequisites

- Linux system (Kali/Ubuntu)
- Internet connection
- 10 minutes of time

---

## Step 1: Install Docker (5 minutes)

```bash
# One-line install
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add yourself to docker group
sudo usermod -aG docker $USER
newgrp docker

# Test installation
docker run hello-world
```

**Expected**: "Hello from Docker!" message

---

## Step 2: Build Your First Container (3 minutes)

```bash
cd /home/player2/signal_experiments

# Build the image (takes ~5 min first time)
docker build -t qa-signal:latest .

# Verify it built
docker images | grep qa-signal
```

---

## Step 3: Run the Experiment (1 minute)

```bash
# Run signal processing experiment
docker run --rm -v $(pwd)/results:/app/results qa-signal:latest

# Check the output
ls -lh results/*.png
```

**Expected**: `signal_classification_results.png` created

---

## Step 4: Try Multi-Container Setup (1 minute)

```bash
# Start Jupyter + PostgreSQL + Redis
docker-compose up -d jupyter postgres redis

# Get Jupyter token
docker-compose logs jupyter | grep token

# Open browser: http://localhost:8888/?token=<your-token>
```

---

## You're Done! 🎉

**What you just did:**
- Installed Docker
- Built a custom image for QA experiments
- Ran an experiment in a container
- Started a multi-container environment

**Next steps:**
- Read `PHASE2_DOCKER_GUIDE.md` for detailed learning path
- Try building images for other experiments
- Explore Jupyter notebooks in the container

---

**Stuck?** Check `PHASE2_DOCKER_GUIDE.md` → Troubleshooting section.
