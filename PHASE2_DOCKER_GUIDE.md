# Phase 2: Docker Foundation - Complete Guide

**Timeline**: Weeks 3-6 (4 weeks)
**Goal**: Containerize all QA experiments for reproducibility and portability

---

## Week 3: Docker Basics & First Container

### Day 1-2: Installation

**Install Docker on Linux:**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

**Expected output**: Docker version info + "Hello from Docker!" message

---

### Day 3-4: Learn Docker Basics

**Run the interactive tutorial:**
```bash
docker run -d -p 80:80 docker/getting-started
# Open browser: http://localhost
```

**Complete these sections** (2-3 hours total):
1. What is a container?
2. Build your first image
3. Update your app
4. Share your image
5. Persist data with volumes
6. Use bind mounts

**Key commands to practice:**
```bash
# List images
docker images

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Remove container
docker rm <container-id>

# Remove image
docker rmi <image-name>

# View logs
docker logs <container-id>

# Interactive shell
docker exec -it <container-id> bash
```

---

### Day 5-7: Build Your First QA Container

**Step 1: Build the image**
```bash
cd /home/player2/signal_experiments

# Build image (takes 5-10 minutes first time)
docker build -t qa-signal-experiment:latest .

# Verify it was created
docker images | grep qa-signal
```

**Step 2: Run the container**
```bash
# Run experiment in container
docker run --rm -v $(pwd)/results:/app/results qa-signal-experiment:latest

# Check output
ls -lh results/
# Should see: signal_classification_results.png
```

**Explanation:**
- `--rm`: Auto-remove container when done
- `-v $(pwd)/results:/app/results`: Mount results directory
- Container runs `run_signal_experiments_final.py` automatically

**Step 3: Interactive exploration**
```bash
# Run container with bash shell
docker run -it --rm qa-signal-experiment:latest bash

# Inside container, you can:
python run_signal_experiments_final.py
ls -la
exit
```

---

## Week 4: Multi-Container Setup with Docker Compose

### What is Docker Compose?

**docker-compose** lets you define and run **multiple containers** together. Perfect for:
- Your experiments + Jupyter notebook server
- Your experiments + PostgreSQL database
- Your experiments + Redis queue

### Create docker-compose.yml

**File**: `docker-compose.yml` (already created below)

```yaml
version: '3.8'

services:
  # Main experiment runner
  qa-experiment:
    build: .
    image: qa-signal-experiment:latest
    volumes:
      - ./results:/app/results
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    command: python run_signal_experiments_final.py

  # Jupyter notebook server
  jupyter:
    image: jupyter/scipy-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - .:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes

  # PostgreSQL for storing results
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: qa_research
      POSTGRES_USER: qa_user
      POSTGRES_PASSWORD: change_this_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Redis for task queue (Phase 4)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres-data:
```

### Use Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Start just Jupyter
docker-compose up -d jupyter

# Access Jupyter
# Look for token in logs: docker-compose logs jupyter
# Open: http://localhost:8888/?token=<token>
```

---

## Week 5-6: Containerize All Experiments

### Goal: Docker image for each experiment type

**Create specialized Dockerfiles:**

```
dockerfiles/
├── Dockerfile.signal          # Signal processing
├── Dockerfile.belltest         # Bell test validation
├── Dockerfile.gnn              # Theorem generation (GNN)
├── Dockerfile.financial        # Backtesting
├── Dockerfile.quartz           # Quartz simulations
└── Dockerfile.multimodal       # Hyperspectral processing
```

### Build all images

```bash
# Signal processing
docker build -f dockerfiles/Dockerfile.signal -t qa-signal:latest .

# Bell tests
docker build -f dockerfiles/Dockerfile.belltest -t qa-belltest:latest .

# GNN (requires GPU support)
docker build -f dockerfiles/Dockerfile.gnn -t qa-gnn:latest .

# Financial backtesting
docker build -f dockerfiles/Dockerfile.financial -t qa-financial:latest .

# List all images
docker images | grep qa-
```

### Run experiments via Docker

```bash
# Signal processing
docker run --rm -v $(pwd)/results:/app/results qa-signal:latest

# Bell tests
docker run --rm -v $(pwd)/results:/app/results qa-belltest:latest

# GNN (with GPU)
docker run --rm --gpus all -v $(pwd)/results:/app/results qa-gnn:latest

# Financial backtesting
docker run --rm -v $(pwd)/results:/app/results qa-financial:latest
```

---

## Advanced Topics (Optional)

### GPU Support for PyTorch

**Install NVIDIA Container Toolkit:**
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Docker BuildKit (Faster Builds)

```bash
# Enable BuildKit (faster, better caching)
export DOCKER_BUILDKIT=1

# Or add to ~/.bashrc
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
```

### Multi-Stage Builds (Smaller Images)

```dockerfile
# Build stage
FROM python:3.13 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage (smaller final image)
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "run_signal_experiments_final.py"]
```

---

## Troubleshooting

### "permission denied" when running docker

**Fix**: Add user to docker group
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Build fails with "cannot connect to Docker daemon"

**Fix**: Start Docker service
```bash
sudo systemctl start docker
sudo systemctl enable docker  # Start on boot
```

### Container exits immediately

**Debug**: Check logs
```bash
docker ps -a  # Find container ID
docker logs <container-id>
```

### Out of disk space

**Clean up**:
```bash
# Remove unused images
docker image prune -a

# Remove unused containers
docker container prune

# Remove unused volumes
docker volume prune

# Nuclear option (removes everything)
docker system prune -a --volumes
```

### Can't connect to Jupyter

**Fix**: Check token in logs
```bash
docker-compose logs jupyter | grep token
# Copy the token, open browser:
# http://localhost:8888/?token=<token>
```

---

## Learning Resources

**Official Docker Docs:**
- Tutorial: https://docs.docker.com/get-started/
- Dockerfile reference: https://docs.docker.com/engine/reference/builder/
- Compose file reference: https://docs.docker.com/compose/compose-file/

**Video Tutorials:**
- Docker in 100 Seconds: https://www.youtube.com/watch?v=Gjnup-PuquQ
- Network Chuck Docker series: https://www.youtube.com/c/NetworkChuck

**Practice:**
- Docker Hub (find images): https://hub.docker.com/
- Play with Docker (free online lab): https://labs.play-with-docker.com/

---

## Success Metrics for Phase 2

By end of Week 6, you should have:

- [x] Docker installed and working
- [x] Completed official tutorial
- [x] Built first Dockerfile for signal processing
- [x] Created docker-compose.yml with 4 services
- [x] Containerized at least 3 different experiments
- [ ] All 70+ scripts Dockerized (stretch goal)
- [ ] Jupyter notebook server running
- [ ] PostgreSQL connected and storing results

---

## Next: Phase 3 - Kubernetes (Weeks 7-12)

Once you're comfortable with Docker, we'll move to **Kubernetes** for:
- Container orchestration across multiple machines
- Auto-scaling based on load
- Self-healing (containers restart on failure)
- Load balancing
- Service discovery

**Prerequisites for Phase 3:**
- Comfortable with Docker CLI
- Understand docker-compose
- Can build and run multi-container apps
- Familiar with volumes and networking

---

**Ready to start? Install Docker and run the hello-world container!**

See you in Week 3! 🐳
