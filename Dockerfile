# Dockerfile for QA Signal Processing Experiment
# This containerizes run_signal_experiments_final.py

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (we'll create this next)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment code
COPY run_signal_experiments_final.py .
COPY qa_core.py .

# Create output directory
RUN mkdir -p /app/results

# Run the experiment when container starts
CMD ["python", "run_signal_experiments_final.py"]

# Alternative: Interactive mode
# CMD ["bash"]
