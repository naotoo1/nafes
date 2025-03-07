FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Upgrade pip and setuptools first
RUN pip3 install --no-cache-dir --upgrade pip setuptools>=61

# Copy the entire project
COPY . /app

# Copy Docker-specific frozen requirements
COPY requirements-frozen-docker.txt /app/requirements-frozen.txt

# Install dependencies from frozen requirements
RUN pip3 install --no-cache-dir -r requirements-frozen.txt

# Install the local package (not in editable mode)
RUN pip3 install --no-cache-dir -e .

# Set environment variables for PyTorch
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

# Default command
CMD ["python3"]
