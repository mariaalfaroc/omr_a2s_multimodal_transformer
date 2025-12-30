FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install ALL system dependencies in ONE layer (combine apt runs)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    vim \
    fluidsynth \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv FIRST (before copying files)
RUN pip install --no-cache-dir --upgrade pip uv

# Copy project files first (for better layer caching)
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY grandstaff/ ./grandstaff/
COPY run_experiments.sh ./

# Create virtual environment and sync dependencies (creates .venv automatically)
RUN uv sync --frozen  # Uses uv.lock for reproducible installs

# Docker-only: pybind11 (not in local pyproject.toml)
RUN uv pip install pybind11

# Activate virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["/bin/bash"]
