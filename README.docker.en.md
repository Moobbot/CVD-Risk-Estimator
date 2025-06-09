# Docker Deployment Guide

This guide provides quick instructions for deploying the CVD Risk Estimator using Docker. For detailed deployment documentation, see [Deployment Guide](docs/deployment.md).

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA drivers installed (for GPU support)

## Quick Start

### With GPU

```bash
# Build and start the container with GPU
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Without GPU

```bash
# Use the CPU service in docker-compose.yml
docker-compose up -d app-cpu

# View logs
docker-compose logs -f cvd-risk-estimator-cpu

# Stop the container
docker-compose down
```

## Docker Configuration

The Docker configuration includes:

- Python 3.10 base image
- Required system libraries (ffmpeg, libsm6, libxext6)
- Virtual environment for clean dependency management
- Optimized image size using multi-stage builds
- Non-root user for improved security
- Volume mounts for persistent data storage
- Environment variable configuration
- GPU support using NVIDIA Container Toolkit

For more detailed information about Docker deployment, configuration, and troubleshooting, please refer to the [Deployment Guide](docs/deployment.md).
