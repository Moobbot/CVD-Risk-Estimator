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

The current image (`Dockerfile`):

- Uses the `python:3.10-slim` base image and runs as root
- Starts with a `torch-base` stage (system libraries OpenCV needs, torch 2.5.1 + torchvision 0.20.1
  with CUDA 12.1) that is identical in Sybil's Dockerfile: built together, the two images share it
- Installs the Python dependencies from `requirements.txt`, then downloads the two checkpoints into
  `/app/checkpoint` with `python setup.py --skip-packages`
- Is configured through environment variables
- Supports the GPU through the NVIDIA Container Toolkit

Persistent data lives in volume mounts (see [Volumes](#volumes)).

For more detailed information about Docker deployment, configuration, and troubleshooting, please refer to the [Deployment Guide](docs/deployment.md).

## Main Features

- Cardiovascular disease risk prediction from DICOM images
- Automatic heart region detection with RetinaNet, or a simple fallback method
- Grad-CAM images to explain the result
- Animated GIF built directly from the Grad-CAM images
- Logs organised by year/month/day
- Automatic switch to CPU mode when no GPU is available
- Model loading optimised during startup

## Configuration

### Environment variables

You can configure the application by editing the `.env` file or through the environment variables in `docker-compose.yml`:

```yaml
environment:
  - ENV=prod
  - HOST_CONNECT=0.0.0.0
  - PORT=5556
  - CUDA_VISIBLE_DEVICES=0
  - DEVICE=cuda
```

### Volumes

The following volumes keep data between container runs:

- `./checkpoint:/app/checkpoint`: downloaded models
- `./logs:/app/logs`: application logs (organised by year/month/day)
- `./uploads:/app/uploads`: temporary uploaded files
- `./results:/app/results`: prediction results and GIF files
- `./.env:/app/.env`: environment configuration file

#### Log layout

Logs are organised automatically by year/month, with file names based on the date:

```plaintext
logs/
├── 2023/
│   ├── 01/
│   │   ├── api_2023-01-01.log
│   │   ├── api_2023-01-02.log
│   │   └── ...
│   └── ...
└── ...
```

This layout makes it easy to find the logs for a given day and keeps log files from growing too large.

## Troubleshooting

### The GPU cannot be used

If you get GPU-related errors, make sure that:

1. The NVIDIA Container Toolkit is installed correctly
2. The NVIDIA driver is installed and working
3. `nvidia-smi` works

The Docker Compose file is already configured to access the GPU using the modern Docker Compose format:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

If the problem persists, you can switch to CPU mode by:

1. Using the CPU service in docker-compose.yml: `docker-compose up -d app-cpu`
2. Or setting `DEVICE=cpu` in the environment variables of an existing container

Note that CPU mode is much slower for inference, but lets the application run on any machine without a GPU.

### Errors while loading the model

If you get errors while loading the model, make sure that:

1. The model files have been downloaded into the `checkpoint` folder
2. The `checkpoint` folder is mounted correctly into the container

## Performance

To improve performance, you can:

1. Increase `BATCH_SIZE` if there is enough GPU memory
2. Use `--shm-size` to increase shared memory when running the container

## Production Deployment

When deploying to production, make sure to:

1. Set `ENV=prod` to disable debug features
2. Configure `CORS_ORIGINS` to allow only trusted origins
3. Use a reverse proxy such as Nginx for HTTPS and load balancing
