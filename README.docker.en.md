# Docker Guide for CVD Risk Estimator

This document provides instructions on how to use Docker to deploy the CVD Risk Estimator application in both GPU and non-GPU environments.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (only when using GPU)

## Usage

### 1. Build and Run with Docker Compose (Recommended)

Docker Compose makes container management easier and allows configuration through the `docker-compose.yml` file.

#### With GPU

```bash
# Build and start the container with GPU
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

#### Without GPU

```bash
# Use the special docker-compose.cpu.yml file for non-GPU environments
docker-compose -f docker-compose.cpu.yml up -d

# View logs
docker-compose -f docker-compose.cpu.yml logs -f

# Stop the container
docker-compose -f docker-compose.cpu.yml down
```

> **Note**: The CPU version uses a separate Dockerfile.cpu that has been optimized for non-GPU environments.

### 2. Build and Run with Docker

If you don't use Docker Compose, you can use Docker commands directly:

#### Using GPU

```bash
# Build the image for GPU
docker build --build-arg USE_GPU=true -t cvd-risk-estimator-gpu .

# Run the container with GPU
docker run --gpus all -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env --name cvd-risk-estimator -d cvd-risk-estimator-gpu
```

#### Using CPU

```bash
# Build the image for CPU using Dockerfile.cpu
docker build -f Dockerfile.cpu -t cvd-risk-estimator-cpu .

# Run the container with CPU
docker run -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env --name cvd-risk-estimator -d cvd-risk-estimator-cpu
```

## Configuration

### Environment Variables

You can configure the application by editing the `.env` file or through environment variables in `docker-compose.yml`:

```yaml
environment:
  - ENV=prod
  - HOST_CONNECT=0.0.0.0
  - PORT=5556
  - CUDA_VISIBLE_DEVICES=0
  - DEVICE=cuda
  - CUDA_VERSION=12.1
```

### Volumes

The following volumes are used to store data between container runs:

- `./checkpoint:/app/checkpoint`: Stores downloaded models
- `./logs:/app/logs`: Stores application logs
- `./uploads:/app/uploads`: Stores temporary uploaded files
- `./results:/app/results`: Stores prediction results
- `./.env:/app/.env`: Environment configuration file

## Troubleshooting

### Cannot Use GPU

If you encounter GPU-related errors, ensure:

1. NVIDIA Container Toolkit is properly installed
2. NVIDIA drivers are installed and working
3. Check with the `nvidia-smi` command

If you still have issues, you can switch to CPU mode by:

1. Using the CPU-specific Docker Compose file: `docker-compose -f docker-compose.cpu.yml up -d`
2. Or building with Dockerfile.cpu: `docker build -f Dockerfile.cpu -t cvd-risk-estimator-cpu .`
3. Or setting `DEVICE=cpu` in the environment variables when running an existing container

Note that CPU mode will be significantly slower for inference but allows the application to run on any machine without GPU requirements.

### Model Loading Errors

If you encounter errors when loading models, ensure:

1. Model files are downloaded to the `checkpoint` directory
2. The `checkpoint` directory is properly mounted to the container

## Optimization

### Reducing Image Size

To reduce Docker image size, you can:

1. Add large files to `.dockerignore`
2. Use multi-stage builds
3. Clean up unnecessary packages after installation

### Improving Performance

To improve performance, you can:

1. Increase the `BATCH_SIZE` value if you have enough GPU memory
2. Use `--shm-size` to increase shared memory when running the container

## Production Deployment

When deploying in a production environment, ensure:

1. Set `ENV=prod` to disable debug features
2. Configure `CORS_ORIGINS` to only allow trusted origins
3. Use a reverse proxy like Nginx to handle HTTPS and load balancing
4. Set up monitoring and alerts
