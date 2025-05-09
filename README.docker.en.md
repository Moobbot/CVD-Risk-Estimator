# Docker Guide for CVD Risk Estimator

This document provides instructions on how to use Docker to deploy the CVD Risk Estimator application in both GPU and non-GPU environments.

## Key Features

- Predict cardiovascular disease risk from DICOM images
- Detect heart region automatically using RetinaNet or simple method
- Generate Grad-CAM visualizations for explainability
- Create animated GIFs directly from Grad-CAM visualizations
- Organize logs in a year/month/day structure
- Automatically fall back to CPU mode when GPU is not available
- Optimize model loading during application startup

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
# Use the CPU service in docker-compose.yml
docker-compose up -d cvd-risk-estimator-cpu

# View logs
docker-compose logs -f cvd-risk-estimator-cpu

# Stop the container
docker-compose down
```

### 2. Build and Run with Docker

If you don't use Docker Compose, you can use Docker commands directly:

#### Using GPU

```bash
# Build the image
docker build -t cvd-risk-estimator .

# Run the container with GPU
docker run --gpus all -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env --name cvd-risk-estimator -d cvd-risk-estimator
```

#### Using CPU

```bash
# Build the image (same image as GPU)
docker build -t cvd-risk-estimator .

# Run the container with CPU (specify DEVICE=cpu environment variable)
docker run -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env -e DEVICE=cpu -e CUDA_VISIBLE_DEVICES= --name cvd-risk-estimator-cpu -d cvd-risk-estimator
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
- `./logs:/app/logs`: Stores application logs (organized by year/month/day)
- `./uploads:/app/uploads`: Stores temporary uploaded files
- `./results:/app/results`: Stores prediction results and GIF files
- `./.env:/app/.env`: Environment configuration file

#### Log Organization

Logs are automatically organized in a year/month structure with date-based filenames:

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

This organization makes it easy to find logs for specific dates and prevents log files from growing too large.

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

The Dockerfile has been optimized to reduce image size by:

1. Using multi-stage builds to separate build and runtime environments
2. Using slim base images to reduce size
3. Installing only necessary packages and cleaning apt cache after installation
4. Adding large files to `.dockerignore`

### Security

The application has improved security by:

1. Running the application as a non-root user (appuser)
2. Installing only necessary packages for runtime
3. Using minimal permissions for directories

### Improving Performance

To improve performance, you can:

1. Increase the `BATCH_SIZE` value if you have enough GPU memory
2. Use `--shm-size` to increase shared memory when running the container

## Production Deployment

When deploying in a production environment, ensure:

1. Set `ENV=prod` to disable debug features
2. Configure `CORS_ORIGINS` to only allow trusted origins
3. Use a reverse proxy like Nginx to handle HTTPS and load balancing
