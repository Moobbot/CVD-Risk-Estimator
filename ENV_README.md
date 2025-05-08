# Environment Variables Configuration

This document explains how to configure the application using environment variables.

## Setup

1. Copy the `.env.example` file to create a new `.env` file:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to customize your environment settings.

3. Make sure the `python-dotenv` package is installed:

```bash
pip install python-dotenv
```

## Available Environment Variables

### Environment Configuration

- `ENV`: Application environment (dev, test, prod)
  - Default: `dev`

### Server Configuration

- `HOST_CONNECT`: Host IP address to bind the server
  - Default: `0.0.0.0` (all interfaces)
- `PORT`: Port number for the server
  - Default: `8080`

### File Configuration

- `MAX_FILE_SIZE`: Maximum allowed file size in bytes
  - Default: `104857600` (100MB)
- `FILE_RETENTION`: File retention period in hours
  - Default: `1`

### Model Configuration

- `BATCH_SIZE`: Batch size for model inference
  - Default: `16`
- `DEVICE`: Device to use for model inference (cuda, cpu)
  - Default: `cuda` if CUDA is available, otherwise `cpu`
- `CUDA_VISIBLE_DEVICES`: CUDA device indices to use
  - Default: `0`
- `MODEL_ITER`: Model iteration checkpoint to load
  - Default: `700`

### Logging Configuration

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Default: `DEBUG` in dev environment, `INFO` otherwise
- `LOG_MAX_BYTES`: Maximum log file size in bytes before rotation
  - Default: `10485760` (10MB)
- `LOG_BACKUP_COUNT`: Number of backup log files to keep
  - Default: `5`

### Cleanup Configuration

- `CLEANUP_ENABLED`: Enable automatic cleanup of old files
  - Default: `true`
- `CLEANUP_INTERVAL_HOURS`: Interval between cleanup runs in hours
  - Default: `24`
- `CLEANUP_MAX_AGE_DAYS`: Maximum age of files to keep in days
  - Default: `7`

### Security Configuration

- `ALLOWED_IPS`: Comma-separated list of allowed IPs or CIDR notations
  - Default: `127.0.0.1,192.168.1.0/24,10.0.0.0/8`
- `CORS_ORIGINS`: Comma-separated list of allowed origins for CORS
  - Default: `*` in dev environment, `https://example.com` otherwise

## Example .env File

```.env
# Environment Configuration
ENV=dev

# Server Configuration
HOST_CONNECT=0.0.0.0
PORT=8080

# File Configuration
MAX_FILE_SIZE=104857600
FILE_RETENTION=1

# Model Configuration
BATCH_SIZE=16
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
MODEL_ITER=700

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5

# Cleanup Configuration
CLEANUP_ENABLED=true
CLEANUP_INTERVAL_HOURS=24
CLEANUP_MAX_AGE_DAYS=7

# Security Configuration
ALLOWED_IPS=127.0.0.1,192.168.1.0/24,10.0.0.0/8
CORS_ORIGINS=*
```
