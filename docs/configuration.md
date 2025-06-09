# Configuration Guide

## Configuration System

The application uses a flexible configuration system with three layers:

1. **Default Values**: Defined directly in `config.py`
2. **`.env` File**: Optional file for overriding defaults
3. **Environment Variables**: Highest priority, override both defaults and `.env` values

## Available Configuration Options

### Environment Configuration

- `ENV`: Application environment (dev, test, prod)
  - Default: `dev`

### Server Configuration

- `HOST_CONNECT`: Host IP address to bind the server
  - Default: `0.0.0.0` (all interfaces)
- `PORT`: Port number for the server
  - Default: `5556`

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
  - The application automatically detects if CUDA is available and falls back to CPU if needed
  - Set to `cpu` to force CPU mode even if CUDA is available
- `CUDA_VISIBLE_DEVICES`: CUDA device indices to use
  - Default: `0`
  - Set to empty string to force CPU mode: `CUDA_VISIBLE_DEVICES=`
  - Set to specific device index for multi-GPU systems: `CUDA_VISIBLE_DEVICES=1`
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
  - Default: `3`
- `CLEANUP_MAX_AGE_DAYS`: Maximum age of files to keep in days
  - Default: `1`

### Security Configuration

- `ALLOWED_IPS`: Comma-separated list of allowed IPs or CIDR notations
  - Default: `127.0.0.1,192.168.1.0/24,10.0.0.0/8`
- `CORS_ORIGINS`: Comma-separated list of allowed origins for CORS
  - Default: `*` in dev environment, `https://example.com` otherwise

## Example .env File

```env
# Environment Configuration
ENV=dev

# Server Configuration
HOST_CONNECT=0.0.0.0
PORT=5556

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
CLEANUP_INTERVAL_HOURS=3
CLEANUP_MAX_AGE_DAYS=1

# Security Configuration
ALLOWED_IPS=127.0.0.1,192.168.1.0/24,10.0.0.0/8
CORS_ORIGINS=*
```

## Configuration Methods

### Option 1: Using Environment Variables

Set environment variables directly in your system or container:

```bash
# Windows
set PORT=5556
python api.py

# Linux/macOS
PORT=5556 python api.py
```

### Option 2: Using a .env File

1. Copy the `.env.example` file to create a new `.env` file:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to customize your environment settings.

3. Make sure the `python-dotenv` package is installed:

   ```bash
   pip install python-dotenv
   ```

## Important Notes

### Comments in .env Files

When adding comments to your environment variables in a `.env` file, make sure to place them on a separate line **above** the variable, not on the same line. For example:

```plaintext
# Correct: Comment on a separate line
# This is the port number
PORT=5556

# Incorrect: Comment on the same line
PORT=5556  # This is the port number
```

### Configuration Precedence

The configuration system follows this precedence order (highest to lowest):

1. Environment variables set in the system or container
2. Values in the `.env` file (if it exists)
3. Default values defined in `config.py`

### GPU/CPU Configuration

The application includes an intelligent device selection system:

1. **Automatic Detection**: The system checks if CUDA is available during startup
2. **Configuration-Based Selection**: Uses the `DEVICE` and `CUDA_VISIBLE_DEVICES` settings
3. **Fallback Mechanism**: Automatically falls back to CPU if CUDA is not available

#### Forcing CPU Mode

You can force CPU mode in several ways:

```bash
# Method 1: Set DEVICE to cpu
DEVICE=cpu python api.py

# Method 2: Set CUDA_VISIBLE_DEVICES to empty
CUDA_VISIBLE_DEVICES= python api.py

# Method 3: In .env file
DEVICE=cpu
```

#### Forcing GPU Mode

To explicitly use GPU (will fail if CUDA is not available):

```bash
DEVICE=cuda python api.py
```

#### Multi-GPU Systems

On systems with multiple GPUs, you can select a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python api.py  # Use second GPU
```
