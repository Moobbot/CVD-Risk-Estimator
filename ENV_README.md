# Configuration Options

This file provides a quick reference for the available configuration options. For detailed configuration documentation, see [Configuration Guide](docs/configuration.md).

## Quick Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Application environment (dev, test, prod) | `dev` |
| `PORT` | Port number for the server | `5556` |
| `DEVICE` | Device to use for model inference (cuda, cpu) | `cuda` if available, auto-fallback to `cpu` |
| `CUDA_VISIBLE_DEVICES` | CUDA device indices to use | `0` (set empty to force CPU) |
| `MODEL_ITER` | Model iteration checkpoint to load | `700` |
| `LOG_LEVEL` | Logging level | `DEBUG` in dev, `INFO` otherwise |
| `CLEANUP_ENABLED` | Enable automatic cleanup of old files | `true` |

## Example .env File

```env
# Environment
ENV=dev
PORT=5556

# Device Configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Model Configuration
MODEL_ITER=700

# Logging
LOG_LEVEL=DEBUG

# Cleanup
CLEANUP_ENABLED=true
```

For more detailed information about each configuration option and advanced usage, please refer to the [Configuration Guide](docs/configuration.md).

## Configuration System

The application uses a flexible configuration system with three layers:

1. **Default Values**: Defined directly in `config.py`
2. **`.env` File**: Optional file for overriding defaults
3. **Environment Variables**: Highest priority, override both defaults and `.env` values

## Setup Options

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

## Available Configuration Options

All configuration options have default values defined directly in `config.py`. You can override these using environment variables or the `.env` file.

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

The application uses a date-based logging system that automatically organizes logs by year and month in the following structure:

```plaintext
logs/
├── 2023/
│   ├── 01/
│   │   ├── api_2023-01-01.log
│   │   ├── api_2023-01-02.log
│   │   └── ...
│   ├── 02/
│   │   └── ...
│   └── ...
└── ...
```

This organization makes it easy to find logs for specific dates and prevents log files from growing too large. The application automatically creates new log files for each day and rotates them when they exceed the configured size limit.

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