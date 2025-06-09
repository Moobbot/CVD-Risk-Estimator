# Deployment Guide

## Local Deployment

### Prerequisites

- Python 3.10 or higher
- CUDA Toolkit (for GPU support)
- Virtual environment (recommended)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/CVD-Risk-Estimator.git
   cd CVD-Risk-Estimator
   ```

2. Create and activate a virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python setup.py
   ```

4. (Optional) Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. Run the API:

   ```bash
   python api.py
   ```

## Docker Deployment

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Quick Start

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
# Use the CPU service
docker-compose up -d app-cpu

# View logs
docker-compose logs -f cvd-risk-estimator-cpu

# Stop the container
docker-compose down
```

### Docker Configuration

The Docker configuration includes:

- Python 3.10 base image
- Required system libraries
- Virtual environment for clean dependency management
- Optimized image size using multi-stage builds
- Non-root user for improved security
- Volume mounts for persistent data storage
- Environment variable configuration
- GPU support using NVIDIA Container Toolkit

### Docker Volumes

The following volumes are used to store data between container runs:

- `./checkpoint:/app/checkpoint`: Stores downloaded models
- `./logs:/app/logs`: Stores application logs
- `./uploads:/app/uploads`: Stores temporary uploaded files
- `./results:/app/results`: Stores prediction results
- `./.env:/app/.env`: Environment configuration file

### GPU Configuration

The Docker Compose file is configured to properly access the GPU using the modern Docker Compose format:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Production Deployment

### Security Considerations

1. Set `ENV=prod` to disable debug features
2. Configure `CORS_ORIGINS` to only allow trusted origins
3. Use a reverse proxy like Nginx to handle HTTPS and load balancing
4. Set appropriate file permissions
5. Configure firewall rules

### Performance Optimization

1. Increase `BATCH_SIZE` if you have enough GPU memory
2. Use `--shm-size` to increase shared memory when running the container
3. Configure appropriate log rotation
4. Set up monitoring and alerting

### Monitoring

1. Monitor log files
2. Track GPU memory usage
3. Monitor disk space usage
4. Set up alerts for errors

### Backup Strategy

1. Regular backup of model checkpoints
2. Backup of configuration files
3. Backup of important logs
4. Document recovery procedures

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   - Check NVIDIA drivers
   - Verify NVIDIA Container Toolkit installation
   - Check GPU visibility with `nvidia-smi`

2. **Model Loading Errors**
   - Verify model files in checkpoint directory
   - Check file permissions
   - Verify CUDA compatibility

3. **Memory Issues**
   - Reduce batch size
   - Monitor GPU memory usage
   - Check for memory leaks

4. **Performance Issues**
   - Check GPU utilization
   - Monitor CPU usage
   - Verify disk I/O performance

### Log Analysis

Logs are organized by date in the following structure:

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

Check logs for:

- Error messages
- Performance metrics
- Resource usage
- Security events

## Maintenance

### Regular Tasks

1. Monitor log files
2. Clean up old sessions
3. Update model checkpoints
4. Verify GPU memory usage
5. Check disk space
6. Update dependencies

### Backup Procedures

1. Regular backup of:
   - Model checkpoints
   - Configuration files
   - Important logs
   - User data

2. Document recovery procedures for:
   - System failure
   - Data corruption
   - Configuration issues

### Update Procedures

1. Pull latest code
2. Update dependencies
3. Test in staging environment
4. Deploy to production
5. Monitor for issues
