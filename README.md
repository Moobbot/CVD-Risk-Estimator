# CVD Risk Estimator

API for predicting cardiovascular disease risk from DICOM images using Tri2D-Net model.

## Features

- Upload and process DICOM images in ZIP format
- Detect heart region using RetinaNet or simple method
- Predict CVD risk using Tri2D-Net model
- Generate Grad-CAM visualizations for explainability
- Create animated GIFs from Grad-CAM visualizations directly during image processing
- Attention score calculation for important slices
- Environment variable configuration for easy deployment
- Unicode support for logging in multiple languages
- Date-based log organization with automatic rotation
- Automatic CPU fallback when GPU is not available
- Docker containerization with GPU and CPU support
- Optimized model loading during application startup

## Documentation

Detailed documentation is available in the `docs` directory:

- [System Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)

## Quick Start

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

4. Run the API:

   ```bash
   python api.py
   ```

The API will automatically find an available port (default: 5556) and display the URLs:

```plaintext
Running on: http://127.0.0.1:5556 (localhost)
Running on: http://192.168.x.x:5556 (local network)
```

## Docker Deployment

For Docker deployment instructions, see:

- [English](README.docker.en.md)
- [Vietnamese](README.docker.md)

## Configuration

For detailed configuration options, see [Configuration Guide](docs/configuration.md).

## API Usage

For detailed API documentation and examples, see [API Documentation](docs/api.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
