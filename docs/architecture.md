# System Architecture

## Core Components

1. **API Layer (`api.py`)**
   - FastAPI-based REST API
   - Handles HTTP requests and responses
   - Manages application lifecycle
   - Provides static file serving

2. **Model Layer**
   - `call_model.py`: Model loading and inference
   - `heart_detector.py`: Heart region detection using RetinaNet
   - `tri_2d_net/`: Tri2D-Net model implementation

3. **Processing Layer**
   - `image.py`: DICOM image processing
   - `utils.py`: Utility functions
   - `bbox_cut.py`: Bounding box processing

4. **Configuration**
   - `config.py`: Application configuration
   - Environment variables support
   - Docker configuration

## Model Architecture

### Tri2D-Net Model

- **Purpose**: Cardiovascular disease risk prediction
- **Input**: DICOM images of heart region
- **Output**: Risk score (0-1)
- **Features**:
  - 2D convolutional neural network
  - Attention mechanism
  - Grad-CAM visualization support
  - Multi-slice analysis

### Heart Detector (RetinaNet)

- **Purpose**: Heart region detection in DICOM images
- **Features**:
  - Object detection
  - Bounding box prediction
  - Confidence scoring
  - Fallback to simple detection method

## Directory Structure

```
CVD-Risk-Estimator/
├── api.py                 # Main API entry point
├── routes.py             # API route definitions
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── image.py              # Image processing
├── call_model.py         # Model inference
├── heart_detector.py     # Heart detection
├── logger.py             # Logging configuration
├── setup.py              # Installation script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker compose setup
├── uploads/              # Temporary upload storage
├── results/              # Processing results
├── logs/                 # Application logs
├── checkpoint/           # Model checkpoints
└── tri_2d_net/           # Model implementation
```
