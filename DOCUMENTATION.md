# CVD Risk Estimator Documentation

## Project Overview

The CVD Risk Estimator is a Python-based API service that predicts cardiovascular disease risk from DICOM medical images using the Tri2D-Net model. The system processes DICOM images, detects heart regions, and provides risk predictions along with visual explanations using Grad-CAM.

### Technology Stack

- **Backend Framework**: FastAPI (Python)
- **Image Processing**: SimpleITK, OpenCV
- **Deep Learning**: PyTorch
- **Model Architecture**: Tri2D-Net with RetinaNet
- **Containerization**: Docker
- **GPU Support**: CUDA

## System Architecture

### Core Components

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

### Model Architecture

#### Tri2D-Net Model

- **Purpose**: Cardiovascular disease risk prediction
- **Input**: DICOM images of heart region
- **Output**: Risk score (0-1)
- **Features**:
  - 2D convolutional neural network
  - Attention mechanism
  - Grad-CAM visualization support
  - Multi-slice analysis

#### Heart Detector (RetinaNet)

- **Purpose**: Heart region detection in DICOM images
- **Features**:
  - Object detection
  - Bounding box prediction
  - Confidence scoring
  - Fallback to simple detection method

### Directory Structure

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

## API Endpoints

### 1. Prediction Endpoints

#### POST `/api_predict`

- **Purpose**: Process existing DICOM files in a session
- **Input**:

  ```json
  {
    "session_id": "string"
  }
  ```

- **Output**:

  ```json
  {
    "session_id": "string",
    "predictions": [
      {
        "score": float
      }
    ],
    "attention_info": {
      "attention_scores": [
        {
          "file_name_pred": "string",
          "attention_score": float
        }
      ],
      "total_images": int,
      "returned_images": int
    },
    "message": "string"
  }
  ```

#### POST `/api_predict_zip`

- **Purpose**: Upload and process new DICOM files
- **Input**:
  - `file`: ZIP file containing DICOM images (multipart/form-data)
- **Output**:

  ```json
  {
    "session_id": "string",
    "predictions": [
      {
        "score": float
      }
    ],
    "overlay_images": "string (URL)",
    "overlay_gif": "string (URL)",
    "attention_info": {
      "attention_scores": [
        {
          "file_name_pred": "string",
          "attention_score": float
        }
      ],
      "total_images": int,
      "returned_images": int
    },
    "message": "string"
  }
  ```

### 2. Download Endpoints

#### GET `/download_zip/{session_id}`

- **Purpose**: Download ZIP with overlay images
- **Parameters**:
  - `session_id`: Unique session identifier
- **Response**: ZIP file containing overlay images

#### GET `/download_gif/{session_id}`

- **Purpose**: Download animated GIF visualization
- **Parameters**:
  - `session_id`: Unique session identifier
- **Response**: GIF file with animated Grad-CAM visualizations

### 3. Preview Endpoint

#### GET `/preview/{session_id}/{filename}`

- **Purpose**: Preview specific overlay image
- **Parameters**:
  - `session_id`: Unique session identifier
  - `filename`: Name of the file to preview
- **Response**: Image file

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Application environment | `dev` |
| `PORT` | Server port | `5556` |
| `DEVICE` | Inference device | `cuda`/`cpu` |
| `CUDA_VISIBLE_DEVICES` | GPU device indices | `0` |
| `MODEL_ITER` | Model checkpoint | `700` |
| `LOG_LEVEL` | Logging level | `DEBUG`/`INFO` |
| `CLEANUP_ENABLED` | Auto cleanup | `true` |

### Configuration Methods

1. Environment Variables
2. `.env` file
3. Default values in `config.py`

## Setup and Installation

### Local Setup

1. Clone repository:

   ```bash
   git clone https://github.com/yourusername/CVD-Risk-Estimator.git
   cd CVD-Risk-Estimator
   ```

2. Create virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   python setup.py
   ```

4. Run the API:

   ```bash
   python api.py
   ```

### Docker Setup

1. Build and run with Docker Compose:

   ```bash
   docker-compose up --build
   ```

2. For GPU support:

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
   ```

## Usage Examples

### Python Client

```python
import requests

# Make prediction with ZIP file
url = "http://localhost:5556/api_predict_zip"
files = {"file": ("dicom_images.zip", open("path/to/dicom_images.zip", "rb"))}
response = requests.post(url, files=files)
result = response.json()

# Make prediction with existing session
url = "http://localhost:5556/api_predict"
data = {"session_id": "existing-session-id"}
response = requests.post(url, json=data)
result = response.json()

# Download results
session_id = result["session_id"]
download_url = f"http://localhost:5556/download_zip/{session_id}"
download_response = requests.get(download_url)
with open("results.zip", "wb") as f:
    f.write(download_response.content)

# Download GIF if available
if result.get("overlay_gif"):
    gif_url = result["overlay_gif"]
    gif_response = requests.get(gif_url)
    with open("results.gif", "wb") as f:
        f.write(gif_response.content)

# Preview specific image
image_name = result["attention_info"]["attention_scores"][0]["file_name_pred"]
preview_url = f"http://localhost:5556/preview/{session_id}/{image_name}"
preview_response = requests.get(preview_url)
with open("preview.png", "wb") as f:
    f.write(preview_response.content)
```

### cURL Client

```bash
# Make prediction with ZIP file
curl -X POST "http://localhost:5556/api_predict_zip" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/dicom_images.zip"

# Make prediction with existing session
curl -X POST "http://localhost:5556/api_predict" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "existing-session-id"}'

# Download results
curl -X GET "http://localhost:5556/download_zip/{session_id}" \
     -o results.zip

# Download GIF
curl -X GET "http://localhost:5556/download_gif/{session_id}" \
     -o results.gif

# Preview specific image
curl -X GET "http://localhost:5556/preview/{session_id}/{filename}" \
     -o preview.png
```

## Error Handling

The API implements comprehensive error handling:

- Input validation
- File processing errors
- Model inference errors
- Resource cleanup
- Detailed error logging

### Common Error Responses

```json
{
  "error": "Missing session_id"
}
```

```json
{
  "error": "Invalid file format. Only ZIP is allowed."
}
```

```json
{
  "error": "No valid files found in the ZIP archive"
}
```

```json
{
  "error": "Processing error: {error_message}"
}
```

## Logging

- Date-based log organization
- Automatic log rotation
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Unicode support for internationalization

## Performance Considerations

1. **Model Loading**
   - Optimized during startup
   - Automatic CPU fallback
   - GPU memory management

2. **File Processing**
   - Automatic cleanup of old files
   - Efficient DICOM processing
   - Optimized image visualization

3. **Resource Management**
   - Automatic port selection
   - Memory-efficient processing
   - Temporary file cleanup

## Security

1. **CORS Configuration**
   - Configurable origins
   - Method restrictions
   - Header controls

2. **File Handling**
   - Secure file uploads
   - Input validation
   - Temporary file management

## Maintenance

### Regular Tasks

1. Monitor log files
2. Clean up old sessions
3. Update model checkpoints
4. Verify GPU memory usage

### Troubleshooting

1. Check application logs
2. Verify model loading
3. Monitor system resources
4. Validate file permissions

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request
4. Follow coding standards
5. Update documentation

## License

See [LICENSE](LICENSE) file for details.
