# CVD Risk Estimator

API for predicting cardiovascular disease risk from DICOM images using Tri2D-Net model.

## Features

- Upload and process DICOM images in ZIP format
- Detect heart region using RetinaNet or simple method
- Predict CVD risk using Tri2D-Net model
- Generate Grad-CAM visualizations for explainability
- Create animated GIFs from Grad-CAM visualizations
- Attention score calculation for important slices
- Environment variable configuration for easy deployment
- Unicode support for logging in multiple languages
- Automatic CPU fallback when GPU is not available

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/CVD-Risk-Estimator.git
   cd CVD-Risk-Estimator
   ```

2. Create and activate a virtual environment (recommended):

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
   # Copy the example environment file (optional)
   cp .env.example .env

   # Edit the .env file with your configuration
   # This is optional as all configuration options have default values
   # See Configuration section below for details
   ```

5. Run the API:

   ```bash
   python api.py
   ```

The API will automatically find an available port (default: 5556) and display the URLs:

```plaintext
Running on: http://127.0.0.1:5556 (localhost)
Running on: http://192.168.x.x:5556 (local network)
```

## Configuration

The application uses a flexible configuration system with default values defined directly in `config.py`. You can override these defaults using environment variables or a `.env` file.

### Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Application environment (dev, test, prod) | `dev` |
| `PORT` | Port number for the server | `5556` |
| `DEVICE` | Device to use for model inference (cuda, cpu) | `cuda` if available, auto-fallback to `cpu` |
| `CUDA_VISIBLE_DEVICES` | CUDA device indices to use | `0` (set empty to force CPU) |
| `MODEL_ITER` | Model iteration checkpoint to load | `700` |
| `LOG_LEVEL` | Logging level | `DEBUG` in dev, `INFO` otherwise |
| `CLEANUP_ENABLED` | Enable automatic cleanup of old files | `true` |

For a complete list of configuration options, see the [ENV_README.md](ENV_README.md) file.

### Configuration Methods

You can configure the application in three ways (in order of precedence):

1. **Environment Variables**: Set directly in your system or container
2. **`.env` File**: Create a `.env` file in the project root
3. **Default Values**: Defined in `config.py` (used if no override is provided)

Example of using environment variables:

```bash
# Windows
set PORT=5556
python api.py

# Linux/macOS
PORT=5556 python api.py
```

## API Endpoints

### POST /api_predict

Predict CVD risk from DICOM images in a ZIP file.

**Parameters:**

- `file`: ZIP file containing DICOM images (multipart/form-data)

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "predictions": [
    {
      "score": 0.75
    }
  ],
  "overlay_images": "http://localhost:5556/download_zip/550e8400-e29b-41d4-a716-446655440000",
  "overlay_gif": "http://localhost:5556/download_gif/550e8400-e29b-41d4-a716-446655440000",
  "attention_info": {
    "attention_scores": [
      {
        "file_name_pred": "pred_image1.png",
        "attention_score": 0.85
      },
      {
        "file_name_pred": "pred_image2.png",
        "attention_score": 0.65
      }
    ],
    "total_images": 10,
    "returned_images": 2
  },
  "message": "Prediction successful."
}
```

### GET /download_zip/{session_id}

Download the ZIP file containing overlay images with Grad-CAM visualizations.

**Parameters:**

- `session_id`: Session ID from the prediction response

**Response:**

- ZIP file containing overlay images

### GET /download_gif/{session_id}

Download an animated GIF of the Grad-CAM visualizations.

**Parameters:**

- `session_id`: Session ID from the prediction response

**Response:**

- GIF file containing animated Grad-CAM visualizations

### GET /preview/{session_id}/{filename}

Preview a specific overlay image.

**Parameters:**

- `session_id`: Session ID from the prediction response
- `filename`: Name of the file to preview

**Response:**

- Image file

## Usage Examples

### Using curl

```bash
# Predict CVD risk from a ZIP file containing DICOM images
curl -X POST "http://localhost:5556/api_predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/dicom_images.zip"

# Download the results ZIP file
curl -X GET "http://localhost:5556/download_zip/550e8400-e29b-41d4-a716-446655440000" \
     -o results.zip

# Download the animated GIF
curl -X GET "http://localhost:5556/download_gif/550e8400-e29b-41d4-a716-446655440000" \
     -o results.gif

# Preview a specific overlay image
curl -X GET "http://localhost:5556/preview/550e8400-e29b-41d4-a716-446655440000/pred_image1.png" \
     -o preview.png
```

### Using Python requests

```python
import requests

# Predict CVD risk from a ZIP file containing DICOM images
url = "http://localhost:5556/api_predict"
files = {
    "file": ("dicom_images.zip", open("path/to/dicom_images.zip", "rb"))
}

response = requests.post(url, files=files)
result = response.json()
print(result)

# Get the session_id from the response
session_id = result["session_id"]

# Download the results ZIP file
download_url = f"http://localhost:5556/download_zip/{session_id}"
download_response = requests.get(download_url)
with open("results.zip", "wb") as f:
    f.write(download_response.content)

# Download the animated GIF if available
if result.get("overlay_gif"):
    gif_url = result["overlay_gif"]
    gif_response = requests.get(gif_url)
    with open("results.gif", "wb") as f:
        f.write(gif_response.content)

# Preview a specific overlay image
image_name = result["attention_info"]["attention_scores"][0]["file_name_pred"]
preview_url = f"http://localhost:5556/preview/{session_id}/{image_name}"
preview_response = requests.get(preview_url)
with open("preview.png", "wb") as f:
    f.write(preview_response.content)
```

## Directory Structure

```plaintext
CVD-Risk-Estimator/
├── api.py                # FastAPI implementation
├── heart_detector.py     # Heart detection using RetinaNet
├── image.py              # DICOM image processing
├── tri_2d_net/           # Tri2D-Net model implementation
├── checkpoint/           # Model checkpoints
├── config.py             # Configuration with built-in defaults
├── logger.py             # Logging configuration
├── requirements.txt      # Project dependencies
├── .env.example          # Environment variables template
├── .env                  # Optional environment variables (local)
├── ENV_README.md         # Configuration documentation
├── README.md             # This file
├── uploads/              # Temporary upload directory
├── results/              # Results and visualizations
└── logs/                 # Application logs
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 200: Success
- 400: Bad Request (invalid input, missing files)
- 422: Unprocessable Entity (could not detect heart)
- 500: Internal Server Error

The application includes robust error handling for various scenarios:

- **Missing GPU**: When NVIDIA drivers are not available, the application automatically falls back to CPU mode
- **Model Loading Failures**: Clear error messages are provided when models fail to load
- **Heart Detection Fallback**: If the heart detection model fails, a simple geometric method is used as fallback
- **Graceful Degradation**: The API continues to operate in a degraded mode when possible, with appropriate error responses

## Logging

Logs are stored in the `logs` directory with UTF-8 encoding to support multiple languages. The log format is:

```log
2023-01-01 12:00:00,000 - api - INFO - Loading models on application startup...
2023-01-01 12:00:01,000 - api - ERROR - Error during processing: Could not detect heart
```

## Model Loading Optimization

The application is optimized to load models only once during startup using FastAPI's lifespan context manager. This improves performance and reduces memory usage by preventing duplicate model loading.

## Configuration Optimization

The application uses a three-tier configuration system:

1. Default values defined directly in `config.py`
2. Optional `.env` file for environment-specific overrides
3. Environment variables for runtime configuration

This approach provides flexibility while ensuring the application can run with minimal setup. All configuration options have sensible defaults, making the `.env` file completely optional.

## GPU/CPU Compatibility

The application is designed to work on both GPU and CPU environments:

- **Automatic Device Detection**: The system automatically detects if CUDA is available and configures models accordingly
- **CPU Fallback**: If no NVIDIA GPU is available or if CUDA drivers are not installed, the application automatically falls back to CPU mode
- **Manual Override**: You can force CPU mode by setting `DEVICE=cpu` in your environment variables or `.env` file
- **Graceful Error Handling**: The application provides clear error messages when model loading fails and continues to operate in a degraded mode when possible

To explicitly set the device for inference:

```bash
# Force CPU mode
DEVICE=cpu python api.py

# Force GPU mode (requires CUDA)
DEVICE=cuda python api.py
```

## Unicode Support

The application supports Unicode characters in logs and messages, making it suitable for use in multiple languages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
