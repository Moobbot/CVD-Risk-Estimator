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

4. Set up environment variables:

   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit the .env file with your configuration
   # See Environment Variables section below for details
   ```

5. Run the API:

   ```bash
   python api.py
   ```

The API will automatically find an available port (default: 8080) and display the URLs:

```plaintext
Running on: http://127.0.0.1:8080 (localhost)
Running on: http://192.168.x.x:8080 (local network)
```

## Environment Variables

The application uses environment variables for configuration. You can set these in the `.env` file.

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Application environment (dev, test, prod) | `dev` |
| `PORT` | Port number for the server | `8080` |
| `DEVICE` | Device to use for model inference (cuda, cpu) | `cuda` if available |
| `MODEL_ITER` | Model iteration checkpoint to load | `700` |
| `LOG_LEVEL` | Logging level | `DEBUG` in dev, `INFO` otherwise |
| `CLEANUP_ENABLED` | Enable automatic cleanup of old files | `true` |

For a complete list of environment variables, see the [ENV_README.md](ENV_README.md) file.

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
  "overlay_images": "http://localhost:8080/download_zip/550e8400-e29b-41d4-a716-446655440000",
  "overlay_gif": "http://localhost:8080/download_gif/550e8400-e29b-41d4-a716-446655440000",
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
curl -X POST "http://localhost:8080/api_predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/dicom_images.zip"

# Download the results ZIP file
curl -X GET "http://localhost:8080/download_zip/550e8400-e29b-41d4-a716-446655440000" \
     -o results.zip

# Download the animated GIF
curl -X GET "http://localhost:8080/download_gif/550e8400-e29b-41d4-a716-446655440000" \
     -o results.gif

# Preview a specific overlay image
curl -X GET "http://localhost:8080/preview/550e8400-e29b-41d4-a716-446655440000/pred_image1.png" \
     -o preview.png
```

### Using Python requests

```python
import requests

# Predict CVD risk from a ZIP file containing DICOM images
url = "http://localhost:8080/api_predict"
files = {
    "file": ("dicom_images.zip", open("path/to/dicom_images.zip", "rb"))
}

response = requests.post(url, files=files)
result = response.json()
print(result)

# Get the session_id from the response
session_id = result["session_id"]

# Download the results ZIP file
download_url = f"http://localhost:8080/download_zip/{session_id}"
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
preview_url = f"http://localhost:8080/preview/{session_id}/{image_name}"
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
├── config.py             # Configuration settings
├── logger.py             # Logging configuration
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (local)
├── .env.example          # Environment variables template
├── ENV_README.md         # Environment variables documentation
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

## Logging

Logs are stored in the `logs` directory with UTF-8 encoding to support multiple languages. The log format is:

```log
2023-01-01 12:00:00,000 - api - INFO - Loading models on application startup...
2023-01-01 12:00:01,000 - api - ERROR - Error during processing: Could not detect heart
```

## Model Loading Optimization

The application is optimized to load models only once during startup using FastAPI's lifespan context manager. This improves performance and reduces memory usage by preventing duplicate model loading.

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
