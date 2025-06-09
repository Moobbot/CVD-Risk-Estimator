# API Documentation

## Endpoints

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
