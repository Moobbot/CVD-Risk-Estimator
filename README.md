# CVD Risk Prediction API

API for predicting cardiovascular disease risk from DICOM images using Tri2D-Net model.

## Features

- Upload and process DICOM images
- Detect heart region using RetinaNet or simple method
- Predict CVD risk using Tri2D-Net model
- Generate detailed reports and visualizations
- Debug mode for detailed analysis

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Moobbot/cvd-risk-prediction.git
cd cvd-risk-prediction
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the API:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /predict

Predict CVD risk from DICOM images.

**Parameters:**

- `files`: List of DICOM files (multipart/form-data)
- `detection_method`: Method for heart detection ("auto", "model", "simple")
- `visualize`: Whether to generate visualization (boolean)
- `debug`: Whether to enable debug mode (boolean)

**Response:**

```json
{
    "risk_score": 0.75,
    "risk_details": {
        "risk_level": "High",
        "risk_percentage": "75.00%",
        "recommendations": [
            "Consult a cardiologist",
            "Monitor cardiovascular risk factors"
        ]
    },
    "metadata": {
        "PatientID": "12345",
        "StudyDate": "2023-01-01"
    },
    "report_path": "reports/cvd_risk_report_20230101_120000.txt",
    "visualization_path": "visualizations/cvd_risk_result_20230101_120000.png",
    "timestamp": "2023-01-01T12:00:00"
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
    "status": "healthy",
    "timestamp": "2023-01-01T12:00:00"
}
```

## Usage Examples

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@path/to/dicom1.dcm" \
     -F "files=@path/to/dicom2.dcm" \
     -F "detection_method=auto" \
     -F "visualize=true" \
     -F "debug=false"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/predict"
files = [
    ("files", ("dicom1.dcm", open("path/to/dicom1.dcm", "rb"))),
    ("files", ("dicom2.dcm", open("path/to/dicom2.dcm", "rb")))
]
data = {
    "detection_method": "auto",
    "visualize": "true",
    "debug": "false"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Directory Structure

```
cvd-risk-prediction/
├── api.py                 # FastAPI implementation
├── dicom_detect.py        # Core DICOM processing and prediction
├── tri_2d_net/           # Tri2D-Net model implementation
├── requirements.txt       # Project dependencies
├── README.md             # This file
├── reports/              # Generated reports
├── visualizations/       # Generated visualizations
└── debug/               # Debug information
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 200: Success
- 400: Bad Request
- 500: Internal Server Error

## Logging

Logs are stored in `cvd_api.log` with the following format:

```
2023-01-01 12:00:00,000 - INFO - Processing request
2023-01-01 12:00:01,000 - ERROR - Error in prediction: Invalid DICOM file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
