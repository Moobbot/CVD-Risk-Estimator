# Hướng dẫn sử dụng Batch Processing API

## 1. Truy cập Swagger UI

FastAPI tự động tạo Swagger UI documentation. Sau khi chạy API:

```bash
python api.py
```

Truy cập Swagger UI tại:

- **Swagger UI**: http://localhost:5556/docs
- **ReDoc**: http://localhost:5556/redoc

### Cách sử dụng Swagger UI:

1. Mở trình duyệt và truy cập `http://localhost:5556/docs`
2. Tìm endpoint `POST /api_batch_process`
3. Click vào endpoint để mở rộng
4. Click nút **"Try it out"**
5. Nhập đường dẫn folder vào trường `folder_path`:
   ```json
   {
     "folder_path": "C:/path/to/your/folder"
   }
   ```
6. Click **"Execute"**
7. Chờ quá trình xử lý hoàn tất
8. File ZIP chứa các CSV files sẽ được tải xuống tự động

## 2. Sử dụng Script Test

### Cài đặt dependencies (nếu chưa có):

```bash
pip install requests
```

### Chạy script:

```bash
# Cách 1: Truyền folder path qua argument
python test_batch_api.py "C:/path/to/your/folder"

# Cách 2: Chạy script và nhập folder path khi được hỏi
python test_batch_api.py
```

### Ví dụ:

```bash
# Windows
python test_batch_api.py "D:\Work\Clients\A_Giap\dicom-sybil\model-ai-code\CVD-Risk-Estimator\test_data"

# Linux/macOS
python test_batch_api.py "/home/user/dicom_folders"
```

## 3. Sử dụng cURL

```bash
curl -X POST "http://localhost:5556/api_batch_process" \
     -H "Content-Type: application/json" \
     -d "{\"folder_path\": \"C:/path/to/your/folder\"}" \
     --output batch_results.zip
```

## 4. Sử dụng Python requests

```python
import requests

url = "http://localhost:5556/api_batch_process"
payload = {
    "folder_path": "C:/path/to/your/folder"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    # Lưu file ZIP
    with open("batch_results.zip", "wb") as f:
        f.write(response.content)
    print("✅ Thành công! File ZIP đã được lưu.")
    print("📦 File ZIP chứa:")
    print("   - Mỗi subfolder có 1 CSV riêng với kết quả từng ảnh")
    print("   - 1 file summary CSV tổng hợp")
else:
    print(f"❌ Error: {response.json()}")
```

## 5. Cấu trúc Folder

Folder bạn chọn nên có cấu trúc như sau:

```
your_folder/
├── subfolder1/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── ...
├── subfolder2/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── ...
└── subfolder3/
    ├── image1.dcm
    └── ...
```

## 6. Cấu trúc Output

### File ZIP chứa:

- **Mỗi subfolder có 1 CSV riêng** với kết quả từng ảnh trong subfolder đó
- **1 file summary CSV** tổng hợp tất cả subfolders

### CSV cho mỗi subfolder

Mỗi CSV chứa thông tin từng ảnh trong subfolder:

- `file_name`: Tên file ảnh
- `attention_score`: Điểm attention của ảnh
- `overall_cvd_score`: Điểm CVD risk tổng thể của subfolder
- `subfolder_name`: Tên subfolder
- `subfolder_path`: Đường dẫn đầy đủ

**Ví dụ CSV cho subfolder (patient_001_results.csv):**

```csv
file_name,attention_score,overall_cvd_score,subfolder_name,subfolder_path
68_patient_001_68.png,0.85,0.75,patient_001,C:/data/patient_001
69_patient_001_69.png,0.78,0.75,patient_001,C:/data/patient_001
70_patient_001_70.png,0.65,0.75,patient_001,C:/data/patient_001
```

### Summary CSV

File summary CSV tổng hợp tất cả subfolders:

- `subfolder_name`: Tên subfolder
- `status`: "success" hoặc "error"
- `overall_score`: Điểm dự đoán CVD risk
- `total_images`: Tổng số ảnh đã xử lý
- `returned_images`: Số ảnh được trả về
- `error_message`: Thông báo lỗi (nếu có)
- `csv_file`: Tên file CSV tương ứng

**Ví dụ Summary CSV (batch_20241201_120000_summary.csv):**

```csv
subfolder_name,status,overall_score,total_images,returned_images,error_message,csv_file
patient_001,success,0.75,120,45,,patient_001_results.csv
patient_002,success,0.82,98,38,,patient_002_results.csv
patient_003,error,,0,0,No valid DICOM files found,patient_003_results.csv
```

### Cấu trúc ZIP file:

```
batch_20241201_120000_results.zip
├── patient_001_results.csv  (chứa kết quả từng ảnh trong patient_001)
├── patient_002_results.csv  (chứa kết quả từng ảnh trong patient_002)
├── patient_003_results.csv  (chứa kết quả từng ảnh trong patient_003)
└── batch_20241201_120000_summary.csv  (tổng hợp tất cả subfolders)
```

## 7. Lưu ý

- **Timeout**: Quá trình xử lý có thể mất nhiều thời gian tùy thuộc vào số lượng subfolder và kích thước file. Đảm bảo client có timeout đủ lớn (ví dụ: 1 giờ).
- **Đường dẫn**:
  - Windows: Sử dụng `C:/path/to/folder` hoặc `C:\\path\\to\\folder`
  - Linux/macOS: Sử dụng `/path/to/folder`
- **Quyền truy cập**: Đảm bảo API có quyền đọc folder và các file bên trong.
- **Model**: Đảm bảo model đã được load thành công khi khởi động API.

## 8. Troubleshooting

### Lỗi: "Folder not found"

- Kiểm tra đường dẫn có đúng không
- Kiểm tra quyền truy cập folder

### Lỗi: "No subfolders found"

- Đảm bảo folder chứa ít nhất một subfolder
- Kiểm tra subfolder là thư mục, không phải file

### Lỗi: "Model not found"

- Đảm bảo model đã được load khi khởi động API
- Kiểm tra log để xem lỗi load model

### Lỗi: Connection timeout

- Tăng timeout trong client
- Kiểm tra API vẫn đang chạy
- Xử lý batch có thể mất nhiều thời gian
