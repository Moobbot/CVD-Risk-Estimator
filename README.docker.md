# Hướng Dẫn Triển Khai Docker

Hướng dẫn này cung cấp các hướng dẫn nhanh để triển khai CVD Risk Estimator bằng Docker. Để biết thêm thông tin chi tiết về triển khai, xem [Hướng Dẫn Triển Khai](docs/deployment.md).

## Yêu Cầu

- Đã cài đặt Docker và Docker Compose
- NVIDIA Container Toolkit (để hỗ trợ GPU)
- Đã cài đặt driver NVIDIA (để hỗ trợ GPU)

## Bắt Đầu Nhanh

### Với GPU

```bash
# Build và chạy container với GPU
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng container
docker-compose down
```

### Không Có GPU

```bash
# Sử dụng service CPU trong docker-compose.yml
docker-compose up -d app-cpu

# Xem logs
docker-compose logs -f cvd-risk-estimator-cpu

# Dừng container
docker-compose down
```

## Cấu Hình Docker

Cấu hình Docker bao gồm:

- Image cơ sở Python 3.10
- Các thư viện hệ thống cần thiết (ffmpeg, libsm6, libxext6)
- Môi trường ảo để quản lý dependencies sạch sẽ
- Tối ưu kích thước image bằng multi-stage builds
- Người dùng không phải root để tăng cường bảo mật
- Volume mounts để lưu trữ dữ liệu liên tục
- Cấu hình biến môi trường
- Hỗ trợ GPU bằng NVIDIA Container Toolkit

Để biết thêm thông tin chi tiết về triển khai Docker, cấu hình và xử lý sự cố, vui lòng tham khảo [Hướng Dẫn Triển Khai](docs/deployment.md).

## Tính năng chính

- Dự đoán nguy cơ bệnh tim mạch từ ảnh DICOM
- Phát hiện vùng tim tự động bằng RetinaNet hoặc phương pháp đơn giản
- Tạo hình ảnh Grad-CAM để giải thích kết quả
- Tạo GIF động trực tiếp từ các hình ảnh Grad-CAM
- Tổ chức logs theo cấu trúc năm/tháng/ngày
- Tự động chuyển sang chế độ CPU khi không có GPU
- Tối ưu hóa việc tải mô hình trong quá trình khởi động

## Cấu hình

### Biến môi trường

Bạn có thể cấu hình ứng dụng bằng cách chỉnh sửa file `.env` hoặc thông qua biến môi trường trong `docker-compose.yml`:

```yaml
environment:
  - ENV=prod
  - HOST_CONNECT=0.0.0.0
  - PORT=5556
  - CUDA_VISIBLE_DEVICES=0
  - DEVICE=cuda
```

### Volumes

Các volume sau được sử dụng để lưu trữ dữ liệu giữa các lần chạy container:

- `./checkpoint:/app/checkpoint`: Lưu trữ các mô hình đã tải xuống
- `./logs:/app/logs`: Lưu trữ logs của ứng dụng (được tổ chức theo năm/tháng/ngày)
- `./uploads:/app/uploads`: Lưu trữ các file tải lên tạm thời
- `./results:/app/results`: Lưu trữ kết quả dự đoán và file GIF
- `./.env:/app/.env`: File cấu hình môi trường

#### Tổ chức logs

Logs được tự động tổ chức theo cấu trúc năm/tháng với tên file dựa trên ngày:

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

Cấu trúc này giúp dễ dàng tìm kiếm logs cho các ngày cụ thể và ngăn các file log trở nên quá lớn.

## Xử lý sự cố

### Không thể sử dụng GPU

Nếu bạn gặp lỗi liên quan đến GPU, hãy đảm bảo:

1. NVIDIA Container Toolkit đã được cài đặt đúng cách
2. Driver NVIDIA đã được cài đặt và hoạt động
3. Kiểm tra với lệnh `nvidia-smi`

File Docker Compose đã được cấu hình để truy cập GPU đúng cách sử dụng định dạng Docker Compose hiện đại:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Nếu vẫn gặp vấn đề, bạn có thể chuyển sang chế độ CPU bằng cách:

1. Sử dụng dịch vụ CPU trong docker-compose.yml: `docker-compose up -d app-cpu`
2. Hoặc đặt `DEVICE=cpu` trong biến môi trường khi chạy container đã tồn tại

Lưu ý rằng chế độ CPU sẽ chậm hơn đáng kể cho việc suy luận nhưng cho phép ứng dụng chạy trên bất kỳ máy nào mà không cần GPU.

### Lỗi khi tải mô hình

Nếu bạn gặp lỗi khi tải mô hình, hãy đảm bảo:

1. Các file mô hình đã được tải xuống vào thư mục `checkpoint`
2. Thư mục `checkpoint` đã được gắn kết đúng cách vào container

## Tối ưu hóa

### Giảm kích thước image

Dockerfile đã được tối ưu hóa để giảm kích thước image bằng cách:

1. Sử dụng multi-stage build để tách biệt môi trường build và runtime
2. Sử dụng image cơ sở slim để giảm kích thước
3. Chỉ cài đặt các gói cần thiết và dọn dẹp cache apt sau khi cài đặt
4. Thêm các file lớn vào `.dockerignore`

### Bảo mật

Ứng dụng đã được cải thiện bảo mật bằng cách:

1. Chạy ứng dụng với người dùng không phải root (appuser)
2. Chỉ cài đặt các gói cần thiết cho runtime
3. Sử dụng các quyền tối thiểu cho các thư mục

### Cải thiện hiệu suất

Để cải thiện hiệu suất, bạn có thể:

1. Tăng giá trị `BATCH_SIZE` nếu có đủ bộ nhớ GPU
2. Sử dụng `--shm-size` để tăng bộ nhớ chia sẻ khi chạy container

## Triển khai trong môi trường sản xuất

Khi triển khai trong môi trường sản xuất, hãy đảm bảo:

1. Đặt `ENV=prod` để tắt các tính năng debug
2. Cấu hình `CORS_ORIGINS` để chỉ cho phép các nguồn gốc được tin cậy
3. Sử dụng proxy ngược như Nginx để xử lý HTTPS và cân bằng tải
