# Hướng dẫn sử dụng Docker với CVD Risk Estimator

Tài liệu này hướng dẫn cách sử dụng Docker để triển khai ứng dụng CVD Risk Estimator trong cả môi trường có GPU và không có GPU.

## Yêu cầu

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (tùy chọn, nhưng được khuyến nghị)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (chỉ khi sử dụng GPU)

## Cách sử dụng

### 1. Xây dựng và chạy với Docker Compose (Khuyến nghị)

Docker Compose giúp quản lý container dễ dàng hơn và cho phép cấu hình thông qua file `docker-compose.yml`.

#### Với GPU

```bash
# Xây dựng và khởi động container với GPU
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng container
docker-compose down
```

#### Không có GPU

```bash
# Sử dụng file docker-compose.cpu.yml đặc biệt cho môi trường không có GPU
docker-compose -f docker-compose.cpu.yml up -d

# Xem logs
docker-compose -f docker-compose.cpu.yml logs -f

# Dừng container
docker-compose -f docker-compose.cpu.yml down
```

> **Lưu ý**: Phiên bản CPU sử dụng Dockerfile.cpu riêng biệt đã được tối ưu hóa cho môi trường không có GPU.

### 2. Xây dựng và chạy với Docker

Nếu bạn không sử dụng Docker Compose, bạn có thể sử dụng các lệnh Docker trực tiếp:

#### Sử dụng GPU

```bash
# Xây dựng image cho GPU
docker build --build-arg USE_GPU=true -t cvd-risk-estimator-gpu .

# Chạy container với GPU
docker run --gpus all -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env --name cvd-risk-estimator -d cvd-risk-estimator-gpu
```

#### Sử dụng CPU

```bash
# Xây dựng image cho CPU sử dụng Dockerfile.cpu
docker build -f Dockerfile.cpu -t cvd-risk-estimator-cpu .

# Chạy container với CPU
docker run -p 5556:5556 -v $(pwd)/checkpoint:/app/checkpoint -v $(pwd)/logs:/app/logs -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results -v $(pwd)/.env:/app/.env --name cvd-risk-estimator -d cvd-risk-estimator-cpu
```

## Cấu hình

### Biến môi trường

Bạn có thể cấu hình ứng dụng bằng cách chỉnh sửa file `.env` hoặc thông qua biến môi trường trong `docker-compose.yml`:

```yaml
environment:
  - ENV=prod
  - HOST_CONNECT=0.0.0.0
  - PORT=8080
  - CUDA_VISIBLE_DEVICES=0
  - DEVICE=cuda
```

### Volumes

Các volume sau được sử dụng để lưu trữ dữ liệu giữa các lần chạy container:

- `./checkpoint:/app/checkpoint`: Lưu trữ các mô hình đã tải xuống
- `./logs:/app/logs`: Lưu trữ logs của ứng dụng
- `./uploads:/app/uploads`: Lưu trữ các file tải lên tạm thời
- `./results:/app/results`: Lưu trữ kết quả dự đoán
- `./.env:/app/.env`: File cấu hình môi trường

## Xử lý sự cố

### Không thể sử dụng GPU

Nếu bạn gặp lỗi liên quan đến GPU, hãy đảm bảo:

1. NVIDIA Container Toolkit đã được cài đặt đúng cách
2. Driver NVIDIA đã được cài đặt và hoạt động
3. Kiểm tra với lệnh `nvidia-smi`

Nếu vẫn gặp vấn đề, bạn có thể chuyển sang chế độ CPU bằng cách:

1. Sử dụng file Docker Compose dành riêng cho CPU: `docker-compose -f docker-compose.cpu.yml up -d`
2. Hoặc xây dựng với Dockerfile.cpu: `docker build -f Dockerfile.cpu -t cvd-risk-estimator-cpu .`
3. Hoặc đặt `DEVICE=cpu` trong biến môi trường khi chạy container đã tồn tại

Lưu ý rằng chế độ CPU sẽ chậm hơn đáng kể cho việc suy luận nhưng cho phép ứng dụng chạy trên bất kỳ máy nào mà không cần GPU.

### Lỗi khi tải mô hình

Nếu bạn gặp lỗi khi tải mô hình, hãy đảm bảo:

1. Các file mô hình đã được tải xuống vào thư mục `checkpoint`
2. Thư mục `checkpoint` đã được gắn kết đúng cách vào container

## Tối ưu hóa

### Giảm kích thước image

Để giảm kích thước image Docker, bạn có thể:

1. Thêm các file lớn vào `.dockerignore`
2. Sử dụng multi-stage build
3. Dọn dẹp các gói không cần thiết sau khi cài đặt

### Cải thiện hiệu suất

Để cải thiện hiệu suất, bạn có thể:

1. Tăng giá trị `BATCH_SIZE` nếu có đủ bộ nhớ GPU
2. Sử dụng `--shm-size` để tăng bộ nhớ chia sẻ khi chạy container

## Triển khai trong môi trường sản xuất

Khi triển khai trong môi trường sản xuất, hãy đảm bảo:

1. Đặt `ENV=prod` để tắt các tính năng debug
2. Cấu hình `CORS_ORIGINS` để chỉ cho phép các nguồn gốc được tin cậy
3. Sử dụng proxy ngược như Nginx để xử lý HTTPS và cân bằng tải
4. Thiết lập giám sát và cảnh báo
