# Sử dụng image Python 3.10 với CUDA 12.1 làm base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh

# Cài đặt các gói phụ thuộc hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3-setuptools \
    python3-wheel \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo symbolic link cho python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép setup.py và requirements.txt trước để tận dụng Docker cache
COPY setup.py requirements.txt ./

# Cài đặt các gói phụ thuộc Python thông qua setup.py
# Sử dụng CUDA 12.1 và không bỏ qua cài đặt gói
RUN python setup.py --cuda-version 12.1

# Sao chép mã nguồn ứng dụng
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p uploads results logs checkpoint detector

# Thiết lập biến môi trường cho ứng dụng
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5556 \
    ENV=prod \
    CUDA_VISIBLE_DEVICES=0 \
    DEVICE=cuda

# Mở cổng cho API
EXPOSE 5556

# Lệnh khởi động ứng dụng
CMD ["python", "api.py"]
