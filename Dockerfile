FROM python:3.10

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Tạo và sử dụng virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cập nhật pip lên phiên bản 24.0
RUN pip install --upgrade pip==24.0

COPY . .

# Cài đặt dependencies từ setup.py
RUN python setup.py

# Thiết lập biến môi trường cho ứng dụng
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5556 \
    ENV=prod \
    DEVICE=cuda

# Mở cổng cho API
EXPOSE 5556

# Lệnh khởi động ứng dụng
CMD ["python", "api.py"]
