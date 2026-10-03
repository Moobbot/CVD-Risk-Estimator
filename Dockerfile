# ---- torch-base: keep this stage IDENTICAL in Sybil/Dockerfile and CVD-Risk-Estimator/Dockerfile
# (dicom-diagnosis/scripts/__tests__/dockerfiles.test.js checks it). Built together
# (`docker compose build`), or one after the other on the same machine, BuildKit builds it once and
# both images share its layers: the ~4.9 GB of torch + CUDA libraries is stored once, not twice.
FROM python:3.10-slim AS torch-base

# pip otherwise keeps every downloaded wheel in /root/.cache/pip (1.9 GB, never used at run time).
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries the Python packages link against (found with ldd on the previous image):
# OpenCV needs GL, glib, X11 and libatomic. The full python:3.10 image and ffmpeg are not needed
# (GIFs are written by imageio through Pillow).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libatomic1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==24.0

# The build setup.py used to install (CUDA 12.1 wheels), pinned: 2.5.1 is the last cu121 release.
# torchaudio is not installed: nothing imports it.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# ---- CVD service
FROM torch-base

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt setup.py ./

RUN pip install -r requirements.txt

# The two checkpoints (~280 MB: heart detector + CVD encoder) are downloaded at build time, as before,
# and must be the files the reference results were measured with: a file that was downloaded but
# differs (a download cut short, an error page) stops the build (a failed step is not cached: the
# next build downloads again); a file that could not be downloaded at all only warns, and that
# layer is cached without the file: `docker compose build --no-cache cvd` downloads again.
# They go to /app/checkpoint-seed, NOT /app/checkpoint: docker-compose mounts a host folder on
# /app/checkpoint, which hides whatever the image has there. At start-up the service copies a
# missing file from the seed into the mounted folder (checkpoints.py), so an empty folder needs
# neither a manual step nor internet access.
COPY checkpoints.py ./
RUN python checkpoints.py --seed /app/checkpoint-seed

# Copy the rest of the application
COPY . .

# Set environment variables
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5556 \
    ENV=prod \
    DEVICE=cuda

EXPOSE 5556

CMD ["python", "api.py"]
