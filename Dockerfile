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

# The two checkpoints (~280 MB: heart detector + CVD encoder) are downloaded into /app/checkpoint
# at build time, as before: the cvd-checkpoint volume is seeded from them when it is first created.
# They must be the files the reference results were measured with. setup.py saves whatever the
# link returns and keeps a download cut short, so a file that is there but differs stops the build
# (a failed step is not cached: the next build downloads again). A file that could not be downloaded
# at all only warns: an existing cvd-checkpoint volume already has it, and the update must not
# depend on reaching Dropbox.
RUN python setup.py --skip-packages \
    && cd checkpoint \
    && check() { \
         if [ ! -e "$2" ]; then echo "WARNING: $2 was not downloaded: a NEW cvd-checkpoint volume would not have it"; \
         else echo "$1  $2" | sha256sum -c - || { echo "ERROR: $2 is not the expected checkpoint (download cut short or changed)"; exit 1; }; fi; } \
    && check dccf38ef25b478dcb77a2d86a4ea4fd3a6beccd9f9776c648d9edd42da39982d retinanet_heart.pt \
    && check 5d591131d18c9b7e23d3235f82f5550261d14b968d9f4950c2e76c9c5ec0fc1e NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm

# Copy the rest of the application
COPY . .

# Set environment variables
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5556 \
    ENV=prod \
    DEVICE=cuda

EXPOSE 5556

CMD ["python", "api.py"]
