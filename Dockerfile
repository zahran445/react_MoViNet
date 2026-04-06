FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies (FFmpeg, OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files (includes models/ since we updated .dockerignore)
COPY . .

# Default port used by web/app.py
EXPOSE 5000

# Persistence for runtime data
# NOTE: We keep /app/models outside of VOLUME if we want them baked into the image.
# If you want to persist them on host, mount them via docker-compose.
VOLUME ["/app/outputs"]

# Run verification script during build to ensure models are present
RUN python -c "import os; \
    assert os.path.exists('models/movinet/movinet_best.pt'), 'MoViNet weights missing'; \
    assert os.path.exists('models/yolo/plates_yolov8/weights/best.pt'), 'YOLO weights missing'; \
    print('--- Model verification successful ---')"

# Run the Flask dashboard
CMD ["python", "web/app.py"]

