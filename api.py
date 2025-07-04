import os
import traceback
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import (
    FOLDERS,
    API_CONFIG,
    SECURITY_CONFIG,
    HOST_CONNECT,
    PORT_CONNECT,
    IS_DEV,
)
from logger import setup_logger
from utils import cleanup_old_files, get_local_ip
from call_model import load_model

# Set up logger with date-based organization
logger = setup_logger("api")


# Define lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Lifespan context manager for FastAPI
    Handles startup and shutdown events
    """
    global heart_detector, model

    # Startup: Load models and clean up old directories
    cleanup_old_files([FOLDERS["CLEANUP"]])
    # cleanup_old_files([FOLDERS["UPLOAD"], FOLDERS["RESULTS"]])

    # Only load models if they haven't been loaded yet
    if heart_detector is None or model is None:
        try:
            logger.info("Loading models on application startup...")
            heart_detector, model = load_model()

            # Log model status
            if heart_detector is None:
                logger.warning(
                    "Heart detector model not loaded. Will use simple method for heart detection."
                )
            else:
                logger.info("Heart detector model loaded successfully.")

            if model is None:
                logger.warning(
                    "CVD risk prediction model not loaded. API will return errors for prediction requests."
                )
            else:
                logger.info("CVD risk prediction model loaded successfully.")

        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't raise exception here so the application can still start

    # Update the models in the routes module
    import routes

    routes.heart_detector = heart_detector
    routes.model = model

    yield  # This is where FastAPI runs

    # Shutdown: Clean up resources if needed
    logger.info("Application shutting down...")


# Initialize global model variables
heart_detector = None
model = None

# Khởi tạo ứng dụng FastAPI with lifespan
app = FastAPI(
    title=API_CONFIG["TITLE"],
    description=API_CONFIG["DESCRIPTION"],
    version=API_CONFIG["VERSION"],
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=SECURITY_CONFIG["CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=SECURITY_CONFIG["CORS_METHODS"],
    allow_headers=SECURITY_CONFIG["CORS_HEADERS"],
)

# Phục vụ các file tĩnh từ thư mục kết quả
app.mount("/results", StaticFiles(directory=FOLDERS["RESULTS"]), name="results")

# Include the router
from routes import router

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    import socket

    custom_port = PORT_CONNECT

    # Check if the port is available
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    # Find an available port
    while is_port_in_use(custom_port):
        custom_port += 1

    LOCAL_IP = get_local_ip()
    print(f"Running on: http://127.0.0.1:{custom_port} (localhost)")
    print(f"Running on: http://{LOCAL_IP}:{custom_port} (local network)")

    # Run without reload to avoid loading models twice
    uvicorn.run("api:app", host=HOST_CONNECT, port=custom_port, reload=IS_DEV)
