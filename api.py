import os
import traceback
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import FOLDERS, API_TITLE, API_DESCRIPTION, API_VERSION
from tri_2d_net.init_model import init_model
from heart_detector import HeartDetector
from logger import setup_logger
from utils import cleanup_old_files, get_local_ip

# Set up logger with date-based organization
logger = setup_logger("api")


def load_model():
    """
    Load necessary models for the application

    Returns:
        tuple: (heart_detector, model) - Loaded models
    """
    heart_detector = None
    model = None

    # Try to load heart detector
    try:
        logger.info("Loading heart detection model...")
        heart_detector = HeartDetector()
        if not heart_detector.load_model():
            logger.warning(
                "Could not load heart detection model, will use simple method"
            )
    except Exception as e:
        logger.error(f"Error loading heart detection model: {str(e)}")
        logger.error(traceback.format_exc())
        # Continue with heart_detector = None, simple method will be used

    # Try to load CVD risk prediction model
    try:
        logger.info("Loading cardiovascular risk prediction model...")
        model = init_model()
        model.load_model()
        logger.info("CVD risk prediction model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load CVD risk prediction model: {str(e)}")
        logger.error(traceback.format_exc())
        # This is a critical error, but we'll return None and handle it in the API
        model = None

    # Return both models, even if one or both are None
    # The API will check and handle appropriately
    return heart_detector, model


# Define lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Lifespan context manager for FastAPI
    Handles startup and shutdown events
    """
    global heart_detector, model

    # Startup: Load models and clean up old directories
    cleanup_old_files([FOLDERS["UPLOAD"], FOLDERS["RESULTS"]])

    # Only load models if they haven't been loaded yet
    if heart_detector is None or model is None:
        try:
            logger.info("Loading models on application startup...")
            heart_detector, model = load_model()

            # Log model status
            if heart_detector is None:
                logger.warning("Heart detector model not loaded. Will use simple method for heart detection.")
            else:
                logger.info("Heart detector model loaded successfully.")

            if model is None:
                logger.warning("CVD risk prediction model not loaded. API will return errors for prediction requests.")
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
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ các file tĩnh từ thư mục kết quả
app.mount("/results", StaticFiles(directory=FOLDERS["RESULTS"]), name="results")

# Include the router
from routes import router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    import socket
    from config import HOST_CONNECT, PORT_CONNECT

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
    uvicorn.run("api:app", host=HOST_CONNECT, port=custom_port, reload=False)
