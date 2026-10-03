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
from checkpoints import ensure_checkpoints, summary as checkpoint_summary
from model_info import build_info, source_digest
from config import MODEL_CONFIG

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
        # The checkpoint folder is mounted from the host and is empty on a fresh checkout: a
        # missing file is copied from the image, or downloaded (checkpoints.py). Files already in
        # the folder are used as they are. Done here, not in call_model.py, which is part of the
        # model version identity: how the files get into the folder is not inference code.
        # The summary also goes to stdout: outside dev the loggers write to files only, and
        # `docker compose logs cvd` is where an operator looks.
        try:
            print(checkpoint_summary(ensure_checkpoints(FOLDERS["CHECKPOINT"], log=logger)), flush=True)
        except Exception as e:
            logger.error(f"Could not prepare the checkpoint folder: {str(e)}")
            print(f"Could not prepare the checkpoint folder: {str(e)}", flush=True)

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
    try:
        routes.MODEL_INFO = _build_model_info(heart_detector, model)
    except Exception as e:
        # A failing version identity must NOT stop the service from starting:
        # version=None -> the backend records "unknown".
        logger.error(f"Could not build the version identity: {e}")
        routes.MODEL_INFO = {"model": "cvd", "loaded": model is not None, "version": None}
    logger.info(f"[model_info] {routes.MODEL_INFO.get('version')}")

    yield  # This is where FastAPI runs

    # Shutdown: Clean up resources if needed
    logger.info("Application shutting down...")


def _build_model_info(heart_detector, model):
    """P4c: identify EXACTLY the weights + inference code that were loaded (see model_info.py)."""
    if model is None:
        return {"model": "cvd", "loaded": False, "version": None}
    base = os.path.dirname(os.path.abspath(__file__))
    src = source_digest([os.path.join(base, p) for p in (
        "tri_2d_net", "detector", "call_model.py", "image.py",
        "heart_detector.py", "bbox_cut.py", "utils.py", "config.py")])
    code = f"iter{MODEL_CONFIG['ITER']}.src.{src[:8] if src else 'unknown'}"
    weights = [getattr(model, "loaded_checkpoint_path", None)]
    if os.path.exists(MODEL_CONFIG["RETINANET_PATH"]):
        weights.append(MODEL_CONFIG["RETINANET_PATH"])
    info = build_info("cvd", code, [w for w in weights if w] if all(weights) else [], [], MODEL_CONFIG["DEVICE"])
    # Whether the detector loaded at startup — for display only; the method ACTUALLY
    # used for each case is the `+det.simple` flag in the response's model_version.
    info["heart_detector_loaded"] = bool(heart_detector is not None and getattr(heart_detector, "model", None) is not None)
    return info


# Initialize global model variables
heart_detector = None
model = None

# Create the FastAPI application with lifespan
app = FastAPI(
    title=API_CONFIG["TITLE"],
    description=API_CONFIG["DESCRIPTION"],
    version=API_CONFIG["VERSION"],
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=SECURITY_CONFIG["CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=SECURITY_CONFIG["CORS_METHODS"],
    allow_headers=SECURITY_CONFIG["CORS_HEADERS"],
)

# Serve static files from the results folder
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
