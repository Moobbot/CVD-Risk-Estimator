import os
import zipfile
import uuid
import shutil
import traceback

import SimpleITK as sitk
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse

from config import FOLDERS, ERROR_MESSAGES
from logger import setup_logger
from utils import create_zip_result
from fastapi.concurrency import run_in_threadpool
from call_model import predict as _predict_unserialized
from inference_gate import serialized, start_watchdog, status as inference_status

# SimpleITK configuration
sitk.ProcessObject.SetGlobalDefaultThreader("platform")

# Set up logger with date-based organization
logger = setup_logger("routes")

# Initialize router
router = APIRouter()

# Global variables for models (will be set by the main app)
heart_detector = None
model = None
# P4c: set by api.py when the model loads (model_info.build_info).
MODEL_INFO: dict = {"model": "cvd", "loaded": False, "version": None}


def _case_version(pred_dict: dict) -> str | None:
    """The version for THIS case: appends `+det.simple` if this case located the heart
    with the "simple" method (the detector could not run for this case)."""
    base = MODEL_INFO.get("version")
    if base and pred_dict.get("heart_detection") == "simple":
        return f"{base}+det.simple"
    return base

# Every inference runs one after another (see inference_gate.py).
predict = serialized(_predict_unserialized)
start_watchdog()


@router.get("/health")
async def health() -> JSONResponse:
    """Alive or dead, idle or busy. Answers even during inference, because
    inference runs in the threadpool and does not block the event loop.

    503 when the model is not loaded (load_model failed -> model=None) or the running
    case is HUNG. Previously a failed model load still reported "ok" and was treated as usable.
    """
    st = inference_status()
    loaded = model is not None
    ok = loaded and not st["stuck"]
    body = {"status": "ok" if ok else "unhealthy", "model_loaded": loaded, **st}
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/info")
async def info() -> JSONResponse:
    """P4c: the running model version. Weight file names only (no paths)."""
    if model is None or not MODEL_INFO.get("version"):
        return JSONResponse({"model": "cvd", "loaded": False, "version": None}, status_code=503)
    return JSONResponse(MODEL_INFO)


@router.post("/api_predict")
async def api_predict(request: Request) -> JSONResponse:
    """
    API that takes a session_id, opens the already-extracted folder and runs the prediction
    Args:
        request: Request object (the body holds session_id)
    Returns:
        JSONResponse: prediction result with the risk score and the paths to the result images
    """
    logger.info("API predict (session_id) called")
    data = await request.json()
    session_id = data.get("session_id") if data else None
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    dicom_uuid_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    result_uuid_dir = os.path.join(FOLDERS["RESULTS"], session_id, "cvd")

    if not os.path.exists(dicom_uuid_dir):
        return JSONResponse(
            {"error": f"Session folder not found: {dicom_uuid_dir}"}, status_code=404
        )

    os.makedirs(result_uuid_dir, exist_ok=True)

    # Check the files after extraction
    valid_files = []
    for root, _, files in os.walk(dicom_uuid_dir):
        for filename in files:
            if filename.lower().endswith((".dcm", ".png")):
                valid_files.append(os.path.join(root, filename))
    if not valid_files:
        return JSONResponse(
            {"error": "No valid files found in the session folder"}, status_code=400
        )

    logger.info(f"Found {len(valid_files)} valid files")

    # Find the sub-folder that contains the DICOM files (if any)
    dicom_dir = dicom_uuid_dir
    for root, _, files in os.walk(dicom_uuid_dir):
        if any(file.endswith(".dcm") for file in files):
            dicom_dir = root
            logger.info(f"Found directory containing DICOM files: {root}")
            break

    if model is None:
        return JSONResponse(
            {"error": ERROR_MESSAGES["model_not_found"]}, status_code=500
        )

    try:
        pred_dict, attention_info, gif_path = await run_in_threadpool(
            predict,
            dicom_dir=dicom_dir,
            output_dir=result_uuid_dir,
            heart_detector=heart_detector,
            model=model,
            session_id=session_id,
            create_gif=True,
        )

        response = {
            "session_id": session_id,
            "predictions": pred_dict["predictions"],
            "attention_info": attention_info,
            "message": "Prediction successful.",
            "model_version": _case_version(pred_dict),
        }
        logger.info(f"Prediction {session_id} successful.")
        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)


@router.post("/api_predict_zip")
async def api_predict_zip(
    request: Request, file: UploadFile = File(...)
) -> JSONResponse:
    """
    API that takes a ZIP file of DICOM images, converts it to NIFTI and runs the prediction

    Args:
        request: Request object
        file: ZIP file containing DICOM images

    Returns:
        JSONResponse: prediction result with the risk score and the paths to the result images
    """
    logger.info("API predict_zip called")

    # Check the file format
    if not file or file.filename == "":
        return JSONResponse({"error": ERROR_MESSAGES["invalid_file"]}, status_code=400)

    if not file.filename.endswith(".zip"):
        return JSONResponse(
            {"error": "Invalid file format. Only ZIP is allowed."}, status_code=400
        )

    logger.info(f"File upload: {file.filename}")

    # Create a unique UUID for each request
    session_id = str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")

    # Create the folders for this session
    dicom_uuid_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    result_uuid_dir = os.path.join(FOLDERS["RESULTS"], session_id)

    try:
        # Create the folders
        os.makedirs(dicom_uuid_dir, exist_ok=True)
        os.makedirs(result_uuid_dir, exist_ok=True)

        # Path of the temporary ZIP file
        zip_path = os.path.join(FOLDERS["UPLOAD"], f"{session_id}.zip")

        # Save the ZIP file
        content = await file.read()
        with open(zip_path, "wb") as temp_zip:
            temp_zip.write(content)

        # Extract the ZIP file
        try:
            logger.info(f"Extracting file {file.filename} to {dicom_uuid_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dicom_uuid_dir)

            # Delete the ZIP file after extraction
            os.remove(zip_path)
        except zipfile.BadZipFile:
            os.remove(zip_path)
            return JSONResponse({"error": "Invalid ZIP file"}, status_code=400)

        # Check the files after extraction
        valid_files = []
        for root, _, files in os.walk(dicom_uuid_dir):
            for filename in files:
                if filename.lower().endswith((".dcm", ".png")):
                    valid_files.append(os.path.join(root, filename))

        if not valid_files:
            shutil.rmtree(dicom_uuid_dir)  # Remove the empty folder
            return JSONResponse(
                {"error": "No valid files found in the ZIP archive"}, status_code=400
            )

        logger.info(f"Found {len(valid_files)} valid files")

        # Find the sub-folder that contains the DICOM files (if any)
        dicom_dir = dicom_uuid_dir
        for root, _, files in os.walk(dicom_uuid_dir):
            if any(file.endswith(".dcm") for file in files):
                dicom_dir = root
                logger.info(f"Found directory containing DICOM files: {root}")
                break

        # Check if models are available
        if model is None:
            return JSONResponse(
                {"error": ERROR_MESSAGES["model_not_found"]}, status_code=500
            )

        # Run prediction
        try:
            pred_dict, attention_info, gif_path = await run_in_threadpool(
                predict,
                dicom_dir=dicom_dir,
                output_dir=result_uuid_dir,
                heart_detector=heart_detector,
                model=model,
                session_id=session_id,
                create_gif=True,
            )

            # Check that the results folder exists and has images
            overlay_files = os.listdir(result_uuid_dir)
            if not overlay_files:
                return JSONResponse(
                    {"error": "No overlay images generated"}, status_code=500
                )

            logger.info(
                f"Found {len(overlay_files)} overlay images in {result_uuid_dir}"
            )

            # Compress the results into a ZIP file
            try:
                zip_path = create_zip_result(result_uuid_dir, session_id)
                logger.info(f"Created ZIP file at: {zip_path}")

                if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                    return JSONResponse(
                        {"error": "Failed to create zip file"}, status_code=500
                    )

            except Exception as e:
                logger.error(f"Error creating ZIP file: {str(e)}")
                return JSONResponse(
                    {"error": f"Failed to create zip file: {str(e)}"}, status_code=500
                )

            # Build the URLs for the ZIP and GIF files
            base_url = str(request.base_url).rstrip("/")
            zip_download_link = f"{base_url}/download_zip/{session_id}"
            gif_download_link = (
                f"{base_url}/download_gif/{session_id}" if gif_path else None
            )

            # Build the response
            response = {
                "session_id": session_id,
                "predictions": pred_dict["predictions"],
                "overlay_images": zip_download_link,
                "overlay_gif": gif_download_link,
                "attention_info": attention_info,
                "message": "Prediction successful.",
                "model_version": _case_version(pred_dict),
            }

            logger.info(f"Prediction {session_id} successful.")
            return JSONResponse(response)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                {"error": f"Processing error: {str(e)}"}, status_code=500
            )

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)


@router.get("/download_zip/{session_id}")
async def download_zip(session_id: str):
    """API to download the ZIP of overlay images for a session ID"""
    file_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.zip")
    if os.path.exists(file_path):
        logger.info(f"✅ File found: {file_path}, preparing download...")
        return FileResponse(file_path, filename=f"{session_id}_results.zip")

    logger.warning(f"⚠️ File not found: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id}, status_code=404
    )


@router.get("/download_gif/{session_id}")
async def download_gif(session_id: str):
    """API to download the GIF of overlay images for a session ID"""
    # New path: the GIF file is inside the session_id folder
    session_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    file_path = os.path.join(session_dir, "results.gif")

    if os.path.exists(file_path):
        logger.info(f"✅ GIF file found: {file_path}, preparing download...")
        return FileResponse(
            file_path, filename=f"{session_id}_results.gif", media_type="image/gif"
        )

    # Check the old path for backward compatibility (if needed)
    old_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.gif")
    if os.path.exists(old_path):
        logger.info(f"✅ GIF file found at old path: {old_path}, preparing download...")
        return FileResponse(
            old_path, filename=f"{session_id}_results.gif", media_type="image/gif"
        )

    logger.warning(f"⚠️ GIF file not found: {file_path}")
    return JSONResponse(
        {"error": "GIF file not found", "session_id": session_id}, status_code=404
    )


@router.get("/preview/{session_id}/{filename}")
async def preview_file(session_id: str, filename: str):
    """API to preview an overlay image"""
    overlay_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    file_path = os.path.join(overlay_dir, filename)

    if os.path.exists(file_path):
        logger.info(f"✅ Preview file: {file_path}")
        return FileResponse(file_path)

    logger.warning(f"⚠️ Preview file not found: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id, "filename": filename},
        status_code=404,
    )
