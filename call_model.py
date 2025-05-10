import os
import traceback
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from config import MODEL_CONFIG, FOLDERS
from tri_2d_net.init_model import init_model
from heart_detector import HeartDetector
from image import Image
from logger import setup_logger

# Set up logger with date-based organization
logger = setup_logger("call_model")


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
        else:
            logger.info("Heart detector model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading heart detection model: {str(e)}")
        logger.error(traceback.format_exc())
        # Continue with heart_detector = None, simple method will be used

    # Try to load CVD risk prediction model
    try:
        logger.info("Loading cardiovascular risk prediction model...")
        model = init_model()
        model.load_model(MODEL_CONFIG["ITER"])
        logger.info("CVD risk prediction model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load CVD risk prediction model: {str(e)}")
        logger.error(traceback.format_exc())
        # This is a critical error, but we'll return None and handle it in the API
        model = None

    # Return both models, even if one or both are None
    # The API will check and handle appropriately
    return heart_detector, model


def process_attention_scores(
    cam_data: np.ndarray, 
    heart_indices: List[int], 
    dicom_names: List[str],
    top_k: int = None
) -> Dict[str, Any]:
    """Process and rank attention scores.

    Args:
        cam_data: Model attention data (Grad-CAM)
        heart_indices: Indices of heart slices
        dicom_names: Names of DICOM files
        top_k: Number of top images to return (None for all)

    Returns:
        dict: Processed attention information containing:
            - attention_scores: List of dicts with file info and scores
            - total_images: Total number of images processed
            - returned_images: Number of images returned
    """
    attention_scores = []

    for idx, orig_idx in enumerate(heart_indices):
        if idx >= len(cam_data):
            break

        # Get corresponding slice from cam_data
        cam_slice = cam_data[idx]

        # Calculate attention score (average value in slice)
        score = float(np.mean(cam_slice))

        if score > 0:
            attention_scores.append(
                {
                    "file_name_pred": f"{orig_idx}_{dicom_names[orig_idx]}.png",
                    "attention_score": score,
                }
            )

    # Sort by attention score in descending order
    attention_scores.sort(key=lambda x: x["attention_score"], reverse=True)

    # Limit to top_k if specified
    if top_k is not None and top_k > 0:
        attention_scores = attention_scores[:top_k]

    # Create result
    result = {
        "attention_scores": attention_scores,
        "total_images": len(heart_indices),
        "returned_images": len(attention_scores),
    }

    return result


def predict(
    dicom_dir: str,
    output_dir: str,
    heart_detector: HeartDetector,
    model,
    session_id: str,
    create_gif: bool = True
) -> Tuple[Dict, Dict, Optional[str]]:
    """Run the model prediction.

    Args:
        dicom_dir: Directory containing DICOM files
        output_dir: Directory to save results
        heart_detector: Heart detector model
        model: CVD risk prediction model
        session_id: Unique session ID
        create_gif: Whether to create a GIF of the results

    Returns:
        tuple: (prediction_dict, attention_info, gif_path)
    """
    # Check if model is available
    if model is None:
        logger.error("CVD risk prediction model is not available")
        raise ValueError("CVD risk prediction model is not available")

    # Read DICOM images and detect heart
    logger.info(f"Reading DICOM images from directory: {dicom_dir}")
    img = Image(dicom_dir, heart_detector)

    # Detect heart
    logger.info("Detecting heart...")
    if not img.detect_heart():
        logger.error("Could not detect heart in DICOM images")
        raise ValueError("Could not detect heart in DICOM images")

    # Convert to model input
    logger.info("Converting to model input...")
    network_input = img.to_network_input()

    # Predict risk score
    logger.info("Estimating cardiovascular risk...")
    pred_result = model.aug_transform(network_input)
    score = pred_result[1].item()
    logger.info(f"Risk score: {score}")

    # Calculate and save Grad-CAM
    logger.info("Calculating and saving Grad-CAM...")
    cam_data = model.grad_cam_visual(network_input)

    # Save Grad-CAM images and create GIF
    success, gif_path = img.save_grad_cam_on_original(
        cam_data, output_dir, create_gif=create_gif, session_id=session_id
    )
    logger.info(f"Saved Grad-CAM to directory: {output_dir}")

    if not success:
        logger.error("Failed to save overlay images")
        raise ValueError("Failed to save overlay images")

    # Process attention scores
    heart_indices = [i for i, val in enumerate(img.bbox_selected) if val == 1]
    attention_info = process_attention_scores(
        cam_data, heart_indices, img.dicom_names
    )

    # Create prediction dictionary
    prediction_dict = {"predictions": [{"score": float(score)}]}

    return prediction_dict, attention_info, gif_path
