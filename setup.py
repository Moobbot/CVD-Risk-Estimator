#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for CVD Risk Estimator.
This script installs required packages and downloads model checkpoints.
"""

import argparse
import logging
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 7)

# Model download URLs from Dropbox
# Note: We add 'dl=1' to the end of Dropbox URLs to force direct download
MODEL_DOWNLOAD_URLS = {
    "retinanet_heart.pt": "https://www.dropbox.com/scl/fi/awfnv4elf1d9y9ca9kg8c/retinanet_heart.pt?rlkey=6exxr989ww6zs0cvosepw84sb&st=rpkioyoz&dl=1",
    "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm": "https://www.dropbox.com/scl/fi/egwtns5xrbasg1i6dse19/NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm?rlkey=8csdx2h4dcoxwfo03k59qla94&st=kc1pa3t3&dl=1",
}

# PyTorch versions
PYTORCH_VERSIONS = {
    "cpu": ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"],
    "cuda": ["torch", "torchvision", "torchaudio"],
}

# CUDA versions
CUDA_VERSIONS = {
    "11.8": "cu118",
    "12.1": "cu121",
}

# Default CUDA version
DEFAULT_CUDA_VERSION = "12.1"


def check_python_version() -> bool:
    """
    Check if the current Python version meets the minimum requirements.

    Returns:
        bool: True if the Python version is sufficient, False otherwise.
    """
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        logger.error(
            f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required. "
            f"You are using Python {current_version[0]}.{current_version[1]}."
        )
        return False
    return True


def download_file(url: str, destination: str, show_progress: bool = True) -> bool:
    """
    Download a file from a URL with progress reporting.

    Args:
        url: URL of the file to download
        destination: Path where the file will be saved
        show_progress: Whether to show download progress

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Install requests if not already installed
        try:
            import requests
        except ImportError:
            logger.info("Installing requests package...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "requests"], check=True
            )
            import requests

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        # Check if file already exists
        if os.path.exists(destination):
            logger.info(f"File already exists: {destination}")
            return True

        # Download the file with a session to handle redirects
        logger.info(f"Downloading file from: {url}")
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Get total file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Save the file with progress reporting
        with open(destination, "wb") as f:
            if total_size > 0 and show_progress:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 5%
                        if downloaded % (total_size // 20) < 8192:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {percent:.1f}%")
            else:
                # If content-length is not available or progress not needed
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Successfully downloaded file: {destination}")
        return True

    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False


def download_and_extract_zip(url: str, extract_path: str = ".") -> bool:
    """
    Download a ZIP file from the given URL and extract its contents.

    Args:
        url: The URL of the ZIP file to download
        extract_path: The path where the contents of the ZIP file will be extracted

    Returns:
        bool: True if the download and extraction were successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Get the filename from the URL
        filename = url.split("/")[-1]
        temp_file = os.path.join(extract_path, f"temp_{filename}")

        # Download the ZIP file
        if not download_file(url, temp_file):
            return False

        # Extract the ZIP file
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Remove the temporary file
        os.remove(temp_file)

        logger.info(f"{filename} extracted successfully to {extract_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading or extracting ZIP file: {e}")
        return False


def check_gpu():
    """
    Kiểm tra xem máy có GPU hay không mà không cần PyTorch.
    Hỗ trợ Windows, Linux, macOS và môi trường Docker.

    Returns:
        list: Danh sách các GPU được phát hiện hoặc None nếu không tìm thấy.
    """
    try:
        # Kiểm tra xem có đang chạy trong Docker không
        in_docker = (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER_CONTAINER") == "true"
        )
        if in_docker:
            logger.info("Running in Docker container")
            gpus= ["GPU in Docker"]
            return None

        logger.info(
            f"Checking GPU in environment: {platform.system()}"
            + (" (Docker)" if in_docker else "")
        )

        # === LINUX và DOCKER ===
        if platform.system() == "Linux" or in_docker:
            gpus = []
            logger.info("Checking GPU using Linux methods...")

            # Phương pháp 1: Kiểm tra qua nvidia-smi (tốt nhất cho Docker với NVIDIA GPU)
            try:
                logger.debug("Trying nvidia-smi method...")
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    universal_newlines=True,
                    stderr=subprocess.DEVNULL,
                )
                nvidia_gpus = [
                    line.strip() for line in output.split("\n") if line.strip()
                ]
                if nvidia_gpus:
                    logger.info(f"Found {len(nvidia_gpus)} GPUs via nvidia-smi")
                    return nvidia_gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("nvidia-smi method failed")
                pass

            # Phương pháp 2: Kiểm tra qua lspci
            try:
                logger.debug("Trying lspci method...")
                output = subprocess.check_output(
                    ["lspci"], universal_newlines=True, stderr=subprocess.DEVNULL
                )
                gpu_lines = [
                    line
                    for line in output.split("\n")
                    if "VGA" in line or "3D controller" in line
                ]
                if gpu_lines:
                    logger.info(f"Found {len(gpu_lines)} GPUs via lspci")
                    return gpu_lines
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("lspci method failed")
                pass

            # Phương pháp 3: Kiểm tra thư mục /proc/driver/nvidia
            if os.path.exists("/proc/driver/nvidia/gpus"):
                try:
                    logger.debug("Checking /proc/driver/nvidia/gpus...")
                    gpu_dirs = os.listdir("/proc/driver/nvidia/gpus")
                    if gpu_dirs:
                        logger.info(
                            f"Found {len(gpu_dirs)} GPUs via /proc/driver/nvidia/gpus"
                        )
                        return [f"NVIDIA GPU #{i}" for i in range(len(gpu_dirs))]
                except Exception:
                    logger.debug("/proc/driver/nvidia/gpus method failed")
                    pass

            # Phương pháp 4: Kiểm tra qua /dev/nvidia*
            try:
                logger.debug("Checking /dev/nvidia* devices...")
                nvidia_devices = [
                    dev
                    for dev in os.listdir("/dev")
                    if dev.startswith("nvidia")
                    and dev != "nvidiactl"
                    and dev != "nvidia-modeset"
                ]
                if nvidia_devices:
                    logger.info(
                        f"Found {len(nvidia_devices)} GPUs via /dev/nvidia* devices"
                    )
                    return [f"NVIDIA GPU device: {dev}" for dev in nvidia_devices]
            except (FileNotFoundError, PermissionError):
                logger.debug("/dev/nvidia* method failed")
                pass

        # === WINDOWS ===
        elif platform.system() == "Windows":
            logger.info("Checking GPU using Windows methods...")
            # Phương pháp 1: Sử dụng WMIC
            try:
                logger.debug("Trying WMIC method...")
                output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if line.strip() and "Name" not in line
                ]
                if gpus:
                    logger.info(f"Found {len(gpus)} GPUs via WMIC")
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("WMIC method failed")
                pass

            # Phương pháp 2: Sử dụng PowerShell nếu WMIC thất bại
            try:
                logger.debug("Trying PowerShell method...")
                output = subprocess.check_output(
                    [
                        "powershell",
                        "Get-WmiObject Win32_VideoController | Select-Object Name",
                    ],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if line.strip() and "Name" not in line and "----" not in line
                ]
                if gpus:
                    logger.info(f"Found {len(gpus)} GPUs via PowerShell")
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("PowerShell method failed")
                pass

        # === macOS ===
        elif platform.system() == "Darwin":
            logger.info("Checking GPU using macOS methods...")
            try:
                logger.debug("Trying system_profiler method...")
                output = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                # Tìm kiếm dòng có "Chipset Model" và lấy tên GPU
                gpu_pattern = re.compile(r"Chipset Model: (.+)")
                matches = gpu_pattern.findall(output)
                if matches:
                    logger.info(
                        f"Found {len(matches)} GPUs via system_profiler (regex)"
                    )
                    return [f"Chipset Model: {match}" for match in matches]

                # Phương pháp thay thế nếu regex không hoạt động
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if "Chipset Model" in line
                ]
                if gpus:
                    logger.info(
                        f"Found {len(gpus)} GPUs via system_profiler (line search)"
                    )
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("system_profiler method failed")
                pass

        # === Phương pháp cuối cùng: Kiểm tra biến môi trường ===
        logger.info("Checking GPU using environment variables...")
        # Kiểm tra biến môi trường CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices and cuda_devices != "-1":
            logger.info(f"Found GPUs via CUDA_VISIBLE_DEVICES: {cuda_devices}")
            return [f"CUDA Device #{dev}" for dev in cuda_devices.split(",")]

        # Kiểm tra biến môi trường GPU_DEVICE_ORDINAL (cho ROCm/AMD)
        rocm_devices = os.environ.get("GPU_DEVICE_ORDINAL")
        if rocm_devices:
            logger.info(f"Found GPUs via GPU_DEVICE_ORDINAL: {rocm_devices}")
            return [f"ROCm Device #{dev}" for dev in rocm_devices.split(",")]

        logger.warning("No GPU detected or unsupported operating system.")
        return None

    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return None


def install_packages(
    cuda_version: str = DEFAULT_CUDA_VERSION, force_cpu: bool = False
) -> bool:
    """
    Install required packages including PyTorch.

    Args:
        cuda_version: CUDA version to use (e.g., '11.8', '12.1')
        force_cpu: Force CPU installation even if GPU is detected

    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # Install requests if not already installed
        try:
            import requests
        except ImportError:
            logger.info("Installing requests package...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "requests"], check=True
            )

        # Check for GPU
        gpus = None if force_cpu else ["check_gpu()"]

        # Install PyTorch
        logger.info("Installing PyTorch and related packages...")
        if gpus:
            logger.info("GPU detected:")
            for gpu in gpus:
                logger.info(f" - {gpu}")

            # Validate CUDA version
            if cuda_version not in CUDA_VERSIONS:
                logger.warning(
                    f"Unsupported CUDA version: {cuda_version}. Using default: {DEFAULT_CUDA_VERSION}"
                )
                cuda_version = DEFAULT_CUDA_VERSION

            cuda_suffix = CUDA_VERSIONS[cuda_version]
            logger.info(f"Installing PyTorch with CUDA {cuda_version} support...")

            # Install PyTorch with CUDA support
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                *PYTORCH_VERSIONS["cuda"],
                "--index-url",
                f"https://download.pytorch.org/whl/{cuda_suffix}",
            ]
            result = subprocess.run(cmd, check=True)

            if result.returncode != 0:
                logger.error(
                    "Failed to install PyTorch with CUDA support. Falling back to CPU version."
                )
                cmd = [sys.executable, "-m", "pip", "install", *PYTORCH_VERSIONS["cpu"]]
                subprocess.run(cmd, check=True)
        else:
            logger.info(
                "No GPU detected or CPU version forced. Installing CPU version of PyTorch..."
            )
            cmd = [sys.executable, "-m", "pip", "install", *PYTORCH_VERSIONS["cpu"]]
            subprocess.run(cmd, check=True)

        # Install other requirements
        logger.info("Installing other requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during package installation: {e}")
        return False


def download_model_checkpoints(skip_download: bool = False) -> bool:
    """
    Download model checkpoints from Dropbox.

    Args:
        skip_download: Skip downloading model checkpoints

    Returns:
        bool: True if all model files are available, False otherwise
    """
    if skip_download:
        logger.info("Skipping model checkpoint download as requested.")
        return True

    logger.info("Checking and downloading model checkpoints from Dropbox...")

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoint"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # List of model files to download
    model_files = list(MODEL_DOWNLOAD_URLS.keys())

    # Download each model file
    for model_file in model_files:
        local_file_path = os.path.join(checkpoint_dir, model_file)

        # Check if file already exists
        if os.path.exists(local_file_path):
            logger.info(f"Model file already exists: {local_file_path}")
            continue

        # Get the download URL from the dictionary
        if model_file in MODEL_DOWNLOAD_URLS:
            download_url = MODEL_DOWNLOAD_URLS[model_file]
            download_file(download_url, local_file_path)
        else:
            logger.warning(f"No download URL defined for model file: {model_file}")

    # Check if all model files were downloaded successfully
    missing_files = []
    for model_file in model_files:
        local_file_path = os.path.join(checkpoint_dir, model_file)
        if not os.path.exists(local_file_path):
            missing_files.append(model_file)

    if missing_files:
        logger.warning("Some model files could not be downloaded automatically:")
        for missing_file in missing_files:
            logger.warning(f" - {missing_file}")
        logger.info(
            "Please download these files manually from Dropbox and place them in the checkpoint directory."
        )
        return False
    else:
        logger.info("All model files are available in the checkpoint directory.")
        return True


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Setup script for CVD Risk Estimator")

    parser.add_argument(
        "--skip-packages", action="store_true", help="Skip package installation"
    )

    parser.add_argument(
        "--skip-models", action="store_true", help="Skip model checkpoint download"
    )

    parser.add_argument(
        "--cuda-version",
        type=str,
        choices=list(CUDA_VERSIONS.keys()),
        default=DEFAULT_CUDA_VERSION,
        help=f"CUDA version to use (default: {DEFAULT_CUDA_VERSION})",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU installation even if GPU is detected",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main() -> int:
    """
    Main function to run the setup process.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Print welcome message
    logger.info("Starting CVD Risk Estimator setup...")

    # Check Python version
    if not check_python_version():
        return 1

    # Install packages
    if not args.skip_packages:
        logger.info("Installing packages...")
        if not install_packages(args.cuda_version, args.force_cpu):
            logger.error("Package installation failed.")
            return 1
    else:
        logger.info("Skipping package installation as requested.")

    # Download model checkpoints
    if not args.skip_models:
        logger.info("Downloading model checkpoints...")
        if not download_model_checkpoints():
            logger.warning("Some model checkpoints could not be downloaded.")
            # Continue anyway, as the user might download them manually
    else:
        logger.info("Skipping model checkpoint download as requested.")

    logger.info("Setup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
