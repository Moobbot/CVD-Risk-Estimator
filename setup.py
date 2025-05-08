import io
import logging
import platform
import zipfile
import os
import subprocess
import re
import json

try:
    import requests
except ModuleNotFoundError as e:
    # print("Requests module not found:", e)
    subprocess.run(["pip", "install", "requests"])
    import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model download URLs from Dropbox
# Note: We add 'dl=1' to the end of Dropbox URLs to force direct download
MODEL_DOWNLOAD_URLS = {
    "retinanet_heart.pt": "https://www.dropbox.com/scl/fi/awfnv4elf1d9y9ca9kg8c/retinanet_heart.pt?rlkey=6exxr989ww6zs0cvosepw84sb&st=rpkioyoz&dl=1",
    "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm": "https://www.dropbox.com/scl/fi/egwtns5xrbasg1i6dse19/NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm?rlkey=8csdx2h4dcoxwfo03k59qla94&st=kc1pa3t3&dl=1",
}


def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.

    Args:
    - url (str): URL of the file to download.
    - folder (str): Path to the folder where the file will be saved.
    - filename (str): Name of the file to be saved.

    Returns:
    - str: Full path of the downloaded file, or None if the file already exists.
    """

    # Tạo đường dẫn đầy đủ của tệp
    filepath = os.path.join(folder, filename)

    # Kiểm tra xem tệp đã tồn tại không
    if os.path.exists(filepath):
        print("Tệp đã tồn tại:", filepath)
        return filepath

    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded and saved to: {filepath}")
        return filepath
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        return None


def download_and_extract_zip(url, extract_path="."):
    """
    Download a ZIP file from the given URL and extract its contents.

    Args:
    - url (str): The URL of the ZIP file to download.
    - extract_path (str): The path where the contents of the ZIP file will be extracted. Default is the current directory.

    Returns:
    - bool: True if the download and extraction were successful, False otherwise.
    """
    try:
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        # Get the filename from the URL
        filename = url.split("/")[-1]

        # Check if the file already exists in the destination folder
        if os.path.exists(os.path.join(extract_path, filename)):
            print(f"{filename} already exists. Skipping download.")
            return True

        # Download the ZIP file
        logger.info(f"Downloading {filename}...")
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"{filename} downloaded and extracted successfully.")
        return True
    except (requests.RequestException, zipfile.BadZipFile, Exception) as e:
        logger.error(f"An error occurred: {e}")
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

        # === LINUX và DOCKER ===
        if platform.system() == "Linux" or in_docker:
            gpus = []

            # Phương pháp 1: Kiểm tra qua nvidia-smi (tốt nhất cho Docker với NVIDIA GPU)
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    universal_newlines=True,
                    stderr=subprocess.DEVNULL,
                )
                nvidia_gpus = [
                    line.strip() for line in output.split("\n") if line.strip()
                ]
                if nvidia_gpus:
                    return nvidia_gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # Phương pháp 2: Kiểm tra qua lspci
            try:
                output = subprocess.check_output(
                    ["lspci"], universal_newlines=True, stderr=subprocess.DEVNULL
                )
                gpu_lines = [
                    line
                    for line in output.split("\n")
                    if "VGA" in line or "3D controller" in line
                ]
                if gpu_lines:
                    return gpu_lines
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # Phương pháp 3: Kiểm tra thư mục /proc/driver/nvidia
            if os.path.exists("/proc/driver/nvidia/gpus"):
                try:
                    gpu_dirs = os.listdir("/proc/driver/nvidia/gpus")
                    if gpu_dirs:
                        return [f"NVIDIA GPU #{i}" for i in range(len(gpu_dirs))]
                except Exception:
                    pass

            # Phương pháp 4: Kiểm tra qua /dev/nvidia*
            try:
                nvidia_devices = [
                    dev
                    for dev in os.listdir("/dev")
                    if dev.startswith("nvidia")
                    and dev != "nvidiactl"
                    and dev != "nvidia-modeset"
                ]
                if nvidia_devices:
                    return [f"NVIDIA GPU device: {dev}" for dev in nvidia_devices]
            except (FileNotFoundError, PermissionError):
                pass

        # === WINDOWS ===
        elif platform.system() == "Windows":
            # Phương pháp 1: Sử dụng WMIC
            try:
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
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # Phương pháp 2: Sử dụng PowerShell nếu WMIC thất bại
            try:
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
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # === macOS ===
        elif platform.system() == "Darwin":
            try:
                output = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                # Tìm kiếm dòng có "Chipset Model" và lấy tên GPU
                gpu_pattern = re.compile(r"Chipset Model: (.+)")
                matches = gpu_pattern.findall(output)
                if matches:
                    return [f"Chipset Model: {match}" for match in matches]

                # Phương pháp thay thế nếu regex không hoạt động
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if "Chipset Model" in line
                ]
                if gpus:
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # === Phương pháp cuối cùng: Kiểm tra biến môi trường ===
        # Kiểm tra biến môi trường CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices and cuda_devices != "-1":
            return [f"CUDA Device #{dev}" for dev in cuda_devices.split(",")]

        # Kiểm tra biến môi trường GPU_DEVICE_ORDINAL (cho ROCm/AMD)
        rocm_devices = os.environ.get("GPU_DEVICE_ORDINAL")
        if rocm_devices:
            return [f"ROCm Device #{dev}" for dev in rocm_devices.split(",")]

        print("Không phát hiện được GPU hoặc không hỗ trợ hệ điều hành này.")
        return None

    except Exception as e:
        print(f"Lỗi khi kiểm tra GPU: {e}")
        return None


def install_packages():
    print("Install Torch")
    packages = ["torch", "torchvision", "torchaudio"]
    # Only use with Windows
    if check_gpu():
        print("GPU detected:")
        for gpu in check_gpu():
            print(f" - {gpu}")
        print("Installing PyTorch with CUDA support...")
        subprocess.run(
            [
                "pip",
                "install",
                *packages,
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    else:
        packages = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        subprocess.run(["pip", "install", *packages])
        print("No GPU detected, installing CPU version of PyTorch...")

    print("Install requirements")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


def download_model_checkpoints():
    """
    Download model checkpoints from Dropbox.
    This function downloads the necessary model files from Dropbox
    and places them in the correct locations.
    """
    logger.info("Checking and downloading model checkpoints from Dropbox...")

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoint"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # List of model files to download
    model_files = [
        "retinanet_heart.pt",
        "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm",
    ]

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
            logger.info(f"Downloading model file: {model_file} from Dropbox...")

            try:
                # Download the file with a session to handle redirects
                session = requests.Session()
                response = session.get(download_url, stream=True)
                response.raise_for_status()

                # Get total file size if available
                total_size = int(response.headers.get("content-length", 0))

                # Save the file with progress reporting
                with open(local_file_path, "wb") as f:
                    if total_size > 0:
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
                        # If content-length is not available
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                logger.info(f"Successfully downloaded model file: {local_file_path}")
            except Exception as e:
                logger.error(f"Error downloading model file {model_file}: {e}")
                logger.info(
                    f"Please download {model_file} manually from Dropbox and place it in the checkpoint directory."
                )
        else:
            logger.warning(f"No download URL defined for model file: {model_file}")
            logger.info(
                f"Please download {model_file} manually from Dropbox and place it in the checkpoint directory."
            )

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
    else:
        logger.info("All model files are available in the checkpoint directory.")


def main():
    print("Installing packages...")
    install_packages()

    print("Downloading model checkpoints...")
    download_model_checkpoints()

    print("Setup completed successfully!")


if __name__ == "__main__":
    main()
