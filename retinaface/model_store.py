import os
import requests
import hashlib
from typing import Dict

from .log import logger


MODEL_URLS: Dict[str, str] = {
    'retinaface_mnet025': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx',
    'retinaface_mnet050': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.onnx',
    'retinaface_mnet_v1': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1.onnx',
    'retinaface_mnet_v2': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.onnx',
    'retinaface_r18': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r18.onnx',
    'retinaface_r34': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx'
}

MODEL_SHA256: Dict[str, str] = {
    'retinaface_mnet025': 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    'retinaface_mnet050': 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    'retinaface_mnet_v1': '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    'retinaface_mnet_v2': '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    'retinaface_r18': 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    'retinaface_r34': 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630'
}


CHUNK_SIZE = 8192


def verify_model_weights(model_name: str, root: str = '~/.retinaface/models') -> str:
    """
    Ensures model weights are available by downloading if missing and verifying integrity with a SHA-256 hash.

    Checks if the specified model weights file exists in `root`. If missing, downloads from a predefined URL.
    The file is then verified using its SHA-256 hash. If verification fails, the corrupted file is deleted,
    and an error is raised.

    Args:
        model_name (str): Name of the model weights to verify or download.
        root (str, optional): Directory to store the model weights. Defaults to '~/.retinaface/models'.

    Returns:
        str: Path to the verified model weights file.

    Raises:
        ValueError: If the model is not found or if verification fails.
        ConnectionError: If downloading the file fails.

    Examples:
        >>> # Download and verify 'retinaface_mnet025' weights
        >>> verify_model_weights('retinaface_mnet025')
        '/home/user/.retinaface/models/retinaface_mnet025.onnx'

        >>> # Use a custom directory
        >>> verify_model_weights('retinaface_r34', root='/custom/dir')
        '/custom/dir/retinaface_r34.onnx'
    """

    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    model_path = os.path.join(root, f'{model_name}.onnx')

    if not os.path.exists(model_path):
        url = MODEL_URLS.get(model_name)
        if not url:
            logger.error(f"No URL found for model '{model_name}'")
            raise ValueError(f"No URL found for model '{model_name}'")

        logger.info(f"Downloading '{model_name}' from {url}")
        download_file(url, model_path)
        logger.info(f"Successfully '{model_name}' downloaded to {model_path}")

    expected_hash = MODEL_SHA256.get(model_name)
    if expected_hash and not verify_file_hash(model_path, expected_hash):
        os.remove(model_path)  # Remove corrupted file
        logger.warning("Corrupted weight detected. Removing...")
        raise ValueError(f"Hash mismatch for '{model_name}'. The file may be corrupted; please try downloading again.")

    return model_path


def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL in chunks and save it to the destination path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download file from {url}. Error: {e}")


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            file_hash.update(chunk)
    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        logger.warning(f"Expected hash: {expected_hash}, but got: {actual_hash}")
    return actual_hash == expected_hash


if __name__ == "__main__":
    model_names = [
        'retinaface_mnet025',
        'retinaface_mnet050',
        'retinaface_mnet_v1',
        'retinaface_mnet_v2',
        'retinaface_r18',
        'retinaface_r34'
    ]

    # Download each model in the list
    for model_name in model_names:
        model_path = verify_model_weights(model_name)
