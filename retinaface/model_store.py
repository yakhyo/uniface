import os
import requests
import hashlib

_model_urls = {
    'retinaface_mnet025': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx',
    'retinaface_mnet050': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.onnx',
    'retinaface_mnet_v1': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1.onnx',
    'retinaface_mnet_v2': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.onnx',
    'retinaface_r18': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r18.onnx',
    'retinaface_r34': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx'
}

_model_sha256 = {
    'retinaface_mnet025': 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    'retinaface_mnet050': 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    'retinaface_mnet_v1': '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    'retinaface_mnet_v2': '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    'retinaface_r18': 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    'retinaface_r34': 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630'
}


def download(model_name, root='~/.retinaface/models'):
    # Expand user directory and set model path
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    model_path = os.path.join(root, f'{model_name}.onnx')

    # Download model if it does not exist
    if not os.path.exists(model_path):
        url = _model_urls.get(model_name)
        if not url:
            raise ValueError(f"No URL found for model {model_name}")
        print(f"Downloading {model_name} from {url}...")

        # Download chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"{model_name} downloaded to {model_path}")

    # Verify hash
    expected_hash = _model_sha256.get(model_name)
    if expected_hash:
        if not verify_file_hash(model_path, expected_hash):
            os.remove(model_path)
            raise ValueError(f"Hash mismatch for {model_name}. The file may be corrupted. Run it again!")

    return model_path


def verify_file_hash(file_path, expected_hash):
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest() == expected_hash


if __name__ == "__main__":
    # List of models to download
    models_to_download = [
        'retinaface_mnet025',
        'retinaface_mnet050',
        'retinaface_mnet_v1',
        'retinaface_mnet_v2',
        'retinaface_r18',
        'retinaface_r34'
    ]

    # Download each model in the list
    for model_name in models_to_download:
        try:
            model_path = download(model_name)
            print(f"Successfully downloaded {model_name} to {model_path}")
        except ValueError as e:
            print(f"Error downloading {model_name}: {e}")
