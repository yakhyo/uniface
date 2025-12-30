import argparse
import hashlib
from pathlib import Path


def compute_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    sha256_hash = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Compute SHA256 hash of a file')
    parser.add_argument('file', type=Path, help='Path to file')
    args = parser.parse_args()

    if not args.file.exists() or not args.file.is_file():
        print(f'File does not exist: {args.file}')
        return

    sha256 = compute_sha256(args.file)
    print(f"SHA256 hash for '{args.file.name}':\n{sha256}")


if __name__ == '__main__':
    main()
