#!/bin/bash

# Exit on errors
set -e

cd "$(dirname "$0")"/..

echo "Deleting existing release-related files..."
rm -rf dist/ build/ *.egg-info

echo "Creating a package for the current release (PyPI compatible)..."
python3 setup.py sdist bdist_wheel

echo "Release package created successfully in the 'dist/' folder."


# echo "Uploading the package to PyPI..."
# twine upload dist/*

# echo "Release uploaded successfully!"