import argparse

from uniface.constants import (
    AgeGenderWeights,
    ArcFaceWeights,
    DDAMFNWeights,
    LandmarkWeights,
    MobileFaceWeights,
    RetinaFaceWeights,
    SCRFDWeights,
    SphereFaceWeights,
)
from uniface.model_store import verify_model_weights

MODEL_TYPES = {
    "retinaface": RetinaFaceWeights,
    "sphereface": SphereFaceWeights,
    "mobileface": MobileFaceWeights,
    "arcface": ArcFaceWeights,
    "scrfd": SCRFDWeights,
    "ddamfn": DDAMFNWeights,
    "agegender": AgeGenderWeights,
    "landmark": LandmarkWeights,
}


def download_models(model_enum):
    for weight in model_enum:
        print(f"Downloading: {weight.value}")
        try:
            verify_model_weights(weight)
            print(f"  Done: {weight.value}")
        except Exception as e:
            print(f"  Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_TYPES.keys()),
        help="Model type to download. If not specified, downloads all.",
    )
    args = parser.parse_args()

    if args.model_type:
        print(f"Downloading {args.model_type} models...")
        download_models(MODEL_TYPES[args.model_type])
    else:
        print("Downloading all models...")
        for name, model_enum in MODEL_TYPES.items():
            print(f"\n{name}:")
            download_models(model_enum)

    print("\nDone!")


if __name__ == "__main__":
    main()
