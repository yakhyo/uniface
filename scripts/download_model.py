import argparse
from uniface.constants import (
    RetinaFaceWeights, SphereFaceWeights, MobileFaceWeights, ArcFaceWeights,
    SCRFDWeights, DDAMFNWeights, AgeGenderWeights, LandmarkWeights
)
from uniface.model_store import verify_model_weights


# All available model types
ALL_MODEL_TYPES = {
    'retinaface': RetinaFaceWeights,
    'sphereface': SphereFaceWeights,
    'mobileface': MobileFaceWeights,
    'arcface': ArcFaceWeights,
    'scrfd': SCRFDWeights,
    'ddamfn': DDAMFNWeights,
    'agegender': AgeGenderWeights,
    'landmark': LandmarkWeights,
}


def main():
    parser = argparse.ArgumentParser(description="Download and verify model weights.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(ALL_MODEL_TYPES.keys()),
        help="Model type to download (e.g. retinaface, arcface). If not specified, all models will be downloaded.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download (e.g. MNET_V2). For RetinaFace backward compatibility.",
    )
    args = parser.parse_args()

    if args.model and not args.model_type:
        # Backward compatibility - assume RetinaFace
        try:
            weight = RetinaFaceWeights[args.model]
            print(f"Downloading RetinaFace model: {weight.value}")
            verify_model_weights(weight)
            print("Model downloaded successfully.")
        except KeyError:
            print(f"Invalid RetinaFace model: {args.model}")
            print(f"Available models: {[m.name for m in RetinaFaceWeights]}")
        return

    if args.model_type:
        # Download all models from specific type
        model_enum = ALL_MODEL_TYPES[args.model_type]
        print(f"Downloading all {args.model_type} models...")
        for weight in model_enum:
            print(f"Downloading: {weight.value}")
            try:
                verify_model_weights(weight)
                print(f"Downloaded: {weight.value}")
            except Exception as e:
                print(f"Failed to download {weight.value}: {e}")
    else:
        # Download all models from all types
        print("Downloading all models...")
        for model_type, model_enum in ALL_MODEL_TYPES.items():
            print(f"\nDownloading {model_type} models...")
            for weight in model_enum:
                print(f"Downloading: {weight.value}")
                try:
                    verify_model_weights(weight)
                    print(f"Downloaded: {weight.value}")
                except Exception as e:
                    print(f"Failed to download {weight.value}: {e}")

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()
