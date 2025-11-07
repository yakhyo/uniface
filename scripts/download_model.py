import argparse
from uniface.constants import RetinaFaceWeights
from uniface.model_store import verify_model_weights


def main():
    parser = argparse.ArgumentParser(description="Download and verify RetinaFace model weights.")
    parser.add_argument(
        "--model",
        type=str,
        choices=[m.name for m in RetinaFaceWeights],
        help="Model to download (e.g. MNET_V2). If not specified, all models will be downloaded.",
    )
    args = parser.parse_args()

    if args.model:
        weight = RetinaFaceWeights[args.model]
        print(f"📥 Downloading model: {weight.value}")
        verify_model_weights(weight)  # Pass enum, not string
    else:
        print("📥 Downloading all models...")
        for weight in RetinaFaceWeights:
            verify_model_weights(weight)  # Pass enum, not string

    print("✅ All requested weights are ready and verified.")


if __name__ == "__main__":
    main()


