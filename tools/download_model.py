import argparse
from enum import Enum
import sys

from uniface.constants import MODEL_REGISTRY
from uniface.model_store import verify_model_weights


def _model_types() -> dict[str, type[Enum]]:
    """Map a CLI name to each weight enum in the registry, e.g. 'retinaface' -> RetinaFaceWeights."""
    types: dict[str, type[Enum]] = {}
    for weight in MODEL_REGISTRY:
        name = type(weight).__name__.removesuffix('Weights').lower()
        types.setdefault(name, type(weight))
    return dict(sorted(types.items()))


MODEL_TYPES = _model_types()

# Older names kept so existing commands keep working
ALIASES = {'ddamfn': 'emotion'}


def download_models(model_enum: type[Enum]) -> int:
    """Download every weight of one model family and return the number of failures."""
    failures = 0
    for weight in model_enum:
        print(f'Downloading: {weight.value}')
        try:
            verify_model_weights(weight)
            print(f'  Done: {weight.value}')
        except Exception as e:
            failures += 1
            print(f'  Failed: {e}')
    return failures


def main():
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=[*MODEL_TYPES, *ALIASES],
        help='Model family to download. If not specified, downloads every model.',
    )
    args = parser.parse_args()

    if args.model_type:
        name = ALIASES.get(args.model_type, args.model_type)
        print(f'Downloading {name} models...')
        failures = download_models(MODEL_TYPES[name])
    else:
        print('Downloading all models...')
        failures = 0
        for name, model_enum in MODEL_TYPES.items():
            print(f'\n{name}:')
            failures += download_models(model_enum)

    if failures:
        print(f'\n{failures} download(s) failed.')
        sys.exit(1)
    print('\nDone!')


if __name__ == '__main__':
    main()
