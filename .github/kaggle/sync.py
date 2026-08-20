# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Publish the notebooks in ``examples/`` to Kaggle as kernels.

Each notebook maps to one kernel slug in ``kernels.json`` beside this script. Pushing a known slug
adds a *new version* to that kernel rather than creating another one, so its upvotes, comments and
version history carry over; nothing is ever deleted or re-created.

The slug is not fixed, though. Kaggle re-slugs a kernel to match its title, so changing a title
moves the kernel's URL and the old address stops resolving (measured 2026-08-20: retitling
``face-detection-with-uniface-python-library`` moved it to ``face-detection-with-uniface``, which
then 404s, while the kernel itself kept its history). Renaming is therefore done by pushing the OLD
slug with the NEW title, and the manifest holds both: ``slug`` is where the kernel lives now,
``title`` is what it should be called. A mismatch on a live entry means a pending rename and is
reported, not rejected; on a new entry it is simply wrong, because a new kernel lands at
slugify(title) regardless. Keep the old title whenever the current URL matters more than the name.

Kaggle runs the kernel itself once the version lands, so this script pushes and exits without
waiting for the run to finish.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess  # nosec B404 - only ever runs the Kaggle CLI with literal arguments
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / 'kernels.json'
EXAMPLES_DIR = REPO_ROOT / 'examples'

# Notebooks reach their assets by cloning the repo, which only happens when this env var is visible.
KAGGLE_ENV_GUARD = 'KAGGLE_KERNEL_RUN_TYPE'

# Each notebook carries an 'Open in Kaggle' badge; this is how one is recognised.
KAGGLE_BADGE = 'kaggle.com/static/images/open-in-kaggle.svg'

# Stands in for the owner when staging without credentials, so a dry run still works offline.
PLACEHOLDER_OWNER = '<KAGGLE_USERNAME>'

# Kaggle refuses a push while five batch sessions are already running; this is how it says so.
SESSION_LIMIT = 'Maximum batch CPU session count'

# Kaggle's own rule for the identifier half of a kernel reference.
SLUG_PATTERN = re.compile(r'[a-z0-9][a-z0-9-]{3,58}[a-z0-9]')


@dataclass(frozen=True)
class RemoteKernel:
    """A kernel as Kaggle currently holds it."""

    title: str
    votes: int

    @property
    def vote_count(self) -> str:
        """Return the vote tally as English, e.g. ``"1 vote"`` or ``"5 votes"``."""
        return f'{self.votes} vote' if self.votes == 1 else f'{self.votes} votes'


def slugify(title: str) -> str:
    """Reduce a kernel title to the slug Kaggle would derive from it.

    This is where a kernel lands after a push, so it doubles as the slug a new kernel is created
    at and the slug a renamed one moves to.

    Args:
        title: Human-readable kernel title.

    Returns:
        Lowercase dash-separated slug.
    """
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-')


def resolve_owner(explicit: str | None) -> str | None:
    """Find the Kaggle account that owns the kernels.

    Args:
        explicit: Owner passed on the command line, or ``None``.

    Returns:
        Kaggle username, or ``None`` when no credentials are available.
    """
    if explicit:
        return explicit
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']

    config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', Path.home() / '.kaggle'))
    config = config_dir / 'kaggle.json'
    if config.is_file():
        return json.loads(config.read_text(encoding='utf-8')).get('username')
    return None


def load_kernels(manifest: Path) -> list[dict]:
    """Read the kernel manifest.

    Args:
        manifest: Path to ``kernels.json``.

    Returns:
        List of kernel entries.
    """
    return json.loads(manifest.read_text(encoding='utf-8'))['kernels']


def validate(kernels: list[dict], examples_dir: Path) -> list[str]:
    """Check the manifest against the notebooks on disk.

    Args:
        kernels: Kernel entries from the manifest.
        examples_dir: Directory holding the source notebooks.

    Returns:
        List of error messages; empty when the manifest is consistent.
    """
    errors: list[str] = []
    seen_slugs: dict[str, str] = {}

    for entry in kernels:
        notebook = examples_dir / entry['notebook']
        if not notebook.is_file():
            errors.append(f'{entry["notebook"]}: listed in the manifest but missing from {examples_dir.name}/')
        else:
            # The notebook's own Kaggle badge has to name the slug the push will land on, or every
            # published copy advertises an address that stopped resolving the moment it moved.
            landing = slugify(entry['title'])
            text = notebook.read_text(encoding='utf-8')
            if KAGGLE_BADGE in text and f'/{landing})' not in text:
                errors.append(f'{entry["notebook"]}: its Kaggle badge does not point at /{landing}')
        if entry['slug'] in seen_slugs:
            errors.append(f'{entry["slug"]}: slug reused by {seen_slugs[entry["slug"]]} and {entry["notebook"]}')
        seen_slugs[entry['slug']] = entry['notebook']
        if not SLUG_PATTERN.fullmatch(entry['slug']):
            errors.append(f'{entry["slug"]}: not a valid Kaggle slug (5-60 chars, lowercase letters, digits, dashes)')
        if len(entry['title']) < 5:
            errors.append(f'{entry["notebook"]}: title "{entry["title"]}" is under Kaggle\'s five-character minimum')
        # Kaggle re-slugs a kernel to match its title, so a kernel that does not exist yet will land
        # at slugify(title) whatever the manifest claims. On a live kernel the same mismatch is
        # meaningful instead of wrong: it is a pending rename, reported by the push rather than
        # rejected here, because renaming requires pushing the OLD slug with the NEW title.
        if not entry.get('live') and entry['slug'] != slugify(entry['title']):
            errors.append(
                f'{entry["notebook"]}: a new kernel titled "{entry["title"]}" lands at '
                f'"{slugify(entry["title"])}", not "{entry["slug"]}". Match them.'
            )

    mapped = {entry['notebook'] for entry in kernels}
    for notebook in sorted(examples_dir.glob('*.ipynb')):
        if notebook.name not in mapped:
            errors.append(f'{notebook.name}: no entry in {MANIFEST.name} — add one so it is not silently skipped')

    return errors


def check_kaggle_guard(notebook: Path) -> bool:
    """Report whether a notebook clones the repo when it runs on Kaggle.

    Args:
        notebook: Path to the source notebook.

    Returns:
        ``True`` when the Kaggle branch of the setup cell is present.
    """
    return KAGGLE_ENV_GUARD in notebook.read_text(encoding='utf-8')


def build_metadata(entry: dict, owner: str) -> dict:
    """Build the ``kernel-metadata.json`` payload for one notebook.

    Args:
        entry: Kernel entry from the manifest.
        owner: Kaggle username that owns the kernel.

    Returns:
        Metadata dict ready to be written next to the notebook.
    """
    return {
        'id': f'{owner}/{entry["slug"]}',
        'title': entry['title'],
        'code_file': entry['notebook'],
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': False,
        # Notebooks pip-install uniface and download ONNX weights on first use.
        'enable_internet': True,
        'enable_gpu': entry.get('enable_gpu', False),
        'enable_tpu': False,
        'dataset_sources': entry.get('dataset_sources', []),
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }


def stage(entry: dict, examples_dir: Path, staging_root: Path, owner: str) -> Path:
    """Copy a notebook and its metadata into a directory Kaggle can push.

    Args:
        entry: Kernel entry from the manifest.
        examples_dir: Directory holding the source notebooks.
        staging_root: Directory to create the per-kernel folder under.
        owner: Kaggle username that owns the kernel.

    Returns:
        The staged directory.
    """
    staged = staging_root / entry['slug']
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(examples_dir / entry['notebook'], staged / entry['notebook'])
    (staged / 'kernel-metadata.json').write_text(
        json.dumps(build_metadata(entry, owner), indent=2) + '\n', encoding='utf-8'
    )
    return staged


def push_kernel(staged: Path, retries: int, wait: int) -> tuple[int, str]:
    """Push one staged kernel, waiting out Kaggle's concurrent-session cap.

    Kaggle runs every pushed version and refuses a push once five batch sessions are already
    running, so a fifteen-notebook sync cannot be submitted in one burst. Each rejection is
    retried rather than reported, since it means "not yet", not "no".

    Args:
        staged: Directory holding the notebook and its ``kernel-metadata.json``.
        retries: How many times to wait for a session slot before giving up.
        wait: Seconds to wait between attempts.

    Returns:
        The final return code and the combined output of the last attempt.

    Raises:
        FileNotFoundError: If the Kaggle CLI is not on PATH.
    """
    for attempt in range(retries + 1):
        result = run_kaggle(['kernels', 'push', '-p', str(staged)])
        output = (result.stdout + result.stderr).strip()
        if SESSION_LIMIT not in output:
            return result.returncode, output
        if attempt < retries:
            print(f'  Kaggle is already running its five batch sessions; retrying in {wait}s')
            time.sleep(wait)
    return result.returncode, output


def run_kaggle(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the Kaggle CLI.

    Args:
        args: Arguments after the ``kaggle`` executable.

    Returns:
        The completed process, with stdout and stderr captured as text.

    Raises:
        FileNotFoundError: If the Kaggle CLI is not on PATH.
    """
    executable = shutil.which('kaggle')
    if executable is None:
        raise FileNotFoundError('kaggle')
    return subprocess.run([executable, *args], capture_output=True, text=True, check=False)  # nosec B603


def fetch_remote(owner: str) -> dict[str, RemoteKernel] | None:
    """Read the account's kernels, keyed by slug.

    ``kernels status`` cannot tell a missing kernel from a private one — both answer with the same
    permission error — so existence is settled from a listing instead. ``--mine`` covers private
    kernels the public ``--user`` view omits, and the two are merged.

    Args:
        owner: Kaggle username to list.

    Returns:
        Slug to :class:`RemoteKernel`, or ``None`` when the listing could not be read at all.
    """
    remote: dict[str, RemoteKernel] = {}
    reachable = False

    for scope in (['--mine'], ['--user', owner]):
        try:
            result = run_kaggle(['kernels', 'list', *scope, '--page-size', '100', '-v'])
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            continue
        reachable = True
        # Columns are ref,title,author,lastRunTime,totalVotes; only real rows carry an owner/slug ref.
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) < 5 or '/' not in row[0]:
                continue
            row_owner, _, slug = row[0].strip().partition('/')
            if row_owner != owner:
                continue
            remote[slug] = RemoteKernel(title=row[1].strip(), votes=int(row[4]) if row[4].isdigit() else 0)

    return remote if reachable else None


def resolve_target(entry: dict, remote: dict[str, RemoteKernel] | None) -> tuple[str | None, RemoteKernel | None]:
    """Work out which slug to push a manifest entry against.

    A kernel is looked for under its manifest slug first, then under the slug its title implies.
    The second lookup is what makes a rename idempotent: once Kaggle has moved a kernel to match a
    new title, the manifest's old slug stops resolving, and only the title-derived one finds it.

    Args:
        entry: Kernel entry from the manifest.
        remote: Kernels currently on the account, or ``None`` when the listing could not be read.

    Returns:
        The slug to push against and the kernel already there, if any. The slug is ``None`` when a
        kernel marked live cannot be found under either name, which means its slug was mistyped.
    """
    if remote is None:
        return entry['slug'], None
    if entry['slug'] in remote:
        return entry['slug'], remote[entry['slug']]
    renamed = slugify(entry['title'])
    if renamed in remote:
        return renamed, remote[renamed]
    return (None, None) if entry.get('live') else (entry['slug'], None)


def list_remote(owner: str, kernels: list[dict]) -> int:
    """Print the account's kernels and reconcile them against the manifest.

    Args:
        owner: Kaggle username to list.
        kernels: Kernel entries from the manifest.

    Returns:
        Process exit code.
    """
    remote = fetch_remote(owner)
    if remote is None:
        print(f'Could not list kernels for {owner}. Check the kaggle CLI and its credentials.', file=sys.stderr)
        return 1

    print(f'Kernels on {owner} ({len(remote)} found):')
    for slug, kernel in sorted(remote.items()):
        print(f'  {kernel.vote_count:>8}  {slug}  "{kernel.title}"')

    print('\nManifest:')
    for entry in kernels:
        kernel = remote.get(entry['slug'])
        state = f'live, {kernel.vote_count}' if kernel else 'new'
        print(f'  [{state}] {entry["slug"]}  <- {entry["notebook"]}')
        if kernel and kernel.title != entry['title']:
            print(f'      retitle "{kernel.title}" -> "{entry["title"]}"; the URL moves to {entry["slug"]}')
        if entry.get('live') and not kernel:
            print('      ORPHAN RISK: marked live in the manifest but absent from Kaggle.')

    unmapped = set(remote) - {entry['slug'] for entry in kernels}
    if unmapped:
        print('\nOn Kaggle but not in the manifest (pushing would not touch these):')
        for slug in sorted(unmapped):
            print(f'  {slug}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Publish examples/*.ipynb to Kaggle as kernels')
    parser.add_argument('--dry-run', action='store_true', help='Stage and validate, push nothing')
    parser.add_argument('--only', help='Only sync kernels whose notebook or slug contains this substring')
    parser.add_argument('--owner', help='Kaggle username (defaults to KAGGLE_USERNAME or ~/.kaggle/kaggle.json)')
    parser.add_argument('--list', action='store_true', help='List the account kernels and reconcile with the manifest')
    parser.add_argument('--stage-dir', type=Path, help='Keep the staged kernels here instead of a temp directory')
    parser.add_argument(
        '--session-wait', type=int, default=60, help='Seconds to wait for a free Kaggle session slot (default: 60)'
    )
    parser.add_argument(
        '--session-retries', type=int, default=30, help='How many times to wait for a slot (default: 30)'
    )
    args = parser.parse_args()

    kernels = load_kernels(MANIFEST)

    errors = validate(kernels, EXAMPLES_DIR)
    if errors:
        print('Manifest is out of sync with examples/:', file=sys.stderr)
        for error in errors:
            print(f'  {error}', file=sys.stderr)
        return 1

    owner = resolve_owner(args.owner)
    if owner is None:
        if args.list or not args.dry_run:
            print('No Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY, or pass --owner.', file=sys.stderr)
            return 1
        owner = PLACEHOLDER_OWNER
        print('No Kaggle credentials found; staging with a placeholder owner.\n')

    if args.list:
        if args.only:
            print('--only is ignored by --list, which always shows every kernel.\n')
        return list_remote(owner, kernels)

    if args.only:
        kernels = [e for e in kernels if args.only in e['notebook'] or args.only in e['slug']]
        if not kernels:
            print(f'No kernel matches --only {args.only!r}', file=sys.stderr)
            return 1

    for entry in kernels:
        if not check_kaggle_guard(EXAMPLES_DIR / entry['notebook']):
            print(f'Warning: {entry["notebook"]} has no {KAGGLE_ENV_GUARD} branch; its assets will be missing.')

    # One listing settles existence for every kernel, so the loop below makes no extra API calls.
    remote = fetch_remote(owner) if owner != PLACEHOLDER_OWNER else None
    if remote is None:
        print('Could not read the account listing; update-vs-create is unknown for every kernel.\n')

    staging_root = args.stage_dir
    temp_dir = None
    if staging_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix='kaggle-sync-')
        staging_root = Path(temp_dir.name)
    staging_root.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    renamed: list[tuple[str, str, str]] = []
    try:
        for entry in kernels:
            target, current = resolve_target(entry, remote)

            if target is None:
                ref = f'{owner}/{entry["slug"]}'
                message = f'{ref}: marked live but Kaggle has it under neither that slug nor the one its title implies'
                if args.dry_run:
                    print(f'[dry-run] {ref} (BLOCKED) <- {entry["notebook"]}')
                    print(f'  ORPHAN RISK: {message}.')
                    continue
                print(f'Refusing {message}.', file=sys.stderr)
                print('  Restore the original slug, or drop "live" if the kernel is genuinely new.', file=sys.stderr)
                failed.append(ref)
                continue

            ref = f'{owner}/{target}'
            moves_to = slugify(entry['title']) if current and current.title != entry['title'] else None
            # Push against the slug Kaggle knows; a differing title is what asks it to rename.
            staged = stage({**entry, 'slug': target}, EXAMPLES_DIR, staging_root, owner)

            if args.dry_run:
                if current:
                    state = f'update existing, {current.vote_count}'
                elif remote is not None:
                    state = 'CREATE NEW'
                else:
                    state = 'unknown'
                print(f'[dry-run] {ref} ({state}) <- {entry["notebook"]}')
                if moves_to:
                    print(f'  retitle "{current.title}" -> "{entry["title"]}"')
                    print(f'  URL MOVES {target} -> {moves_to}; the old address stops resolving.')
                continue

            print(f'Pushing {ref} <- {entry["notebook"]}')
            try:
                returncode, output = push_kernel(staged, args.session_retries, args.session_wait)
            except FileNotFoundError:
                print('The kaggle CLI is not installed. Run: pip install kaggle', file=sys.stderr)
                return 1

            if output:
                print('  ' + output.replace('\n', '\n  '))
            if returncode != 0 or 'error' in output.lower():
                failed.append(ref)
                continue

            landed = moves_to or target
            print(f'  https://www.kaggle.com/code/{owner}/{landed}')
            if moves_to:
                renamed.append((entry['notebook'], target, moves_to))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        elif args.dry_run:
            print(f'\nStaged in {staging_root}')

    if renamed:
        # The manifest now names slugs that no longer exist. resolve_target still finds these
        # kernels by title, but leaving the stale slugs in place makes every later run rely on
        # that fallback instead of saying plainly where each kernel lives.
        print(f'\n{len(renamed)} kernel(s) moved. Update {MANIFEST.name}:')
        for notebook, before, after in renamed:
            print(f'  {notebook}: "slug": "{before}" -> "{after}"')

    if failed:
        print(f'\n{len(failed)} kernel(s) failed to push:', file=sys.stderr)
        for ref in failed:
            print(f'  {ref}', file=sys.stderr)
        return 1

    if not args.dry_run:
        print(f'\nPushed {len(kernels)} kernel(s). Kaggle runs each new version on its own.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
