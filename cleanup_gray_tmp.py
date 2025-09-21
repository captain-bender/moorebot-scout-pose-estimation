#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def find_gray_tmp_dirs(root: Path):
    # Glob pattern to find any directories ending with _gray_tmp
    for p in root.rglob("*_gray_tmp"):
        if p.is_dir():
            yield p


def main():
    parser = argparse.ArgumentParser(description="Remove temporary grayscale folders (*_gray_tmp)")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="List matches but do not delete")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not prompt for confirmation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print extra details")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matches = list(find_gray_tmp_dirs(root))

    if not matches:
        print("No *_gray_tmp directories found under:", root)
        return 0

    print("Found the following *_gray_tmp directories:")
    for m in matches:
        print(" -", m)

    if args.dry_run:
        print("Dry-run: nothing deleted.")
        return 0

    if not args.yes:
        try:
            resp = input("Delete all of the above? [y/N]: ").strip().lower()
        except EOFError:
            resp = "n"
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    for m in matches:
        if args.verbose:
            print("Removing:", m)
        try:
            shutil.rmtree(m)
        except Exception as e:
            print(f"Warning: failed to remove {m}: {e}")

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
