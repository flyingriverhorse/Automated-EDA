#!/usr/bin/env python3
"""Concatenate modular notebook JS files into the legacy notebook.js bundle."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "static" / "js" / "eda" / "notebook" / "modules"
BUNDLE_PATH = ROOT / "static" / "js" / "eda" / "notebook.js"
MODULE_ORDER = [
    "01-core.js",
    "02-analysis-grid.js",
    "03-analysis-execution.js",
    "04-column-insights.js",
    "09-init.js",
    "05-categorical-modal.js",
    "06-numeric-modal.js",
    "12-categorical-numeric-modal.js",
    "07-crosstab-modal.js",
    "10-time-series-modal.js",
    "15-geospatial-modal.js",
    "11-text-modal.js",
    "13-network-modal.js",
    "14-target-modal.js",
    "08-marketing.js",
]

HEADER = """/**\n * NOTE: This file is auto-generated.\n * Edit the files in static/js/eda/notebook/modules/ and run\n *   python tools/build_notebook_bundle.py\n * to regenerate the bundled notebook script.\n */\n\n"""


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_bundle(check_only: bool = False) -> int:
    missing = [name for name in MODULE_ORDER if not (MODULE_DIR / name).exists()]
    if missing:
        print("Missing module files:", ", ".join(missing), file=sys.stderr)
        return 1

    chunks = []
    for name in MODULE_ORDER:
        path = MODULE_DIR / name
        chunks.append(path.read_text(encoding="utf-8"))

    bundle_content = HEADER + "\n\n".join(chunks).rstrip() + "\n"

    if check_only:
        if not BUNDLE_PATH.exists():
            print("Bundle missing.")
            return 1
        existing = BUNDLE_PATH.read_text(encoding="utf-8")
        if compute_hash(existing) == compute_hash(bundle_content):
            print("Bundle is up to date.")
            return 0
        print("Bundle is out of date.")
        return 1

    BUNDLE_PATH.write_text(bundle_content, encoding="utf-8")
    print(f"Wrote bundle to {BUNDLE_PATH.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether the bundle is up to date without writing it.",
    )
    args = parser.parse_args()
    return build_bundle(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
