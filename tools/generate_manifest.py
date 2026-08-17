from __future__ import annotations

import argparse
from pathlib import Path

from overfitting_spaces.data import _cifar, make_split_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic CIFAR-10 split manifest.")
    parser.add_argument("--data-root", required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--download", action="store_true"); parser.add_argument("--split-seed", type=int, default=1729); args = parser.parse_args()
    dataset = _cifar(args.data_root, train=True, download=args.download)
    write_manifest(make_split_manifest(dataset.targets, args.split_seed), args.output)
    print(args.output)


if __name__ == "__main__": main()
