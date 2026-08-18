#!/usr/bin/env python3
import argparse
from pathlib import Path

from alignn_stage2.production import verify_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cell = Path(args.final_root) / f"fold_{args.fold}" / f"seed_{args.seed}"
    if verify_complete(cell, args.fold, args.seed):
        print(f"verified complete cell: {cell}")
        return 0
    if (cell / "COMPLETE.json").exists() or cell.exists():
        print(f"invalid/partial final cell must be quarantined: {cell}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
