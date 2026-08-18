#!/usr/bin/env python3
import argparse
from pathlib import Path

from alignn_stage2.production import atomic_promote, verify_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--quarantine-root", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    relative = Path(f"fold_{args.fold}") / f"seed_{args.seed}"
    staging, final = Path(args.staging_root) / relative, Path(args.final_root) / relative
    if not verify_complete(staging, args.fold, args.seed):
        raise RuntimeError("staging cell is not hash-verified complete")
    final.parent.mkdir(parents=True, exist_ok=True)
    atomic_promote(staging, final, Path(args.quarantine_root))
    if not verify_complete(final, args.fold, args.seed):
        raise RuntimeError("post-promotion verification failed")
    print(f"atomically promoted: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
