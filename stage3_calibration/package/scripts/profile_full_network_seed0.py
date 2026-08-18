#!/usr/bin/env python3
"""Times ONE seed (fold 0, seed 0) of the full-network warm-up arm before
committing to the full 20-seed Stage A run. 938 whole-network optimizer
steps run twice (deterministic-replay contract) is a much larger
intervention than the existing head-only arms — its cost is measured here,
not assumed, per review feedback on the plan.

Writes preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json. A human reviewer
must then write a matching approval JSON (see stage_a_profile.py) before
require_stage_a_gates.py will unlock the full run.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from alignn_stage2.common import write_json
from alignn_stage2.init_diagnostic import time_full_network_warmup_single_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = time_full_network_warmup_single_seed(Path(args.dataset), args.fold, args.seed)
    write_json(Path("preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json"), report)
    print(f"STAGE_A_FULL_NETWORK_SEED0_PROFILE: PASS "
        f"elapsed={report['elapsed_seconds']:.1f}s "
        f"projected_20_seeds={report['projected_seconds_for_20_seeds'] / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
