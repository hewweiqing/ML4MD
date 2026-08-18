#!/usr/bin/env python3
"""Thin wrapper -> alignn_stage2.init_diagnostic. Must be run where
alignn/dgl/CUDA are installed (DelftBlue) — see STAGE3_STAGE_A_PROTOCOL.md.

Dumps ALIGNN_MODULE_INVENTORY.json (full named_modules() name/type list) on
its first invocation, before any measurement, so the activation-RMS trace
can later be joined against real module names without needing them
hardcoded in advance (see plan review item: ALIGNN's internal names are not
verified in the environment this package was written in).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alignn_stage2.common import write_json
from alignn_stage2.init_diagnostic import build_model, main as run_stage_a_main

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = PACKAGE_ROOT / "ALIGNN_MODULE_INVENTORY.json"


def dump_module_inventory() -> None:
    if INVENTORY_PATH.is_file():
        return
    model = build_model()
    inventory = [{"call_order_index": index, "name": name, "type": type(module).__name__,
        "is_leaf": not list(module.children())}
        for index, (name, module) in enumerate(model.named_modules())]
    write_json(INVENTORY_PATH, {"status": "generated_on_first_stage_a_run", "module_count": len(inventory),
        "modules": inventory})
    print(f"wrote {INVENTORY_PATH}", file=sys.stderr)


if __name__ == "__main__":
    dump_module_inventory()
    raise SystemExit(run_stage_a_main())
