#!/usr/bin/env python3
import sys
from alignn_stage2.training import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--condition", "random2"]))

