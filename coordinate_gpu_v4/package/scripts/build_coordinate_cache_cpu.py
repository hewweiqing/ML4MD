#!/usr/bin/env python3
from alignn_stage2.coordinate_cache import build_coordinate_cache
import argparse
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--cache-root",required=True); p.add_argument("--fold",type=int,required=True); p.add_argument("--seed",type=int,required=True)
    a=p.parse_args(); result=build_coordinate_cache(a.dataset,a.cache_root,a.fold,a.seed); print(f"ALIGNN_COORDINATE_CACHE_CPU: {result['status']}")
if __name__ == "__main__": main()
