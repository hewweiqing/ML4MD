#!/usr/bin/env python3
from alignn_stage2.coordinate_cache import verify_coordinate_cache
from alignn_stage2.sigma_authorization import require_authorized_sigma
import argparse
def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache-root",required=True); p.add_argument("--fold",type=int,required=True); p.add_argument("--seed",type=int,required=True); a=p.parse_args()
    result=verify_coordinate_cache(a.cache_root,expected_fold=a.fold,expected_seed=a.seed,authorization=require_authorized_sigma()); print(f"ALIGNN_COORDINATE_CACHE_VERIFY_CPU: {result['status']}")
if __name__ == "__main__": main()
