#!/usr/bin/env python3
"""Dependency-light synthetic CPU execution contract checks."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np
from alignn_stage2.coordinate_noise import balanced_random_labels, derive_stream_seed
from alignn_stage2.cpu_coordinate_oof import clustered_structure_bootstrap
from alignn_stage2.sigma_authorization import CoordinateSigmaAuthorizationMissing, require_authorized_sigma

def main():
    os.environ.pop("ALIGNN_COORDINATE_SIGMA_AUTHORIZATION",None)
    try: require_authorized_sigma(); raise AssertionError("missing sigma did not fail closed")
    except CoordinateSigmaAuthorizationMissing: pass
    os.environ["ALIGNN_COORDINATE_TEST_MODE"]="1"
    test={"sigma_cartesian_per_axis_angstrom":0.02,"test_only":True,"scientific_use_prohibited":True}
    auth=require_authorized_sigma(test_config=test,allow_test_only=True)
    assert auth.test_only and auth.sigma==0.02
    labels=balanced_random_labels(3000,derive_stream_seed(28,0,0,0,"labels")); assert len(labels)==3000 and sum(labels)==1500
    values=np.arange(5*31,dtype=float).reshape(5,31); first=clustered_structure_bootstrap(values,100,17); second=clustered_structure_bootstrap(values,100,17); assert first==second
    print(json.dumps({"status":"passed","missing_sigma_fail_closed":True,"test_only_isolated":True,
        "balanced_labels":True,"clustered_bootstrap_deterministic":True},indent=2))
if __name__=="__main__": main()
