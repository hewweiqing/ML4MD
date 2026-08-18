#!/usr/bin/env python3
"""Dependency-light v29 behavioral/static tests for the packaging workstation."""
from __future__ import annotations
import ast, json, math, os, tempfile
from pathlib import Path
import numpy as np
from alignn_stage2.coordinate_noise import balanced_random_labels, perturb_structure
from alignn_stage2.cpu_coordinate_oof import clustered_structure_bootstrap
from alignn_stage2.cpu_prediction_schema import CONDITIONS, FIELDS, probability
from alignn_stage2.sigma_authorization import CoordinateSigmaAuthorizationMissing, require_authorized_sigma

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

class Lattice:
    def __init__(self,matrix=None,pbc=(True,False,True)): self.matrix=np.eye(3) if matrix is None else np.asarray(matrix); self.pbc=tuple(pbc)
    def get_fractional_coords(self,cart): return np.asarray(cart)@np.linalg.inv(self.matrix)
class Structure:
    def __init__(self,lattice,species,coords,coords_are_cartesian=False,site_properties=None,charge=None):
        self.lattice=lattice; self.species=list(species); self.site_properties=dict(site_properties or {}); self.charge=charge
        self._cart=np.asarray(coords,dtype=float) if coords_are_cartesian else np.asarray(coords,dtype=float)@self.lattice.matrix
    @property
    def cart_coords(self): return self._cart.copy()
    def __len__(self): return len(self.species)

def main():
    for path in ROOT.rglob("*.py"): ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
    check("python_ast",True)
    for path in ROOT.rglob("*.json"): json.loads(path.read_text(encoding="utf-8"))
    check("json_validation",True)
    config=json.loads((ROOT/"COORDINATE_NOISE_CONFIG.json").read_text())
    check("release_sigma_null",config["sigma_cartesian_per_axis_angstrom"] is None and not config["scientific_execution_authorized"])
    os.environ.pop("ALIGNN_COORDINATE_SIGMA_AUTHORIZATION",None)
    try: require_authorized_sigma(); raise AssertionError("missing sigma accepted")
    except CoordinateSigmaAuthorizationMissing: checks.append("missing_sigma_fail_closed")
    os.environ["ALIGNN_COORDINATE_TEST_MODE"]="1"
    test={"sigma_cartesian_per_axis_angstrom":0.02,"test_only":True,"scientific_use_prohibited":True}
    check("test_only_sigma_isolated",require_authorized_sigma(test_config=test,allow_test_only=True).test_only)
    try: require_authorized_sigma(test_config={**test,"test_only":False},allow_test_only=True); raise AssertionError("bad test sigma accepted")
    except CoordinateSigmaAuthorizationMissing: checks.append("test_only_markers_required")
    clean=Structure(Lattice(),["Si","O"],[[0.99,1.25,0.01],[0.02,-0.4,0.98]])
    before=clean.cart_coords.copy(); noisy1,r1=perturb_structure(clean,0.5,7); noisy2,r2=perturb_structure(clean,0.5,7)
    check("coordinate_determinism",np.array_equal(noisy1.cart_coords,noisy2.cart_coords) and np.array_equal(r1.displacement,r2.displacement))
    check("structure_immutability",np.array_equal(clean.cart_coords,before))
    check("species_cell_pbc_preserved",noisy1.species==clean.species and np.array_equal(noisy1.lattice.matrix,clean.lattice.matrix) and noisy1.lattice.pbc==clean.lattice.pbc)
    frac=noisy1.lattice.get_fractional_coords(noisy1.cart_coords)
    check("periodic_only_wrapping",np.all((frac[:,[0,2]]>=0)&(frac[:,[0,2]]<1)) and np.allclose(frac[:,1],before[:,1]+r1.displacement[:,1]))
    labels=balanced_random_labels(3000,91); check("exact_balanced_labels",np.bincount(labels,minlength=2).tolist()==[1500,1500])
    check("balanced_label_repeatability",np.array_equal(labels,balanced_random_labels(3000,91)))
    jobs=sorted((ROOT/"slurm").glob("*.sbatch")); check("ten_cpu_jobs",len(jobs)==10)
    for path in jobs:
        text=path.read_text().lower(); check(f"cpu_policy_{path.name}",all(x in text for x in
            ("--account=research-me-mse","--partition=compute","--ntasks=1","--cpus-per-task=8","--mem-per-cpu=3968m",'export cuda_visible_devices=""')))
        check(f"no_gpu_{path.name}",not any(x in text for x in ("--gres=gpu","--gpus","gpu-a100",".cuda()",'device="cuda"')))
    training=(ROOT/"alignn_stage2/cpu_coordinate_training.py").read_text()
    check("head_only_warmup",all(x in training for x in ('name.startswith("fc.")',"938","batch_size\": 128","encoder_byte_identical")))
    check("paired_order",'if control_order != random2_order' in training)
    check("resume_state",all(x in training for x in ("batch_position","optimizer","scheduler","rng_state","epoch_loss_sum","elapsed_runtime_seconds","os.replace")))
    check("partial_not_complete",'INCOMPLETE_RESUME_REQUIRED.json' in training and 'return history, False' in training)
    cache=(ROOT/"alignn_stage2/coordinate_cache.py").read_text()
    check("graph_rebuilding",'Graph.atom_dgl_multigraph' in cache and 'atom_graph_sha256' in cache and 'line_graph_sha256' in cache)
    check("separate_coordinate_cache",'COORDINATE_CACHE_MANIFEST.json' in cache and 'RECORD_COUNT = 3000' in cache)
    check("dgl_column_regression",'namespace[key]' in cache and 'graph_feature_tensors' in cache and '.ndata.values()' not in cache)
    check("prediction_schema",len(CONDITIONS)==4 and "raw_native_logit_0" in FIELDS and "execution_device" in FIELDS)
    check("raw_probability",math.isclose(probability(-0.4,0.7),1/(1+math.exp(-1.1)),abs_tol=1e-15))
    rng=np.random.default_rng(3); z=rng.normal(size=(1000,2))
    for temperature in (0.005,0.2,1.0,7.0):
        scaled=z/temperature; shifted=scaled-scaled.max(axis=1,keepdims=True); soft=np.exp(shifted); soft=soft[:,1]/soft.sum(1)
        margin=(z[:,1]-z[:,0])/temperature; sig=np.empty_like(margin); positive=margin>=0
        sig[positive]=1/(1+np.exp(-margin[positive])); exponential=np.exp(margin[~positive]); sig[~positive]=exponential/(1+exponential)
        check(f"two_logit_equivalence_T_{temperature}",np.allclose(soft,sig,rtol=0,atol=1e-12))
        check(f"positive_scaling_class_T_{temperature}",np.array_equal(np.argmax(z,1),np.argmax(scaled,1)))
    check("T_one_exact",np.array_equal(z/1.0,z))
    values=np.arange(5*101,dtype=float).reshape(5,101); a=clustered_structure_bootstrap(values,250,11); b=clustered_structure_bootstrap(values,250,11)
    check("bootstrap_pairing_repeatability",a==b and a["unit"]=="structure_cluster_with_all_five_seed_predictions")
    oof=(ROOT/"alignn_stage2/cpu_coordinate_oof.py").read_text(); check("smoke_profile_excluded",'"smoke_and_profile_artifacts_included": False' in oof)
    calibrate=(ROOT/"alignn_stage2/cpu_coordinate_calibrate_export.py").read_text()
    check("validation_only_temperature_fit",calibrate.find("fit_temperature")<calibrate.find("load_outer_test"))
    check("approved_muben_hash",__import__("hashlib").sha256((ROOT/"vendor/muben_temperature_scaling.py").read_bytes()).hexdigest()=="868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719")
    print(json.dumps({"status":"passed","check_count":len(checks),"checks":checks},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
