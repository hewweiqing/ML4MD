"""Runtime tests retained for the certified DelftBlue environment (not locally executable here)."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN
from jarvis.core.atoms import pmg_to_atoms
from pymatgen.core import Lattice, Structure
from alignn_stage2.coordinate_cache import graph_digest
from alignn_stage2.coordinate_noise import perturb_structure
from alignn_stage2.calibration_contract import TemperatureScalingContractError, apply_temperature, fit_temperature
from alignn_stage2.cpu_coordinate_training import (assert_native_logit_contract, component_state, model_config,
    states_byte_identical)
from alignn_stage2.cpu_runtime import assert_model_cpu
from alignn_stage2.cpu_split import GRAPH_SETTINGS

def test_cpu_model_and_head_only_synthetic_warmup():
    torch.manual_seed(3); model=ALIGNN(model_config()).to(torch.device("cpu")); assert_model_cpu(model)
    before=component_state(model,classifier=False); head_before=component_state(model,classifier=True)
    for name,parameter in model.named_parameters(): parameter.requires_grad_(name.startswith("fc."))
    descriptors=torch.randn(128,model.fc.in_features,device="cpu"); labels=torch.tensor([0,1]*64,device="cpu")
    optimizer=torch.optim.AdamW(model.fc.parameters(),lr=1e-4,weight_decay=0.0)
    optimizer.zero_grad(); torch.nn.functional.cross_entropy(model.fc(descriptors),labels).backward(); optimizer.step()
    assert states_byte_identical(before,component_state(model,classifier=False))
    assert not states_byte_identical(head_before,component_state(model,classifier=True))

def test_native_two_class_raw_logit_contract_including_batch_size_one():
    for count in (1,7):
        logits=torch.randn(count,2,device="cpu"); labels=torch.arange(count,device="cpu")%2
        assert_native_logit_contract(torch.log_softmax(logits,dim=1),logits,labels)

def test_noisy_atom_and_line_graphs_are_rebuilt_from_perturbed_coordinates():
    clean=Structure(Lattice.cubic(5.4),["Si","Si"],[[0,0,0],[0.25,0.25,0.25]])
    noisy,_=perturb_structure(clean,0.02,112)
    clean_atom,clean_line=Graph.atom_dgl_multigraph(atoms=pmg_to_atoms(clean),id="clean",**GRAPH_SETTINGS)
    noisy_atom,noisy_line=Graph.atom_dgl_multigraph(atoms=pmg_to_atoms(noisy),id="noisy",**GRAPH_SETTINGS)
    assert graph_digest(noisy_atom)!=graph_digest(clean_atom)
    assert graph_digest(noisy_line)!=graph_digest(clean_line)
    assert np.array_equal(clean.cart_coords,Structure(Lattice.cubic(5.4),["Si","Si"],[[0,0,0],[0.25,0.25,0.25]]).cart_coords)

def test_approved_muben_shared_scalar_temperature_contract():
    logits=np.asarray([[-2.0,1.0],[0.2,-0.4],[1.1,1.3],[-0.8,-0.1],[0.4,0.9]],dtype=np.float64)
    labels=np.asarray([1,0,1,1,0],dtype=np.int64)
    fitted=fit_temperature(logits,labels,split="validation"); repeated=fit_temperature(logits,labels,split="validation")
    scaled=apply_temperature(logits,fitted)
    assert scaled.shape==logits.shape and np.isfinite(fitted.temperature) and fitted.temperature>0
    assert fitted.temperature==repeated.temperature and fitted.objective and fitted.optimization_steps>0
    assert fitted.converged and fitted.validation_nll_after<=fitted.validation_nll_before+1e-12
    assert np.array_equal(np.argmax(logits,axis=1),np.argmax(scaled,axis=1))
    assert np.array_equal(np.argsort(logits[:,1]-logits[:,0],kind="stable"),np.argsort(scaled[:,1]-scaled[:,0],kind="stable"))
    assert np.array_equal(logits/1.0,logits)
    with __import__("pytest").raises(TemperatureScalingContractError): fit_temperature(logits,labels,split="outer_test")
