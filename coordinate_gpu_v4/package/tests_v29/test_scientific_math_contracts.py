from __future__ import annotations
import numpy as np

def test_two_logit_softmax_sigmoid_equivalence():
    rng=np.random.default_rng(17); z=rng.normal(size=(1000,2)); tolerance=1e-12
    for temperature in (0.005,0.2,1.0,3.7,50.0):
        scaled=z/temperature; shifted=scaled-scaled.max(axis=1,keepdims=True)
        softmax=np.exp(shifted); softmax=softmax[:,1]/softmax.sum(axis=1)
        margin=(z[:,1]-z[:,0])/temperature
        sigmoid=np.empty_like(margin); positive=margin>=0
        sigmoid[positive]=1/(1+np.exp(-margin[positive])); exponential=np.exp(margin[~positive]); sigmoid[~positive]=exponential/(1+exponential)
        assert np.allclose(softmax,sigmoid,rtol=0,atol=tolerance)

def test_positive_temperature_preserves_labels_and_ordering():
    rng=np.random.default_rng(4); z=rng.normal(size=(257,2)); margin=z[:,1]-z[:,0]
    for t in (0.01,0.4,1.0,7.0):
        scaled=z/t
        assert np.array_equal(np.argmax(z,axis=1),np.argmax(scaled,axis=1))
        assert np.array_equal(np.argsort(margin,kind="stable"),np.argsort(scaled[:,1]-scaled[:,0],kind="stable"))
    assert np.array_equal(z/1.0,z)

def test_balanced_labels_and_determinism():
    from alignn_stage2.coordinate_noise import balanced_random_labels
    first=balanced_random_labels(3000,123); second=balanced_random_labels(3000,123)
    assert np.array_equal(first,second); assert np.bincount(first,minlength=2).tolist()==[1500,1500]

def test_structure_cluster_bootstrap_keeps_five_seeds_paired():
    from alignn_stage2.cpu_coordinate_oof import clustered_structure_bootstrap
    values=np.arange(5*101,dtype=float).reshape(5,101)
    first=clustered_structure_bootstrap(values,250,9); second=clustered_structure_bootstrap(values,250,9)
    assert first==second and first["repetitions"]==250
