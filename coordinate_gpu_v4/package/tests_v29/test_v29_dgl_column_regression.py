from __future__ import annotations
import torch
from alignn_stage2.coordinate_cache import graph_feature_tensors, validate_graph

class Column:
    """Stand-in for the lazy DGL 1.1.1 object returned by Frame.values()."""

class Namespace:
    def __init__(self): self.data={"z":torch.ones(2),"a":torch.zeros(3)}
    def keys(self): return self.data.keys()
    def __getitem__(self,key): return self.data[key]
    def values(self): return [Column() for _ in self.data]

class Graph:
    def __init__(self): self.ndata=Namespace(); self.edata=Namespace()
    def num_nodes(self): return 2

def test_dgl_111_columns_are_materialized_by_indexed_access():
    graph=Graph()
    assert all(isinstance(value,Column) for value in graph.ndata.values())
    tensors=graph_feature_tensors(graph)
    assert len(tensors)==4 and all(isinstance(value,torch.Tensor) for value in tensors)
    validate_graph(graph,"synthetic")
