# GDL_Project
Git Repository for Final Project in GDL course

### Laplace Pooling layer
Implemented by Jose Ezequiel Castro Elizondo

### Clique Pooling layer
Implemented by Maria Andromachi Kolyvaki

### Deep Graph Mapper and MPR Pooling layers
Implemnted by Simone Eandi. Both layers have been implemented as new branches in a fork of the Pytorch Geometric repository.
1. The Deep Graph Mapper layer is in the `dgm` branch of https://github.com/seandi/pytorch_geometric. New/modified files are:
    - `torch_geometric/nn/models/deep_graph_mapper.py` (the `dgm` layer)
    - `torch_geometric/nn/models/__init__.py` (import `dgm`)
    - `setup.py` (new dependencies)
2. The MPR Pooling layer implementation is instead in the `mpr_ppoling` branch of https://github.com/seandi/pytorch_geometric.
   The following are the contributed files:
   - `torch_geometric/nn/dense/mpr_pool.py` (the `MPRPooling` layer)
   - `torch_geometric/nn/dense/__init__.py` (import `MPRPooling`)
   - `docs/source/modules/nn.rst` (documentation)

Example scripts for running the DGM and MPRPooling layers can be found in the `dgm_and_mpr_pooling` directory of this repo, together with the report for both layers.
