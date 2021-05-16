# GDL_Project
Git Repository for Final Project in GDL course

### Laplace Pooling layer
Implemented by Jose Ezequiel Castro Elizondo
The files included are:
- `LaPool/LaPool.py` (The main Pooling layer)
- `LaPool/LaPool_example.ipyn` (notebook showing how to use the layer)
- `LaPool/Laplacian_Pooling_doc` (report of the layer)

### Clique Pooling layer
Implemented by Maria Andromachi Kolyvaki.
The contributed files for the implementation of the Clique Pooling Layer are the following:
- `clique_pooling/CliquePoolingLayer.py` (the `clique` layer)
- `clique_pooling/GDL_Clique_Pooling.ipynb` (the python notebook containg all the testing)
- `clique_pooling/CliquePooling_report.pdf` (report for the implemented layer)

### Deep Graph Mapper and MPR Pooling layers
Implemnted by Simone Eandi. Both layers have been implemented as new branches in a fork of the Pytorch Geometric repository.
1. The Deep Graph Mapper layer is in the `dgm` branch of the [forked repo](https://github.com/seandi/pytorch_geometric). New/modified files are:
    - `torch_geometric/nn/models/deep_graph_mapper.py` (the `dgm` layer)
    - `torch_geometric/nn/models/__init__.py` (import `dgm`)
    - `setup.py` (new dependencies)
2. The MPR Pooling layer implementation is instead in the `mpr_pooling` branch of the [forked repo](https://github.com/seandi/pytorch_geometric).
   The following are the contributed files:
   - `torch_geometric/nn/dense/mpr_pool.py` (the `MPRPooling` layer)
   - `torch_geometric/nn/dense/__init__.py` (import `MPRPooling`)
   - `docs/source/modules/nn.rst` (documentation)

Example scripts for running the DGM and MPRPooling layers can be found in the `dgm_and_mpr_pooling` directory of this repo, together with the report for both layers.
