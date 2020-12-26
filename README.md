# Learning to Predict Synchronization (L2PSync)

This is the repository for the paper "Learning to Predict Synchronization of
Coupled Oscillators on Heterogeneous Graphs" by Hardeep Bassi, Richard Yim, Joshua Vendrow, Rohith Koduluka, Cherlin Zhu and Hanbaek Lyu.

## Included Folders 

Note: Each folder contains a dedicated README markdown file detailing
instructions on generating data, using scripts and training/testing models.

### ml-models

(graph-lrcn)
- `GraphLRCN.py` - contains model implementation used for LRCN.
- `GraphLRCN_train.py` - contains training script applying `GraphLRCN` model
- `sweep.py` - contains example hyperparameter sweep script for `GraphLRCN`
  training and testing

(neural-network)
- `ffnn_15_30_node.py` - 
- `ffnn_300_600_node.py` - 

(random-forest)
- `gradientboost.py` -
- `random_forest.py` - 

### simulation-data

(main)
- `firefly.py` - 
- `greenberghastings.py` - 
- `kuramoto.py` - 
- `generate_graph_dynamics_pain.py` - 

(LRCN-datagen)
- `OmegaDynamicsKURA.py` -
- `OmegaFCA.py` - 
- `OmegaGH.py` - 

(subgraphs-datagen)
- `large_graph_generator.py` - 
- `NNetworkcop.py` - 
- `random_subgraph_generator.py` - 

For further inquiries email:

