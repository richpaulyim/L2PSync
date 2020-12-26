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
- `ffnn_15_30_node.py` - neural network model and training script for small
  15/30 node graphs
- `ffnn_300_600_node.py` - neural network model and training script for
  majority vote subgraph classifier on 300:600 node graphs

(random-forest)
- `gradientboost.py` - gradient boosting model and training script
- `random_forest.py` - random forest model and training script

### simulation-data

(main)
- `firefly.py` - FCA simulation code and additional functions for generating
  animations
- `greenberghastings.py` - Base Greenberg-Hastings simulation code 
- `kuramoto.py` - Base Kuramoto simulation code
- `generate_graph_dynamics_pair.py` - code for generating data files and CSVs
  for graphs.g6 file type and coloring simulation features (detailed
instructions below)

(LRCN-datagen)
- `OmegaDynamicsKURA.py` - Script for creating omega matrices for Kuramoto data
  (.npy)
- `OmegaFCA.py` - Script for creating omega matrices for FCA data (.npy)
- `OmegaGH.py` - Script for creating omega matrices for Greenberg-Hastings data
  (.npy)

(subgraphs-datagen)
- `large_graph_generator.py` - Script for generating graphs and corresponding
  initial dynamics for 300:600 node graphs for {FCA,GH,KM}
- `NNetworkcop.py` - Modified copy of NNetwork library (Github users: jvendrow and
  hanbaeklyu)
- `random_subgraph_generator.py` - Script for generating random
  induced-subgraphs-induced-dynamics data

## Instructions for Generating Graph-Initial-Dynamics Pairs

1. 
2.
3.
4.

### For further inquiries email:
Hardeep Bassi: hbassi21@gmail.com
Richard Yim: richyim555@g.ucla.edu
