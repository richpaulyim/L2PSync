# Learning to Predict Synchronization (L2PSync)

This is the repository for the paper "Learning to Predict Synchronization of
Coupled Oscillators on Heterogeneous Graphs" by Hardeep Bassi, Richard Yim, Joshua Vendrow, Rohith Koduluka, Cherlin Zhu and Hanbaek Lyu.

![image](https://user-images.githubusercontent.com/59981298/103162146-887afb80-47a1-11eb-8230-8a72291306b7.png)

# Paper 

https://arxiv.org/abs/2012.14048

<p align="center">
<img width="400" src="https://github.com/richpaulyim/L2PSync/blob/master/simulation-data/spintreek8_everykappa_31.gif" alt="logo">
</p>
Figure of firefly cellular automata (FCA) on 31^3 lattice for 8 coloring iterating until synchronization.
# Included Folders 
## ml-models

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

## simulation-data

(main)
- `firefly.py` - FCA simulation code and additional functions for generating
  animations
- `greenberghastings.py` - Base Greenberg-Hastings simulation code 
- `kuramoto.py` - Base Kuramoto simulation code
- `generate_graph_dynamics_pair.py` - code for generating data files and CSVs
  for graphs.g6 file type and coloring simulation features (detailed
instructions below)

(LRCN-datagen)
- `DeltaKM.py` - Script for creating omega matrices for Graph LRCN on Kuramoto data
  (.npy)
- `DeltaFCA.py` - Script for creating omega matrices for Graph LRCN on FCA data (.npy)
- `DeltaGH.py` - Script for creating omega matrices for Graph LRCN on Greenberg-Hastings data
  (.npy)

(subgraphs-datagen)
- `large_graph_generator.py` - Script for generating graphs and corresponding
  initial dynamics for 300:600 node graphs for {FCA,GH,KM}
- `NNetworkcop.py` - Modified copy of NNetwork library (Github users: jvendrow and
  hanbaeklyu)
- `random_subgraph_generator.py` - Script for generating random
  induced-subgraphs-induced-dynamics data

## Instructions for Generating Graph and Initial-Dynamics Pairs

1. Lines 20-24: insert sync/nonsync limits. `AMOUNT_ITS` is the number of
   dynamics iterations to be recorded. `ACCESS_KEY` and `SECRET_KEY` are the AWS
keys used, and are optional for recording actual data to machine.
2. Lines 120-130: insert information related to node count, k for kappa coloring
   when using discrete models, as well as number of iterations as upper bound of
number iterations to be run. 
3. Line 154: set `overshoot` to desired number of graphs to test simulations on
4. Line 155 (loop): defined `pnorm` parameters and `edgesetN` for number of NWS
   functions calls (this can be modified at user discretion)
5. Line 178: define `colperg` as number of initial colorings to simulate per
   graph.
6. Line 181 and 186: define whether the model is continuous or not to generate
   initial states of graphs; then define `mod` corresponding simulation to use
depending on your model, {FCA,GH,KM}.
7. User must decide whether to write to AWS S3 bucket or write to local; the
   latter option requires user to generate her own script.

### For further inquiries email:
Hardeep Bassi (hbassi21@gmail.com) and Richard Yim (richyim555@g.ucla.edu)
