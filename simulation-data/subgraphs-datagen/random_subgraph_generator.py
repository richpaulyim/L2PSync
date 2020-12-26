# -*- coding: utf-8 -*-
import sys; sys.path.insert(1,'../')
from firefly import *
from numpy.random import randint, normal
import networkx as nx
import numpy as np
import csv
from tqdm import trange, tqdm
from scipy import stats, io
import collections
import math
from kuramoto import *
import pandas as pd
import time, random, io
import NNetworkcop as nnc
from io import BytesIO, StringIO
import boto3
kappa=5
featureFile = StringIO() 
ff = csv.writer(featureFile, delimiter=",")
client = boto3.client('s3', aws_access_key_id='',
        aws_secret_access_key='',
        region_name='us-west-1')
syncFile = StringIO()
sf = csv.writer(syncFile, delimiter=',')
TO_CLIENT = False


# ================================================================================
# 1: generate subgraphs
# ================================================================================
new = "fcafixednodes_varied_edges_"
path="/mnt/l/home/fcaFIXEDNODES_VARIEDDENSITY/"

print("READING IN G6 GRAPH TAGS")
taglistlist = pd.read_csv(path+new+"tag.csv",
                    header=None).to_numpy()

print("READING IN INITIAL COLORS")
initialcolors = []
with open(path+new+"color.csv") as f:
    for i in tqdm(f):
        i.replace('\n','')
        try:
            initialcolors.append(np.asarray([int(c) for c in i.split(',')]))
        except:
            continue
print(len(initialcolors))
        
print("READING IN ALL FEATURES")
totalcolors = []
with open(path+new+"features.csv") as f:s
    for i in tqdm(f):
        i.replace('\n','')
        try:
            totalcolors.append(np.asarray([float(c) for c in i.split(',')]))
        except:
            continue
print(len(totalcolors))

print("READING IN SYNC LABELS")
syncparent =  pd.read_csv(path+new+"sync.csv",
                   header=None).to_numpy()

print("TAG LIST-LIST PROCESS 1")
taglist = [i[0] for i in tqdm(taglistlist)]

print("TAG LIST PROCESS 2")
graphs = [parse_graph6(j) for j in tqdm(taglist)]
subgraph_set = []
subgraphs = 16

for coloring, g in tqdm(enumerate(graphs)):
    gnn = nnc.NNetwork()
    gnn.add_edges(g.edges)
    gnn.add_nodes(g.nodes)
    color = initialcolors[coloring]
    numnodes = g.number_of_nodes()
    for ksub in range(subgraphs):
        nnet = gnn.k_node_ind_subgraph(k=30)
        edgeset = nnet.edges
        edgeset = [(x[0],x[1]) for x in edgeset]
        G = nx.Graph()
        G.add_edges_from(edgeset)

        # remap edges
        nodes = sorted(list(set([x for i in edgeset for x in i])))
        dictionarylabel = {j:i for i,j in enumerate(nodes)}
        relabeled_edges = [(dictionarylabel[edge[0]], dictionarylabel[edge[1]]) 
                for edge in edgeset]
        colorlist =  color[np.asarray(G.nodes)]
        f = ColorNNetwork(colorlist, relabeled_edges)
        #sim = basic_FCA(f, 5, its=1200, verbose=False, no_edges=True) 
        sf.writerow([syncparent[coloring]])

        # ================================================================================
        # 2: generate featrues for ffnn
        # ================================================================================
        def empirical_coloring(coloring, kappa):
            #print('asdasd')
            total_hits = []
            dist = np.linspace(0,kappa,num=5)[1:]
            hits = [0] * 4
            for num in coloring:
                for k in range(len(dist)):
                    if( num <= dist[0]):
                        hits[0] += 1
                    elif(num <= dist[1]):
                        hits[1] += 1
                    elif( num <= dist[2]):
                        hits[2] += 1
                    else:
                        hits[3] += 1
            hits = np.asarray(hits)/(4*len(coloring))
            total_hits.append(hits)
            return np.asarray(total_hits).T 

        def get_num_edges(graph):
            return nx.number_of_edges(graph)

        def get_num_vertices(graph):
            return nx.number_of_nodes(graph)

        def amount_colors(coloring):
            return 30

        def get_degree(nodelist):
            return node.degree

        def get_max_degree(graph):
            #import pdb; pdb.set_trace()
            return max([graph.degree(n) for n in graph.nodes])

        def get_min_degree(graph):
            return min([graph.degree(n) for n in graph.nodes])

        def get_kappa():
            return kappa

        def width_compute(coloring, kappa):
            differences = [np.max(coloring) - np.min(coloring)]
            for j in trange(1, kappa+1):
                shifted = (np.array(coloring) + j) % kappa
                differences.append(np.max(shifted) - np.min(shifted))
                #print(np.min(differences))
            return np.min(differences)

        def get_width(colorlist, kappa):
            return width_compute(colorlist, kappa)

        def width_to_kappa(colorlist):
            widths = get_width(colorlist,get_kappa())
            widths = np.asarray(widths)
            kappa = np.asarray(get_kappa())
            return widths/kappa

        def diameter(graph):
            return nx.diameter(graph)

        quartile_colors = empirical_coloring(colorlist, kappa)
        graph = G

        master_feature_list = [quartile_colors[0][0], quartile_colors[1][0], quartile_colors[2][0], quartile_colors[3][0],
        get_num_edges(graph), get_num_vertices(graph), amount_colors(colorlist), get_max_degree(graph),
        get_min_degree(graph), diameter(graph)]
        master_feature_list = np.asarray(master_feature_list)
        master_feature_list = master_feature_list.T
        #import pdb; pdb.set_trace()
        induced_dynamics = [totalcolors[coloring][i*numnodes + np.asarray(nodes)] for i in range(24)]
        features = np.append(induced_dynamics, master_feature_list)
        ff.writerow(features)


# =============================================== #
# end script
# =============================================== #
if TO_CLIENT:
    syncFileBytes = syncFile.getvalue().encode('utf-8')
    client.put_object(Body=syncFileBytes, Bucket='fca-graphs',
            Key='SUBGRAPH_FCA300_N20K_onlyedgevaried/sync_fca16kvaried.csv')
    syncFile.close()
    featureFileBytes = featureFile.getvalue().encode('utf-8')
    client.put_object(Body=featureFileBytes, Bucket='fca-graphs',
            Key='SUBGRAPH_FCA300_N16K_onlyedgevaried/features_fca16kvaried.csv')
    featureFile.close()
else:
    with open(path+"fcaEDGEVARIED600_subgraphs_sync.csv",'w') as f:
        print(syncFile.getvalue(),file=f)
    with open(path+"fcaEDGEVARIED600_subgraphs_features.csv",'w') as f:
        print(featureFile.getvalue(),file=f)

# recorded 24
# simulation ran for 1200

