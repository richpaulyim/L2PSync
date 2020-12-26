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
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from io import BytesIO, StringIO
import boto3

SYNC_LIMIT = 5
NONSYNC_LIMIT = 5
AMOUNT_ITS = 24

client = boto3.client('s3', aws_access_key_id='', 
        aws_secret_access_key='', 
        region_name='us-west-1')

def data_to_graph6(data):
    """Convert 6-bit integer sequence to graph6 character sequence."""
    if len(data) > 0 and (min(data) < 0 or max(data) > 63):
        raise NetworkXError("graph6 data units must be within 0..63")
    return ''.join([chr(d+63) for d in data])


def n_to_data(n):
    """Convert an integer to one-, four- or eight-unit graph6 sequence."""
    if n < 0:
        raise NetworkXError("Numbers in graph6 format must be non-negative.")
    if n <= 62:
        return [n]
    if n <= 258047:
        return [63, (n>>12) & 0x3f, (n>>6) & 0x3f, n & 0x3f]
    if n <= 68719476735:
        return [63, 63,
            (n>>30) & 0x3f, (n>>24) & 0x3f, (n>>18) & 0x3f,
            (n>>12) & 0x3f, (n>>6) & 0x3f, n & 0x3f]


def writeG6_FCA(G, nodes = None, header=False):
    """Generate graph6 format string from a simple undirected graph.
    Parameters
    ----------
    G : Graph (undirected)
    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by G.nodes() is used.
    header: bool
       If True add '>>graph6<<' string to head of data
    Returns
    -------
    s : string
       String in graph6 format
    Raises
    ------
    NetworkXError
        If the graph is directed or has parallel edges
    Examples
    --------
    >>> G = nx.Graph([(0, 1)])
    >>> nx.generate_graph6(G)
    '>>graph6<<A_'
    See Also
    --------
    read_graph6, parse_graph6, write_graph6
    Notes
    -----
    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.
    References
    ----------
    Graph6 specification:
    http://cs.anu.edu.au/~bdm/data/formats.txt for details.
    """
    if nodes is not None:
        ns = list(nodes)
    else:
        ns = list(G)

    def bits():
        for (i,j) in [(i,j) for j in range(1,n) for i in range(j)]:
            yield G.has_edge(ns[i],ns[j])

    n = G.order()
    data = n_to_data(n)
    d = 0
    flush = False
    for i, b in zip(range(n * n), bits()):
        d |= b << (5 - (i % 6))
        flush = True
        if i % 6 == 5:
            data.append(d)
            d = 0
            flush = False
    if flush:
        data.append(d)

    string_data =  data_to_graph6(data)
    if header:
        string_data  =  '>>graph6<<' + string_data
    return string_data



# template graph

def datagen():
    name = 'fca_300node'
    check = nx.empty_graph(300)
    # hard-coded parameters
    kappa = 5
    n = 300
    its = 1000
    inds = []
    # counters
    nsl = 0
    sl = 0
       # file writing
    syncFile = StringIO()
    sf = csv.writer(syncFile, delimiter=',')
    #colorFile = open(name + "/" + name + '_color.csv','w')
    colorFile = StringIO()
    cf = csv.writer(colorFile, delimiter=",")
    # csv of tags
    #tagFile = open(name + "/" + name + '_tag.csv','w')
    tagFile = StringIO()
    tf = csv.writer(tagFile, delimiter=",")
    #indFile = open(name + "/" + name + '_indices.csv','w')
    indFile = StringIO()
    indf = csv.writer(indFile, delimiter=",")
    #tags = open(g6, 'r')
    itsFile = StringIO()
    itsf = csv.writer(itsFile, delimiter=",")

    featureFile = StringIO()
    ff = csv.writer(featureFile, delimiter=",")

    # on-the-fly running and simulation
    for i,j in enumerate(trange(1000000)):
        print("=======================")
        print("nonsync and sync counts")
        print(nsl,sl)
        print("=======================")

        # probability for graph density
        # puniform = random.random() * 0.3 + .32
        # pnorm = normal(puniform, 0.04)
        pnorm = 0.6
        # creating graph
        KAPPA = kappa
        edgesetN = 5
        edgeset = []
        for i in range(edgesetN):
            edgeset = edgeset + edgeset_generator([n, 2, pnorm],type="nwg",show=0, verbose=False)
        #import pdb; pdb.set_trace()
        G = nx.Graph()
        G.add_edges_from(edgeset)
        print('Amount of edges:', len(set([frozenset(o) for o in edgeset])))
        print('Max Degree: ', max([G.degree(n) for n in G.nodes]))

        # checking isomorphic
        if nx.is_isomorphic(G, check): continue
        check = G
   
 
        for i in trange(10):
            # creating initial colored graph
            colorlist = np.random.randint(0,kappa,size=n) 
            #colorlist = np.random.uniform(-np.pi, np.pi, len(G.nodes))
            #print(colorlist)
            f = ColorNNetwork(colorlist, G.edges)

            # run simulation
            sim = basic_FCA(f, KAPPA, its=its, verbose=False, no_edges=True) 
            #sim = simulate_Kuramoto(f, 2, intrinsic=False, widthcheck=True, verbose=False)
            fake_color = 1
            if len(sim[3]) < AMOUNT_ITS:
                while len(sim[3]) < AMOUNT_ITS:
                    sim[3].append([fake_color] * len(sim[3][0]))
                    fake_color += 1
                    fake_color %= kappa
            q = sim[2]
            print(q)
            if q:
                sl += 1
            else:
                nsl += 1
            if nsl > NONSYNC_LIMIT and sl > SYNC_LIMIT:
                #import pdb; pdb.set_trace()
                syncFileBytes = syncFile.getvalue().encode('utf-8')
                colorFileBytes = colorFile.getvalue().encode('utf-8')
                tagFileBytes = tagFile.getvalue().encode('utf-8')
                indFileBytes = indFile.getvalue().encode('utf-8')
                itsFileBytes = itsFile.getvalue().encode('utf-8')
                client.put_object(Body=syncFileBytes, Bucket='fca-graphs', Key=name+ '/sync.csv')
                client.put_object(Body=colorFileBytes, Bucket='fca-graphs', Key=name+ '/color.csv')    
                client.put_object(Body=tagFileBytes, Bucket='fca-graphs', Key=name+ '/tag.csv')
                client.put_object(Body=indFileBytes, Bucket='fca-graphs', Key=name+ '/ind.csv')
                client.put_object(Body=itsFileBytes, Bucket='fca-graphs', Key=name+ '/its.csv')
                syncFile.close()
                colorFile.close()
                tagFile.close()
                indFile.close()
                itsFile.close()
                with BytesIO() as npyf:
                    np.save(npyf, features)
                    client.put_object(Body=npyf.getvalue(), Bucket='fca-graphs', Key=name+ '/features.npy')
                featureFileBytes = featureFile.getvalue().encode('utf-8')
                client.put_object(Body=featureFileBytes, Bucket='fca-graphs', Key=name+ '/features.csv')
                featureFile.close()
                print('==================================')
                print('     MISSION ACCOMPLISHED')
                print('==================================')
                return
            if nsl > NONSYNC_LIMIT and not q:
                continue
            if sl > SYNC_LIMIT and q:
                continue

            inds.append(i)
            cf.writerow(colorlist)
            indf.writerow([i])
            sf.writerow([sim[2]])
            tf.writerow([writeG6_FCA(G, list(range(len(colorlist))))])
            itsf.writerow([sim[4]])


            


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
                return 300

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

            features = np.append(sim[3][:24],master_feature_list)
            ff.writerow(features)

    print('Mission Failed: Not enough graphs to produce split')
    return -1
datagen()
