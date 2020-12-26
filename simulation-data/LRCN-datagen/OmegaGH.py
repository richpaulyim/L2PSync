import numpy as np
import pandas as pd
import networkx as nx
import scipy as cp

import sys; sys.path.insert(1,'../')
#import sys; sys.path.insert(1, 'C:/Users/hbass/Desktop/fca/FCA-ML/')
from firefly import * 
from scipy.sparse import csr_matrix
from math import floor
from scipy.sparse.csgraph import reverse_cuthill_mckee

#path = 'C:/Users/hbass/Desktop/fca/FCA-ML/adjacency-dynamics/'
path="/mnt/l/home/kura30/"
path="/mnt/l/home/fca30/"
path="/mnt/l/home/gh30/"

# read initial coloring, labels, indices and graph6
coloring = pd.read_csv(path+'color.csv', 
        header=None).to_numpy()
# dataout = [i for i in np.load(path+'labels (4).npy',
#     allow_pickle=True)] 
dataout = [i for i in pd.read_csv(path + 'sync.csv', header=None).to_numpy()]
indices = [i for i in pd.read_csv(path+'ind.csv',
    header=None).to_numpy()]
graphs = nx.read_graph6(path+'tag.csv')
count = True

def width(colors, kappa):
    """
    computes width from a color list
    """
    ordered = list(set(colors)) 
    lordered = len(ordered)
    threshold = floor(kappa/2)
    if ordered == 0:
        assert("Empty array or logic error.")

    elif lordered == 1:
        return 0

    elif lordered == 2:
        dw = ordered[1]-ordered[0]
        if dw > threshold:
            return kappa - dw
        else:
            return dw

    else:
        widths = [ordered[-1]-ordered[0]]
        for i in range(lordered-1):
            widths.append(ordered[i+1]-ordered[i])
        return kappa - max(widths)

def omega_buildmatrices():
    cols = []; kappa = 5; its = 24; ind = []
    # run simulations
    print("===================")
    print("RUNNING SIMULATIONS")
    print("===================")
    if count:
        nsl = 0
        sl = 0
    for j in tqdm(indices):
        edgelist = list(graphs[int(j)].edges)
        colorlist = coloring[j][0][1:]
        net = ColorNNetwork(colorlist.tolist(), edgelist)
        coldyn = simulate_dynamics(net, kappa, its=its, timesec=60, verbose=0)[3]
        s = dataout[int(j)]
        if count:
            if s:
                _ = [print(m) for m in coldyn]
                print("----------")
                if sl>=100:
                    pass
                else:
                    cols.append(coldyn)
                    ind.append(int(j))
                    sl+=1
            else:
                if nsl>=100:
                    pass
                else:
                    cols.append(coldyn)
                    ind.append(int(j))
                    nsl+=1
            if sl >= 100 and nsl >= 100:
                break
    print(len(cols),len(ind))

    adjmatsnsl = []
    adjmatssl = []
    n = 30
    dataynsl = []
    dataysl = []
    # create adjacency matrices
    print("==================")
    print("ADJACENCY MATRICES") 
    print("==================") 
    if count:
        nsl = 0
        sl = 0
    for i, j in enumerate(tqdm(ind)): 

        # index from the graphs
        graph = graphs[int(j)] 

        # compute rcm
        rcm = np.asarray(reverse_cuthill_mckee(csr_matrix(\
                    nx.adjacency_matrix(graph).todense())
                    )
            )

        adjdyn = [] 
        # assigning colors
        for col in cols[i]: 
            for x in range(n): 
                for y in range(x+1): 
                    if graph.has_edge(x, y): 
                            widthcol = width([col[x],col[y]], kappa)
                            graph.add_weighted_edges_from([(x, y, widthcol)]) 
            frame = nx.adjacency_matrix(graph).todense()[:,rcm][rcm,:] + \
                                np.diag(np.asarray(col)[rcm])
            adjdyn.append(frame) 

        # pad iterations to uniform length 
        frameseq = np.stack(np.asarray(
            adjdyn + [adjdyn[-1]]*((its+1)-len(cols[i]))
            ),axis=0)
        if count:
            s = dataout[int(j)]
            if s:
                sl+=1
            else:
                nsl+=1

            print("SYNC:", len(dataysl),"NONSYNC:",len(dataynsl))
            if sl > 100:
                pass 
            else:
                adjmatssl.append(frameseq)
                dataysl.append(s)
            if nsl > 100:
                pass 
            else:
                adjmatsnsl.append(frameseq)
                dataynsl.append(s)

            if sl > 100 and nsl > 100:
                break

    print(len(adjmatssl), len(dataynsl))
    print(len(adjmatsnsl), len(dataysl))

    #datain = np.stack(adjmats, axis=0)

    # save results
    with open(path+'Omega.npy', 'wb') as f:
        np.save(f, adjmatssl)
        np.save(f, dataysl)
        np.save(f, adjmatsnsl)
        np.save(f, dataynsl)


omega_buildmatrices()

#with open('LRCN-Data/Omega.npy','rb') as f:
#    datain = np.load(f, allow_pickle=True)
#    dataout = np.load(f) 
#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)
##print(datain[0])
#
#gifmake(datain[0]/4, "omega0a", kappa=False, duration=150)
#gifmake(datain[1]/4, "omega0b", kappa=False, duration=150)
#
#gifmake(datain[-1]/4, "omega1c", kappa=False, duration=150)
#gifmake(datain[-2]/4, "omega1d", kappa=False, duration=150)
#print(datain[-1])
#
