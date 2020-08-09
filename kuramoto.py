# -*- coding: utf-8 -*-
from NNetwork import NNetwork
""" Fun Colors"""

fun_colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 
'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 
'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 
'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 
'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 
'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 
'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 
'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 
'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 
'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 
'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 
'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 
'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 
'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

"""# Color Network (NNetwork subclass)
# Notes
Stochastics and width lemma demonstration on iterations
"""

# import libraries
import numpy as np
import imageio
import networkx as nx
from NNetwork import NNetwork
from math import floor, pow
from scipy.stats import bernoulli
import time, random, io
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from PIL import Image
from tqdm import tqdm,trange

def time_format(seconds):
    minutes = 0 
    if seconds / 60 > 1:
        return(str(floor(seconds/60)) + " min. " + \
               str(round(seconds % 60,3)) + " sec.")
    return("0 min. " + str(round(seconds,3)) + " sec.")

# --------------------------- #
# Color list NNetwork extension
# --------------------------- #
class ColorNNetwork(NNetwork.NNetwork):
    
    """
    Specific case of NNetwork, 
    ***assumes 1-n {0,n-1} vertices uniquely labeled 
    each with its own coloring***
    """

    def __init__(self, colors=[],edges=[]):
        
        super().__init__()
        self.colors = colors
        self.add_edges(edges)
        self.mst_edges = []

    def get_color(self, v):
        """
        gets color of vertex
        """
        return self.colors[v]
    
    def get_colors(self):
        """
        returns colors of corresponding vertices
        i.e. we get each color
        """
        return self.colors
    
    def get_neighbor_colors(self, v):
        """
        gets set of colors of neighbors
        """
        neighbs = self.neighbors(v)
        
        return list(map(self.get_color,list(neighbs)))
    
    def set_color(self, v, col):
        """
        sets color of vertex
        """
        self.colors[v-1] = col
    
    def set_colors(self, color_list):
        """
        Assigns new colors corresponding to each vertex by index
        """
        self.colors = color_list
        

# --------------------------- #
# Kuramoto Implementation 
# --------------------------- #
def simulate_Kuramoto(G, K, T=10000, step=0.02, verbose=True,  
            timesec=30, gifduration=200):
    """
    Function that returns edgelist color changes for FCA
    Useful for visualization
    """
    # initial conditions
    F = G
    if T==0:
        T=its
        its=False
    
    # update rule for given vertex
    def euler_update(vertex):
        vertexcolor = F.get_color(vertex)
        omega = np.random.normal()
        fprime = omega + K * np.sum(
                np.sin(
                    np.asarray(F.get_neighbor_colors(vertex))-vertexcolor
                    )
                )
        new = F.get_color(vertex) + step * fprime
        if np.abs(new) > np.pi:
            print("slip")
            return np.mod(new, np.pi)
        return new

    its = int(T/step)
    current_colors = [F.get_colors()]
    for i in trange(its, disable=not verbose):
        new_col = np.fromiter(map(euler_update, range(0,F.num_nodes())), float)
        F.set_colors(new_col)
        current_colors.append(new_col)
    return np.stack(current_colors,axis=0)


# ----------------------------#
#   Color Scheme Generator 
# ----------------------------#
def color_scheme(kappa):
    random_spectrum = [("%06x" % random.randrange(10**80))[:6]
                     for i in range(0,kappa)]
    cs = {key:('#'+str(col)) for key, col in enumerate(random_spectrum)}
    return cs

def grapher(colors, edges, cs, animate):
    verts = list(range(0, len(colors)))
    G = nx.Graph()
    G.add_nodes_from(verts)

    if animate == 1:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=12)
    cmap = cs
    for node in verts:
        nx.draw_networkx_nodes(G, pos, [verts[node]], 
                        node_color=cs[colors[node]], node_size=400, alpha=0.8)
    G.add_edges_from(edges)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    labels = {}
    for i in range(len(colors)):
        labels[i] = colors[i]
    nx.draw_networkx_labels(G, pos, labels)
    plt.axis('off')
    
def edgeset_generator(dims, type='lattice', verbose=True, show=True,
        linegraphtoo=False):
    #lattice grapher
    mapper = {}
    if type == 'hex':
        edgeset = nx.generators.lattice.hexagonal_lattice_graph(dims[0], dims[1])
        if verbose: print("hex generated")
    elif type == 'tree':
        edgeset = nx.generators.classic.balanced_tree(dims[0], dims[1]) 
        if verbose:print("tree generated")
    elif type == 'star':
        if verbose: print("star generated")
        x = star(dims[0], dims[1])
        numvert = max([i for j in x for i in j])+1
        if verbose: print('The amount of vertices in the graph is: ', numvert)
        return x
    elif type == 'complete':
        edgeset = nx.generators.classic.complete_graph(dims[0])
    elif type == 'er': 
        edgeset = nx.generators.random_graphs.erdos_renyi_graph(dims[0], dims[1])
        if verbose:print("Erdos-Reyni generated")
    elif type == 'cycle':
        edgeset = nx.cycle_graph(dims[0])
        if verbose: print('cycle generated')
    elif type == 'nwg': 
        edgeset = nx.newman_watts_strogatz_graph(dims[0], dims[1], dims[2])
        if verbose: print('Connected Watts Strogatz graph generated.')
    elif type == 'torus':
        edgeset = nx.grid_2d_graph(dims[0], dims[1], periodic=True)
        if verbose: print("Torus generated")
    else:
        edgeset = nx.generators.lattice.grid_graph(dim=dims)
        if verbose: print("lattice generated")

    mapper = {key:i for i, key in enumerate(edgeset.nodes)}
    if show:
        nx.draw_networkx_labels(edgeset, nx.spring_layout(edgeset, seed=1024), mapper, font_size=12)
        nx.draw_networkx(edgeset)
        plt.show()

    vertexlist = list(range(0, len(list(edgeset.nodes))))
    if verbose: print('The amount of vertices in the graph is: ', len(vertexlist))

    pre_edge = [e for e in edgeset.edges]
    if linegraphtoo:
        return ([[mapper[pre_edge[j][0]], mapper[pre_edge[j][1]]] \
                    for j in range(0, len(pre_edge))], \
                    nx.line_graph(edgeset))
    return [[mapper[pre_edge[j][0]], mapper[pre_edge[j][1]]] \
                    for j in range(0, len(pre_edge))]


# ------------------------------------- #
# Other plotting metrics
# ------------------------------------- #
def gifmake(frames, name, pickcol=False,kappa=False, duration=125):
    """
    Input: sequence of matrices, name of gif
        Optional: color(pickcol), every kappa(kappa)
    output: gif dynamic with name as gif name
    """
    if not pickcol:
        pickcol = random.sample(fun_colors,1)[0]
        print(pickcol)

    def make_gif(frame): 
        plt.pcolormesh(np.array(frame),
                norm=colors.Normalize(vmin=0,vmax=1), cmap=pickcol)
        plt.axis('square')
        plt.axis('off') 
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.clf()
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    images = []
    for i, adj in enumerate(frames):
        if kappa:
            if i%kappa!=0:
                continue 
        make_gif(adj)

    tail = images[1:]
    images[0].save(name+'.gif', save_all=True, 
            append_images=tail, optimize=False, duration=duration, loop=0)


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

def lattice2D_mkgif(colors_it, n, name="Lattice2D",
        freeze=False, cap=100000, duration=100,pickcol=0):
    v = n
    images = []
    if pickcol:
        col=pickcol
    else:
        col = random.sample(fun_colors,1)[0]
    vi = np.min(colors_it)
    va = np.max(colors_it)

    def make_gif(frame): 
        plt.pcolormesh(frame,vmin=vi,vmax=va, cmap=col)
        plt.axis('square')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.clf()
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    # Starting generating frames
    for i, colors in enumerate(tqdm(colors_it)):
        #if i > cap:
        #    break
        a = np.reshape(np.asarray(colors_it), (np.shape(colors_it)[0], n, n))
        # Add frame
        make_gif(a[i])

    # append final frame
    make_gif(a[-1])
    plt.close()
    tail = images[1:]; 
    # optionally freeze last frame
    if freeze:
        tail.extend([images[-1] for i in range(0,5)])
    images[0].save(name+'.gif', save_all=True, 
                    append_images=tail, optimize=False, duration=duration, loop=0)

