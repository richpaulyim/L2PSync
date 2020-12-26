import numpy as np
import csv
import ast
import pickle

"""
Neighborhood Network package
Simple and weighted network objects with built-in neighborhood list for scalable random walk and MCMC sampling
Author: Josh Vendrow and Hanbaek Lyu
"""


class NNetwork():

    def __init__(self):
        self.edges = []
        # self.sorted_left = None
        # self.sorted_right = None
        self.neighb = {}
        self.vertices = []
        self.vertices_set = set()
        self.number_nodes = 0

    """
    def set_edges(self, edges):

        self.sort_left(edges)
        self.sort_right(edges)
        self.edges = edges
        self.set_neighbors()
    """

    def add_edges(self, edges):
        """Given an edgelist, add each edge in the edgelist to the network"""
        i = 0
        for edge in edges:
            self.add_edge(edge)
            # print('Loading %i th edge out of %i edges' % (i, len(edges)))
            # i += 1

        # self.node = list(self.neighb.keys())

    def add_edge(self, edge):
        """Given an edge, add this edge to the Network"""
        # edge = [u, v]
        u, v = edge
        self.edges.append(edge)
        if u not in self.neighb:
            self.neighb[u] = set({v})
            self.vertices.append(u)
            self.vertices_set.add(u)
            self.number_nodes += 1

        else:
            self.neighb[u].add(v)

        if v not in self.neighb:
            self.neighb[v] = set({u})
            self.vertices.append(v)
            self.vertices_set.add(v)
            self.number_nodes += 1

        else:
            self.neighb[v].add(u)

    def add_nodes(self, nodes):
        """Given a list of nodes, adds the nodes to the Network"""

        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        """Given a single node, adds the node to the Network"""

        if node not in self.vertices:
            self.vertices.append(node)
            self.neighb[node] = set()
            self.number_nodes += 1
            self.vertices_set.add(node)

    def set_neighbors(self):

        for edge in self.edges:
            self.add_edge(edge)

    def get_edges(self):
        # this may create unecessary large lists
        set_edgelist = []
        for x in self.vertices:
            if x in self.neighb:
                for y in self.neighb[x]:
                    set_edgelist.append([x, y])
        self.edges = set_edgelist
        return self.edges

        # self.node = list(self.neighb.keys())

    def intersection(self, Network_other):
        """
        Given another network, returns all edges found in both networks.
        """
        common_nodeset = self.vertices_set.intersection(Network_other.vertices_set)
        common_edgelist = []

        for x in common_nodeset:
            for y in self.neighb[x].intersection(Network_other.neighbors(x)):
                common_edgelist.append([x, y])

        return common_edgelist

    def neighbors(self, node):
        """
        Given a node, returns all the neighbors of the node.
        """
        if node not in self.nodes():
            print('ERROR: %s not in the node set' % node)
        return self.neighb[node]

    def has_edge(self, u, v):
        """
        Given two nodes, returns true of these is an edge between then, false otherwise.
        """
        try:
            return v in self.neighb[u]
        except KeyError:
            return False

    def nodes(self, is_set=False):
        return self.vertices if not is_set else self.vertices_set

    def num_nodes(self):
        return self.number_nodes

    def edges(self):
        return self.edges

    def get_adjacency_matrix(self):
        mx = np.zeros(shape=(len(self.vertices), len(self.vertices)))
        for i in np.arange(len(self.vertices)):
            for j in np.arange(len(self.vertices)):
                if self.has_edge(self.vertices[i], self.vertices[j]) > 0:
                    mx[i, j] = 1
        return mx

    def subgraph(self, nodelist):
        # Take induced subgraph on the specified nodeset
        V0 = set(nodelist).intersection(self.vertices)
        G_sub = NNetwork()
        for u in V0:
            nbh_sub = self.neighbors(u).intersection(set(nodelist))
            if len(nbh_sub) > 0:
                for v in nbh_sub:
                    G_sub.add_edge([u, v])

        return G_sub

    def k_node_ind_subgraph(self, k, center=None):
        # Computes a random k-node induced subgraph
        # Initialized simple symmetric RW uniformly at "center" node,
        # and collects neighboring node until we get k distinct nodes
        # if center is None, then initial node is chosen uniformly at random
        # Once a set of k distinct nodes are collected, take the induced subgraph on it
        if k > len(self.vertices):
            print("cannot take i% distinct nodes from a graph with %i nodes" % (k, len(self.vertices)))

        V0 = []
        x = np.random.choice(self.vertices)
        if center is not None:
            x = center
        V0.append(x)

        while len(V0) < k:
            x = np.random.choice(list(self.neighbors(x)))
            V0.append(x)
            V0 = list(set(V0))

        # now take subgraph on V0:
        return self.subgraph(V0)


