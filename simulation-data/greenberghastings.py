import time
import numpy as np
def simulate_GH(G, kappa, type='GH', its = int(pow(2,32)-1), verbose=True, timesec=0, tree=False, no_edges=0):
    # initial conditions
    # blink color and lower bound on post-blink
    colors = []
    init_col = G.get_colors()
    colors.append(init_col)
    # local helper function
    def new_color(vertex):
        neighblist = G.get_neighbor_colors(vertex)
        if G.get_color(vertex) == 0 and any(np.array(neighblist) == 1):
            return 1
        if G.get_color(vertex) == 0 and all(np.array(neighblist) != 1):
            return 0
        else:
            return (G.get_color(vertex) + 1) % 5
    # initial coloring and start time
    current_col = G.get_colors()
    s = time.time()
    for i in range(1, its+1):
        # perform updates for each vertex
        current_col = list(map(new_color, range(0, G.num_nodes())))
        #import pdb; pdb.set_trace()
        # check synchronization -> break and return final list and iterations
        if all(np.array(current_col) == 0):
            if verbose: print("Iterations: ",i)
            if no_edges:
                return (colors, 1, i)
            return (colors, 1, i)
        if timesec and time.time()-s > timesec:
            if verbose:
                print("Iterations: ",i)
                print("Time has been reached: ", timesec,"sec.")
            if no_edges:
                return (colors, 0, i)
            return (colors, 0, i)
        # update to new colors
        G.set_colors(current_col)
        colors.append(np.array(G.get_colors()))
        if tree:
            values, parentlist = tree_iter(G, values, parentlist)
            tree = not (len(G.edges) == G.num_nodes()-1)
            if (len(G.edges)<G.num_nodes()-1):
                print("deleted too many edges")
    if verbose:
        print("Max iterations reached: ", its)
    if no_edges:
        return (colors, 0, its)
    return (colors, 0, its)
