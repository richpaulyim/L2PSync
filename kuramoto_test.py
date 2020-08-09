# import libraries
import sys; sys.path.insert(1,'/home/richpaulyim/Projects/Kuramoto-ML')
from kuramoto import *
np.set_printoptions(threshold=sys.maxsize)

# generate the lattice
n=100
lattice = edgeset_generator([n,n], show=False)
initialcolors = np.zeros(n*n)
phased_lattice = ColorNNetwork(initialcolors, lattice)

# start simulations
coupling_constant = 1.33
a = simulate_Kuramoto(phased_lattice, coupling_constant, 
        T=30, step=0.2, verbose=True) 
# create animations
lattice2D_mkgif(a, n=n, name="kura50_"+str(coupling_constant), freeze=True,
        duration=3,pickcol='inferno')

#n=100
#lattice = edgeset_generator([n,n], show=False)
#initialcolors = np.zeros(n*n)
#phased_lattice = ColorNNetwork(initialcolors, lattice)
#
## start simulations
#coupling_constant = 1
#a = simulate_Kuramoto(phased_lattice, coupling_constant, 
#        T=3000, step=0.1, verbose=True) 
## create animations
#lattice2D_mkgif(a, n=n, name="kura50_"+str(coupling_constant), freeze=True,
#        duration=3,pickcol='inferno')
