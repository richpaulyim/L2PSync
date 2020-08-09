# import libraries
import sys; sys.path.insert(1,'/home/richpaulyim/Projects/Kuramoto-ML')
from kuramoto import *
np.set_printoptions(threshold=sys.maxsize)

# generate the lattice
n=80
lattice = edgeset_generator([n,n], show=False)
initialcolors = np.zeros(n*n)
phased_lattice = ColorNNetwork(initialcolors, lattice)

# start simulations
coupling_constant = 1.2
a = simulate_Kuramoto(phased_lattice, coupling_constant, T=100, step=0.02, verbose=True) 
print(np.max(a),np.min(a))
print(len(a[np.abs(np.asarray(a))>np.pi]))
# create animations
lattice2D_mkgif(a, n=n, name="kura100", freeze=True,
        cap=400,duration=10)#,pickcol='Greys')

