# import libraries
import sys; sys.path.insert(1,'/home/richpaulyim/Projects/Kuramoto-ML')
from kuramoto import *
np.set_printoptions(threshold=sys.maxsize)

# generate the lattice
n=30
lattice = edgeset_generator([n,n,n], show=False)
initialcolors = np.zeros(n**3)
phased_lattice = ColorNNetwork(initialcolors, lattice)

# start simulations
coupling_constant = 1.4
a = simulate_Kuramoto(phased_lattice, coupling_constant, 
        T=100, step=0.05, verbose=True) 
print(np.max(a),np.min(a))
print(a)
print(len(a[np.abs(np.asarray(a))>np.pi]))
# create animations
lattice3D_mkgif(a, dim=n, name="kura3d_"+str(coupling_constant), 
        duration=5)#,pickcol='Greys')

