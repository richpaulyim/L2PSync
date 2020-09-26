# import libraries
import sys; sys.path.insert(1,'/home/richpaulyim/Projects/Kuramoto-ML')
from kuramoto import *
np.set_printoptions(threshold=sys.maxsize)

# generate the lattice
n=10
lattice1 = edgeset_generator([n,n], show=False)
lattice2 = edgeset_generator([n,n], show=False)
for i in range(30):
    initialcolors = np.random.uniform(-np.pi,np.pi,n*n)
    phased_lattice1 = ColorNNetwork(initialcolors, lattice1)
    phased_lattice2 = ColorNNetwork(initialcolors, lattice2)
    # start simulations
    coupling_constant = 1
    a = simulate_Kuramoto(phased_lattice1, coupling_constant, 
            T=60, step=0.05, verbose=True, intrinsic=False, fixed=0, rounded=True)
    print(a[1])
    print(a[0].shape)
    print(a[0][-1])
    #if a[1]==0:
    if 0:
        b = a
        a = simulate_Kuramoto(phased_lattice2, coupling_constant, 
            T=60, step=0.05, verbose=True, intrinsic=False, fixed=0,
            rounded=False)
        # create animations 
        lattice2D_mkgif(a[0], n=n,
                name="kura50b_freq0_uniform_startr"+str(coupling_constant),
                freeze=True, cap=200,
                duration=10,pickcol='Greys')
        lattice2D_mkgif(b[0], n=n,
                name="kura50b_freq0_uniform_startnr"+str(coupling_constant),
                freeze=True, cap=200,
                duration=10,pickcol='Greys')



#coupling_constant = 1.68
#a = simulate_Kuramoto(phased_lattice, coupling_constant, 
#        T=60, step=0.02, verbose=True) 
## create animations
#lattice2D_mkgif(a, n=n, name="kura50b_"+str(coupling_constant), freeze=True,
#        duration=3,pickcol='Greys')
#
#coupling_constant = 0.9
#a = simulate_Kuramoto(phased_lattice, coupling_constant, 
#        T=60, step=0.02, verbose=True) 
## create animations
#lattice2D_mkgif(a, n=n, name="kura50b_"+str(coupling_constant), freeze=True,
#        duration=3,pickcol='Greys')
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

# sample the state space


