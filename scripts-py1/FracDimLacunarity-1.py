
# Source: http://connor-johnson.com/2014/03/04/fractal-dimension-and-box-counting/

import numpy as np
import scipy
import scipy.stats
from pandas import Series, DataFrame
import scipy.optimize

from matplotlib import pyplot as plt
import random

#x = range(100)
#u = [random.random() for x1 in x]
#plt.hist(u, 100, alpha=0.5)
#plt.show()

# Matern point process

L = 64
U = list()
rate = L / 6.0
N = scipy.stats.poisson( rate ).rvs()
U = list( scipy.stats.uniform(0,L).rvs(N) )

def matern_children( U, rate, nbhd ):
    T = list()
    for i in range( len( U ) ):
        M = scipy.stats.poisson( rate ).rvs()
        for j in range( M ):
            T.append( scipy.stats.uniform( U[i]-nbhd, 2*nbhd ).rvs() )
    return T

nbhd = 2.0
U = matern_children( U, rate, nbhd )
U = matern_children( U, rate, nbhd/4.0 )
U = matern_children( U, rate, nbhd/8.0 )
U = matern_children( U, rate, nbhd/16.0 )

plt.hist( U, 100, alpha=0.5 ) ;
plt.title('Distribution of Four Tier Matern Point Process')
plt.ylabel('Frequency') ; plt.xlabel('Location') ;
plt.savefig('matern_fractal_distribution.png', fmt='png', dpi=100 )

# Box counting algorithm
def count_boxes( data, box_size, M ):
    data = Series( data )
    N = np.int( np.floor( M / box_size ) )
    counts = list()
    for i in range( N ):
        condition = ( data >= i*box_size )&( data < (i+1)*box_size )
        subset = data[ condition ]
        counts.append( subset.count() )
    counts = [ i for i in counts if i != 0 ]
    return len( counts )

r = np.array([ L/(2.0**i) for i in range(12,0,-1) ])
N = [ count_boxes( U, ri, L ) for ri in r ]
