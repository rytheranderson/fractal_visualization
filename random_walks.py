import numpy as np
from numpy import array
from numpy.random import randint
from itertools import combinations
from numba import jit
from image_creation import image
from matplotlib import pyplot as plt

def construct_moves(basis):

	basis = np.r_[basis,-1*basis,[array([0,0,0])]]
	moves = np.unique(array([b0 + b1 for b0,b1 in combinations(basis,2)]), axis=0)
	moves = array([m for m in moves if np.any(m)])

	return moves

@jit(nopython=True)
def random_walk_3D(moves, niter, width=5, height=5, depth=1, dpi=100, tracking='visitation', displacement=0.0):

	lattice = np.zeros((int(height*dpi), int(width*dpi), int(depth)), dtype=np.float32)
	shape = array([height*dpi, width*dpi, depth])
	nmoves = len(moves)
	
	start = array([height*dpi, width*dpi, depth])/2.0 + displacement*shape

	l0,l1,l2 = shape
	indices = start

	for n in range(niter):

		move = moves[randint(0, nmoves)]
		indices += move
		i,j,k = int(indices[0]%l0), int(indices[1]%l1), int(indices[2]%l2)

		if tracking == 'visitation':
			lattice[i,j,k] += 1.0
		elif tracking == 'temporal':
			if lattice[i,j,k] == 0.0: lattice[i,j,k] += n

	lattice = np.sum(lattice, axis=2)
	
	return (lattice, width, height, dpi)
	
basis = array([[1,0,0],[0,1,0],[0,0,1]])
#basis = array([[1,0,0],[0,1,0]])

moves = construct_moves(basis)
M = random_walk_3D(moves, 10000000, depth=1, dpi=100, displacement=0.0, tracking='temporal')
image(M, cmap=plt.cm.gist_yarg, gamma=1.0)


		