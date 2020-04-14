import numpy as np
from numba import jit, prange
from image_creation import *
from matplotlib import pyplot as plt
from numpy.random import shuffle

@jit
def lyapunov(string, xbound, ybound, maxiter=100, N_warmup=20, width=3, height=3, dpi=100):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
	yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

	lattice = np.zeros((int(nx), int(ny)), dtype=np.float64)

	count = 0
	for i in prange(len(xvals)):
		for j in prange(len(yvals)):

			x = 0.5
			lamd = 0.0

			xv = xvals[j]
			yv = xvals[i]

			for n in range(N_warmup):

				S = string[count%len(string)]
				if S == 'A':
					rn = xv
				else:
					rn = yv
				count += 1

			for n in range(maxiter):

				S = string[count%len(string)]
				if S == 'A':
					rn = xv
				else:
					rn = yv
				count += 1

				x = (rn*x) * (1-x)
				lamd += np.log(np.abs(rn * (1 - (2*x))))

			lamd /= maxiter
			lattice[i,j] += lamd

	return (lattice, width, height, dpi)

lyap_cmap = plt.get_cmap('nipy_spectral')    
M = lyapunov('AB', (1.51,4), (1.51,4), maxiter=100, dpi=300)
image(M, cmap=plt.cm.gist_ncar, gamma=3.0)


