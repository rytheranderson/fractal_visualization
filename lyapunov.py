import numpy as np
from numba import jit, prange
from image_creation import *
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

@jit
def lyapunov(string, xbound, ybound, maxiter=100, width=3, height=3, dpi=100, transpose=False):

	N_warmup = maxiter/3

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

	if transpose:
		lattice = lattice.T

	return (lattice, width, height, dpi)

if __name__ == '__main__':

	colors0 = np.array(plt.cm.YlGnBu_r(np.linspace(0, 1, 1000)))
	colors1 = np.array(plt.cm.YlGnBu_r(np.linspace(0, 1, 1000)))	
	colors = np.vstack((colors1, colors0))
	
	mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

	# interesting strings + regions
	# AABBBBBABABAAAAAA (2.0, 4.0) (2.0, 4.0)
	# AAAABA (2.60, 4.0) (2.45, 4.0)
	# AAB (2.5, 4.0) (2.5, 4.0)
	# maxiter 10-100 + large vert_exag yields a "liquidy" image, increasing maxiter sharpens lines to a more "metallic" image for some regions 
	# gamma > 1.0 tends to help for flat images (vert_exag=0) with 
	# decreasing maxiter < 20 removes fuzzy regions, but makes color variation difficult

	# M = lyapunov('AABBBBBABABAAAAAA', (2.00, 4.0), (2.00, 4.0), maxiter=200, dpi=600, width=3, height=2)
	# save_image_array(M, name='AABBBBBABABAAAAAA')
 
	# M = lyapunov('AAAABA', (2.60, 4.0), (2.45, 4.0), maxiter=10, dpi=400, width=4, height=2)
	# save_image_array(M, name='AAAABA')
 
	# M = lyapunov('AAB', (2.50, 4.0), (2.5, 4.0), maxiter=10, dpi=300, width=8, height=6)
	# save_image_array(M, name='AAB')

	# M = lyapunov('ABBAB', (2.0, 4.0), (2.0, 4.0), maxiter=70, dpi=600, width=12, height=8)
	# save_image_array(M, name='ABBAB')
	
	# M = open_image_array('AAB.pkl')

	# image(M, cmap=plt.cm.bone_r, gamma=0.5, image_type='tiff', vert_exag=0.01)
