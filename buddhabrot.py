import numpy as np
from numpy.random import randint, random, choice
from math import e, floor, ceil
import time
import sys
import pickle

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from skimage.restoration import denoise_nl_means, estimate_sigma

from numba import jit, prange
from complex_dynamics import *

def nebula_image(AB, AG, AR, filename='f', image_type='png', ticks='off', gamma=1.0, denoise=False):

	A_blue, width, height, dpi = AB
	A_green = AG[0]
	A_red = AR[0]

	A_blue = A_blue.T
	A_green = A_green.T
	A_red = A_red.T

	A_blue /= np.amax(A_blue)
	A_green /= np.amax(A_green)
	A_red /= np.amax(A_red)

	w,h = plt.figaspect(A_blue)
	fig, ax0 = plt.subplots(figsize=(w,h), dpi=dpi)
	fig.subplots_adjust(0,0,1,1)
	plt.axis(ticks)

	M = np.dstack((A_red, A_green, A_blue))

	if denoise:
		sigma_est = np.mean(estimate_sigma(M, multichannel=True))
		patch_kw = dict(patch_size=9, patch_distance=15, multichannel=True)
		M = denoise_nl_means(M, h=0.9*sigma_est, fast_mode=True, **patch_kw)

	ax0.imshow(M**gamma, origin='lower')
	F = plt.gcf()
	F.set_size_inches(width, height)

	fig.savefig(filename + '.' + image_type, dpi=dpi)

@jit
def compute_cvals(Ncvals, xbound, ybound, update_func, args=2, width=5, height=5, dpi=100, importance_weight=0.75):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax-xmin)/(nx) for i in range(nx)], dtype=np.float32)
	yvals  = np.array([ymin + i*(ymax-ymin)/(ny) for i in range(ny)], dtype=np.float32)
	xboxs = [(xvals[i],xvals[i+1]) for i in range(len(xvals)-1)]
	yboxs = [(yvals[i],yvals[i+1]) for i in range(len(yvals)-1)]
	xboxs = xboxs + [(xboxs[-1][1], xmax)]
	yboxs = yboxs + [(yboxs[-1][1], ymax)]

	NR = int(round(Ncvals * (1-importance_weight)))
	cvals = []

	for k in range(NR):

		c = xmin+(random()*(xmax-xmin)) + 1j*(ymin+(random()*(ymax-ymin)))
		cvals.append(c)

	if importance_weight > 0.0:

		NI = int(round(Ncvals * importance_weight))
		energy_grid = mandelbrot(xbound, ybound, update_func, args=args, width=width, height=height, dpi=dpi, maxiter=1000, horizon=2.5, log_smooth=False)[0]
		energy_grid = (energy_grid/energy_grid.sum()) * NI

		for i in range(nx):
			for j in range(ny):
	
				N = int(round(energy_grid[i,j]))
		
				xlo,xhi = xboxs[i]
				ylo,yhi = yboxs[j]
				cs = xlo+(random(N)*(xhi-xlo)) + 1j*(ylo+(random(N)*(yhi-ylo)))
		
				cvals.extend(list(cs))

	return np.array(cvals)

@jit(nopython=True, parallel=True)
def buddhabrot(xbound, ybound, cvals, update_func, args=2, width=5, height=5, dpi=100, maxiter=100, horizon=1.0E6):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax-xmin)/(nx) for i in range(nx)], dtype=np.float32)
	yvals  = np.array([ymin + i*(ymax-ymin)/(ny) for i in range(ny)], dtype=np.float32)
	xboxs = [(xvals[i],xvals[i+1]) for i in range(len(xvals)-1)]
	yboxs = [(yvals[i],yvals[i+1]) for i in range(len(yvals)-1)]
	xboxs = xboxs + [(xboxs[-1][1], xmax)]
	yboxs = yboxs + [(yboxs[-1][1], ymax)]

	lattice = np.zeros((int(width*dpi), int(height*dpi)), dtype=np.float32)

	for c in cvals:

		z = c
		trial_sequence = []
		sequence = []

		for N in range(maxiter):
	
			az = np.abs(z)
			trial_sequence.append(z)

			if az > horizon:
				sequence.extend(trial_sequence)
				break

			z = update_func(z, c, args)

		for c in sequence:

			indx = 0
			indy = 0
	
			for bx in range(nx):
				if xboxs[bx][0] < c.real < xboxs[bx][1]:
					indx += bx
					break
	
			for by in range(ny):
				if yboxs[by][0] < c.imag < yboxs[by][1]:
					indy += by
					break
	
			if indx != 0 and indy != 0:
				lattice[indx,indy] += 1

	return (lattice, width, height, dpi)

def save_image_array(A, name='save'):

	with open(name + '.pkl','wb') as f:
		pickle.dump(A, f)

def open_image_array(file):

	with open(file,'rb') as f:
		A = pickle.load(f)
		return A

def run_nebula(xB, yB, Ncvals, update_func, args=2, importance_weight=0.5, width=5, height=5, dpi=100, maxiters=(100,1000,10000)):

	start_time = time.time()
	mi0, mi1, mi2 = maxiters
	
	cvals = compute_cvals(Ncvals, xB, yB, update_func, args=args, width=width, height=height, dpi=dpi, importance_weight=importance_weight)

	bud0 = buddhabrot(xB, yB, cvals, update_func, args=args, horizon=1.0E6, maxiter=mi0, width=width, height=height, dpi=dpi)
	save_image_array(bud0, name='save0')
	
	bud1 = buddhabrot(xB, yB, cvals, update_func, args=args, horizon=1.0E6, maxiter=mi1, width=width, height=height, dpi=dpi)
	save_image_array(bud1, name='save1')
	
	bud2 = buddhabrot(xB, yB, cvals, update_func, args=args, horizon=1.0E6, maxiter=mi2, width=width, height=height, dpi=dpi)
	save_image_array(bud2, name='save2')
	
	nebula_image(bud0, bud1, bud2, gamma=0.5)
	
	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

### interesting regions

xB = np.array([-2.85, 2.85])
yB = np.array([-1.70, 1.70])

#xB = np.array([-1.35, -1.13])
#yB = np.array([0.00,  0.20])

#run_nebula(xB, yB, 50000, power, args=4)

