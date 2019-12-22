import numpy as np
from numpy.random import randint, random, choice
from cmath import sin, cos, tan, sqrt, exp
from math import e, floor
import time
import sys
import pickle

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma

from numba import jit, prange
from complex_dynamics import power, cosine, sine, exponential, magnetic_1, magnetic_2, nd_rational, mandelbrot

def color_map_image(args, binary=True, cmap=plt.cm.hot, filename='f', image_type='png', ticks='off', gamma=1.0, vmin=0.0, vmax=1.0, vert_exag=0, ls=[315,10]):

	A, width, height, dpi = args
	A = A.T
	
	maxcount = np.max(A)

	if maxcount == 0:
		raise ValueError('No colored points observed, change the boundaries or increase maxiter.')

	A /= maxcount

	w,h = plt.figaspect(A)
	fig, ax0 = plt.subplots(figsize=(w,h), dpi=dpi)
	fig.subplots_adjust(0,0,1,1)
	plt.axis(ticks)

	norm = colors.PowerNorm(gamma, )
	light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])
	M = light.shade(A, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode='hsv')
	ax0.imshow(M, origin='lower')

	F = plt.gcf()
	F.set_size_inches(width, height)

	fig.savefig(filename + '.' + image_type, dpi=dpi)

def nebula_image(AB, AG, AR, filename='f', image_type='png', ticks='off', gammas=(1.0,1.0,1.0), vmin=0.0, vmax=1.0):

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
	
	M[0,0,0] *= gammas[0]
	M[0,0,1] *= gammas[1]
	M[0,0,2] *= gammas[2]
	
	ax0.imshow(M, origin='lower', vmin=vmin, vmax=vmax)

	F = plt.gcf()
	F.set_size_inches(width, height)

	fig.savefig(filename + '.' + image_type, dpi=dpi)

@jit(nopython=True)
def weighted_choice(seq, weights):

	x = random()

	for i, elmt in enumerate(seq):

		if x <= weights[i]:
			return elmt
		x -= weights[i]

@jit(nopython=True)
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

	energy_grid = mandelbrot(xbound, ybound, update_func, args=args, width=width, height=height, dpi=dpi, maxiter=2000, horizon=2.5)[0]
	energy_grid = energy_grid/energy_grid.sum()

	candidates = [(i,j) for i in range(nx) for j in range(ny)]
	candidate_indices = [i for i in range(len(candidates))]
	probs = [energy_grid[l[0],l[1]] for  l in candidates]
	cvals = []

	for k in range(Ncvals):

		if random() < importance_weight:

			i,j= weighted_choice(candidates, probs)
			xlo,xhi = xboxs[i]
			ylo,yhi = yboxs[j]
			c = xlo+(random()*(xhi-xlo)) + 1j*(ylo+(random()*(yhi-ylo)))

		else:

			c = xmin+(random()*(xmax-xmin)) + 1j*(ymin+(random()*(ymax-ymin)))

		cvals.append(c)

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

		for N in prange(maxiter):
	
			az = np.abs(z)
			trial_sequence.append(z)

			if az > horizon:
				sequence.extend(trial_sequence)
				break

			z = update_func(z, c, args)

		for k in range(len(sequence)):

			indx = 0
			indy = 0
	
			c = sequence[k]
	
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

start_time = time.time()

xB = np.array([-1.35, -1.13])
yB = np.array([0.00,  0.20])

cvals = compute_cvals(10000, xB, yB, power, args=2, width=3, height=4, dpi=100)

bud0 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=100, width=3, height=4, dpi=100)
save_image_array(bud0, name='save0')

bud1 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=1000, width=3, height=4, dpi=100)
save_image_array(bud1, name='save1')

bud2 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=10000, width=3, height=4, dpi=100)
save_image_array(bud2, name='save2')

nebula_image(bud0, bud1, bud2, gammas=(0.35,0.4,0.4))

print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))
