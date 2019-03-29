import numpy as np
from cmath import sin, cos, tan, sqrt, exp
from math import e, floor

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

from numba import jit

pi  = np.pi
phi = (1 + 5 ** 0.5) / 2

@jit
def power(z, c, n=2):
	return z**n + c

@jit
def cosine(z, c, phase='cos'):
	return c*cos(z)

@jit
def sine(z, c, phase='sin'):
	return c*sin(z)

@jit
def exponential(z, c, phase='cos_sin'):
	return c*exp(z)

@jit
def magnetic_1(z, c, dummy=''):
	return ( (z*z+c-1) / (2*z+c-2) ) * ( (z*z+c-1) / (2*z+c-2) )

@jit
def magnetic_2(z, c, dummy=''):
	return ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) ) * ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) )

@jit
def nd_rational(z, c, nd=(2,2)):
	n,d = nd
	return z**n + (c/z**d)

@jit
def mandelbrot(xbound, ybound, width, height, dpi, maxiter, horizon, log_smooth, update_func, **kwargs):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
	yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

	lattice = np.zeros((int(width*dpi), int(height*dpi)), dtype=np.float64)

	log_horizon = np.log(np.log(horizon))/np.log(2)

	for i in xrange(len(xvals)):
		for j in xrange(len(yvals)):

			c = xvals[i] + 1j * yvals[j]
			z = c

			for n in xrange(maxiter):
				az = abs(z)
				if az > horizon:
					if log_smooth:
						lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
					else:
						lattice[i,j] = n
					break
				z = update_func(z, c, kwargs)

	return (lattice, width, height, dpi)

@jit
def julia(xbound, ybound, width, height, dpi, c, maxiter, horizon, log_smooth, update_func, **kwargs):
	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
	yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

	lattice = np.zeros((int(width*dpi), int(height*dpi)), dtype=np.float64)

	log_horizon = np.log(np.log(horizon))/np.log(2)

	for i in xrange(len(xvals)):
		for j in xrange(len(yvals)):

			z = xvals[i] + 1j * yvals[j]

			for n in xrange(maxiter):
				az = abs(z)
				if az > horizon:
					if log_smooth:
						lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
					else:
						lattice[i,j] = n
					break
				z = update_func(z, c, kwargs)

	return (lattice, width, height, dpi)

def julia_series(c_vals, xbound, ybound, width, height, dpi, maxiter, horizon, log_smooth, uf_args):

	series = []
	for c in c_vals:
		l = julia(xbound, ybound, width, height, dpi, c, maxiter, horizon, log_smooth, uf_args[0], uf_args[1])
		series.append(l)

	return series

def image(lattice, cmap=plt.cm.hot, filename='f', image_type='png', ticks='off', gamma=0.3, vert_exag=0, ls=[315,10]):

	A, width, height, dpi = lattice
	A = A.T

	w,h = plt.figaspect(A)
	fig, ax0 = plt.subplots(figsize=(w,h), dpi=dpi)
	fig.subplots_adjust(0,0,1,1)
	plt.axis(ticks)

	norm = colors.PowerNorm(gamma)
	light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])
	M = light.shade(A, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode='hsv')
	ax0.imshow(M, origin='lower')

	F = plt.gcf()
	F.set_size_inches(width, height)

	fig.savefig(filename + '.' + image_type, dpi=dpi)

def animate(series, fps=15, bitrate=1800, cmap=plt.cm.hot, filename='f', ticks='off', gamma=0.3, vert_exag=0, ls=[315,10]):

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)

	norm = colors.PowerNorm(gamma)
	light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])

	foo, width, height, dpi = series[0]
	FIG = plt.figure()
	F = plt.gcf()
	F.set_size_inches(width, height)
	plt.axis(ticks)
	ims = []

	for s in series:

		A, width, height, dpi = s
		A = A.T
		M = light.shade(A, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode='hsv')
		im = plt.imshow(M, origin='lower', norm=norm)
		ims.append([im])

	ani = animation.ArtistAnimation(FIG, ims, interval=50, blit=True, repeat_delay=1000)
	ani.save(filename + '.mp4', dpi=dpi)

cvals = [complex(0,i) for i in np.linspace(0.01, 1, 100)]
s=julia_series(cvals,[-1.5,1.5],[-1.5,1.5], 5, 5, 100, 100, 2**40, True, [power, (2)])
animate(s, gamma=0.8, cmap=plt.cm.gnuplot2)


#l=julia([-1.5,0.5],[-1.2,1.1], 5, 5, 100, 1j, 100, 2**40, True, cosine, (2))
#image(l, cmap=plt.cm.gnuplot2, filename='mandelbrot', gamma=0.2)

#------------------------------------------------------------------------------#
#                                example sets                                  #
#------------------------------------------------------------------------------#

#----- Julia sets -----#

### quadratic ###
#c = 1j              # dentrite fractal
#c = -0.123 + 0.745j # douady's rabbit fractal
#c = -0.750 + 0j     # San Marco fractal
#c = -0.391 - 0.587j # Siegel disk fractal
#c = -0.7 - 0.3j     # interesting 1
#c = -0.75 - 0.2j    # interesting 2
#c = -0.75 + 0.15j   # interesting 3
#c = -0.7 + 0.35j    # interesting 4

### cosine ###
#c = 1.0 - 0.5j        
#c = pi/2*(1.0 + 0.6j) 
#c = pi/2*(1.0 + 0.4j) 
#c = pi/2*(2.0 + 0.25j)
#c = pi/2*(1.5 + 0.05j)

### mag1 ###
#c = 1.1j

### mag2 ###
#c = 1.5 + 0.75j
#c = 2.0 + 0.80j

### cantor bouquet ###
#c = 1.0/e
#c = 0.5/e
#c = 5.0
#c = 1.025/e
#jul=closure_fractal([0,2],[-1,1],21,13,300)
#jul.cantor_bouquet(c, 1000, 2**40, True)
#image(jul, cmap=plt.cm.afmhot, filename='cantor_bouquet', gamma=0.9, vert_exag=0.0001)

#----- Mandelbrot sets -----#




