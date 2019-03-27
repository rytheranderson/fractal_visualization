import numpy as np
from cmath import sin, cos, tan, sqrt, exp
from math import e

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors

from numba import jit, jitclass
from numba import float64, int32

pi  = np.pi
phi = (1 + 5 ** 0.5) / 2

spec = [('xrange'  , float64[:]  ),
		('yrange'  , float64[:]  ),
		('width'   , int32       ), 
		('height'  , int32       ),
		('xvals'   , float64[:]  ),
		('yvals'   , float64[:]  ),
		('dpi'     , int32       ),
		('lattice' , float64[:,:])]

# I am aware that a single function can be used for each Julia type with an "update" function argument
# (that updates the value of z) however I was unable to get this to work with the @jitclass decorator, 
# which I think is needed to make this not take forever, hence the clunky code.

@jitclass(spec)
class closure_fractal(object):
	def __init__(self, xbound, ybound, width, height, dpi):

		xmin,xmax = [float(xbound[0]),float(xbound[1])]
		ymin,ymax = [float(ybound[0]),float(ybound[1])]

		self.xrange = np.array([xmin,xmax], dtype=np.float64)
		self.yrange = np.array([ymin,ymax], dtype=np.float64)
		self.width  = width
		self.height = height

		nx = width*dpi
		ny = height*dpi

		xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
		yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

		self.xvals = xvals
		self.yvals = yvals

		self.dpi = dpi
		self.lattice = np.zeros((int(width*dpi), int(height*dpi)), dtype=np.float64)

	def mandelbrot_poly(self, maxiter, horizon, power):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				c = xvals[i] + 1j * yvals[j]
				z = c

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						break
					z = z**power + c

	def julia_poly(self, c, maxiter, horizon, power):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						break
					z = z**power + c

	def julia_cos(self, c, maxiter, horizon, log_smooth):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						if log_smooth:
							self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						else:
							self.lattice[i,j] = n
						break
					z = c*cos(z)

	def julia_sin(self, c, maxiter, horizon, log_smooth):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						if log_smooth:
							self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						else:
							self.lattice[i,j] = n
						break
					z = c*sin(z)

	def julia_mag1(self, c, maxiter, horizon):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						break
					z = ( (z*z+c-1) / (2*z+c-2) ) * ( (z*z+c-1) / (2*z+c-2) )

	def julia_mag2(self, c, maxiter, horizon):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
					z = ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) ) * ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) )

	def julia_nd_rational(self, c, maxiter, horizon, n, d):

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				if abs(z) == 0.0:
					z = z + 0.1

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						break
					z = z**n + (c/z**d)

	def cantor_bouquet(self, c, maxiter, horizon, log_smooth):

		""" logarithmic smoothing gives infinite values, a good horizon value
			to start with is 1.0E10
		"""

		xvals   = self.xvals
		yvals   = self.yvals

		log_horizon = np.log(np.log(horizon))/np.log(2)

		for i in xrange(len(xvals)):
			for j in xrange(len(yvals)):

				z = xvals[i] + 1j * yvals[j]

				for n in xrange(maxiter):
					az = abs(z)
					if az > horizon:
						if log_smooth:
							self.lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
						else:
							self.lattice[i,j] = n
						break
					z = c * exp(z)

def image(instance, cmap=plt.cm.hot, filename='f', image_type='png', ticks='off', gamma=0.3, vert_exag=0, ls=[315,10]):

	A = instance.lattice.T

	w,h = plt.figaspect(A)
	fig, ax0 = plt.subplots(figsize=(w,h), dpi=instance.dpi)
	fig.subplots_adjust(0,0,1,1)
	plt.axis(ticks)

	norm = colors.PowerNorm(gamma)
	light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])
	M = light.shade(A, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode='hsv')
	ax0.imshow(M, origin='lower')

	F = plt.gcf()
	F.set_size_inches(instance.width, instance.height)

	fig.savefig(filename + '.' + image_type, dpi=instance.dpi)

#------------------------------------------------------------------------------#
#                             Julia set c values                               #
#------------------------------------------------------------------------------#

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
c = pi/2*(2.0 + 0.25j)
#c = pi/2*(1.5 + 0.05j)

### mag1 ###
#c = 1j

### mag2 ###
#c = 1.5 + 0.75j
#c = 2.0 + 0.80j

jul=closure_fractal([-0.5,0.5],[-0.5,0.5],21,13,300)
jul.julia_cos(c, 5000, 2**40, False)
image(jul, cmap=plt.cm.gnuplot2_r, filename='julia', gamma=0.25, vert_exag=0)

