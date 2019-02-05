import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors

from numba import jitclass
from numba import float64, int32

spec = [('xrange'  , float64[:]  ),
		('yrange'  , float64[:]  ),
		('width'   , int32       ), 
		('height'  , int32       ),
		('xvals'   , float64[:]  ),
		('yvals'   , float64[:]  ),
		('dpi'     , int32       ),
		('lattice' , float64[:,:])]

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

	def mandelbrot(self, maxiter, horizon, power):

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

	def julia(self, c, maxiter, horizon, power):

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

phi=(1 + 5 ** 0.5) / 2

for i in [0.155, 0.156, 0.157, 0.158, 0.159, 0.160]:
	jul=closure_fractal([-0.1,0.1],[-0.1,0.1],21,13,72)
	jul.julia(complex(-0.8,i), 2000, 2.0**40, 2)
	image(jul, cmap=plt.cm.binary, filename='julia' + str(i), gamma=0.6, vert_exag=0.001)


