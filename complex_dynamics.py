import numpy as np
from cmath import sin, cos, exp
from numba import jit, prange

pi  = np.pi
phi = (1 + 5 ** 0.5) / 2

@jit
def power(z, c, n):
	return z**n + c

@jit
def cosine(z, c, args):
	return c*cos(z)

@jit
def sine(z, c, args):
	return c*sin(z)

@jit
def exponential(z, c, args):
	return c*exp(z)

@jit
def magnetic_1(z, c, args):
	return ( (z*z+c-1) / (2*z+c-2) ) * ( (z*z+c-1) / (2*z+c-2) )

@jit
def magnetic_2(z, c, args):
	return ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) ) * ( (z*z*z * 3*(c-1)*z + (c-1)*(c-2) ) / ( 3*z*z + 3*(c-2)*z + (c-1)*(c-2) + 1) )

@jit(nopython=True, parallel=True)
def mandelbrot(xbound, ybound, update_func, args=2, width=5, height=5, dpi=100, maxiter=100, horizon=2.0**40, log_smooth=True):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
	yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

	lattice = np.zeros((int(nx), int(ny)), dtype=np.float64)

	log_horizon = np.log(np.log(horizon))/np.log(2)

	for i in prange(len(xvals)):
		for j in prange(len(yvals)):

			c = xvals[i] + 1j * yvals[j]
			z = c

			for n in prange(maxiter):

				az = abs(z)

				if az > horizon:
					if log_smooth:
						lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
					else:
						lattice[i,j] = n
					break

				z = update_func(z, c, args)

	return (lattice, width, height, dpi)

@jit(nopython=True, parallel=True)
def julia(c, xbound, ybound, update_func, args=2, width=5, height=5, dpi=100, maxiter=100, horizon=2.0**40, log_smooth=True):

	xmin,xmax = [float(xbound[0]),float(xbound[1])]
	ymin,ymax = [float(ybound[0]),float(ybound[1])]

	nx = width*dpi
	ny = height*dpi

	xvals  = np.array([xmin + i*(xmax - xmin)/(nx) for i in range(nx)], dtype=np.float64)
	yvals  = np.array([ymin + i*(ymax - ymin)/(ny) for i in range(ny)], dtype=np.float64)

	lattice = np.zeros((int(nx), int(ny)), dtype=np.float64)

	log_horizon = np.log(np.log(horizon))/np.log(2)

	for i in prange(len(xvals)):
		for j in prange(len(yvals)):

			z = xvals[i] + 1j * yvals[j]

			for n in range(maxiter):

				az = abs(z)

				if az > horizon:
					if log_smooth:
						lattice[i,j] = n - np.log(np.log(az))/np.log(2) + log_horizon
					else:
						lattice[i,j] = n
					break

				z = update_func(z, c, args)

	return (lattice, width, height, dpi)

def julia_series(c_vals, xbound, ybound, update_func, args=2, width=5, height=5, dpi=100, maxiter=100, horizon=2.0**40, log_smooth=True):

	series = []
	
	for c in c_vals:
		l = julia(c, xbound, ybound, update_func, args=args, width=width, height=height, dpi=dpi, maxiter=maxiter, horizon=horizon, log_smooth=log_smooth)
		series.append(l)

	return series

if __name__ == '__main__':

	pass
