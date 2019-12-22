from complex_dynamics import *
from random_walks import random_walk
from buddhabrot import *
import time

#----- Mandelbrot images -----#

def mandelbrot_ex0():

	xB = (-1.70, 0.75)
	yB = (-1.25, 1.25)

	start_time = time.time()

	man = mandelbrot(xB, yB, power, args=2, width=5, height=5, maxiter=500, dpi=300)
	image(man, cmap=plt.cm.hot, filename='mandelbrot_ex0', gamma=0.20)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

def mandelbrot_ex1():

	xB = (-0.748770, -0.748720)
	yB = ( 0.065053,  0.065103)

	start_time = time.time()

	man = mandelbrot(xB, yB, power, args=2, width=5, height=5, maxiter=5000, dpi=300)
	image(man, cmap=plt.cm.gist_yarg, filename='mandelbrot_ex1', gamma=0.3)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

#----- Julia images -----#

def julia_ex0():

	c = 1j
	xB = (-1.5, 1.5)
	yB = (-1.5, 1.5)

	start_time = time.time()

	jul = julia(c, xB, yB, power, args=2, width=5, height=5, maxiter=500, dpi=300)
	image(jul, cmap=plt.cm.viridis, filename='julia_ex0', gamma=0.6)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

def julia_ex1():

	c = -0.391 - 0.587j 
	xB = (-1.5, 1.5)
	yB = (-1.5, 1.5)

	start_time = time.time()

	jul = julia(c, xB, yB, power, args=2, width=5, height=5, maxiter=1000, dpi=300)
	image(jul, cmap=plt.cm.gist_ncar, filename='julia_ex1', gamma=0.6)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

def cos_julia_ex():

	c = pi/2*(1.0 + 0.6j) 
	xB = (-1.5, 1.5)
	yB = (-1.5, 1.5)

	start_time = time.time()

	jul = julia(c, xB, yB, cosine, args='', width=5, height=5, maxiter=1000, dpi=300)
	image(jul, cmap=plt.cm.plasma, filename='cos_julia_ex', gamma=1.0)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

def mag_julia_ex0():

	c = 2.0j
	xB = (-2.5, 0.5)
	yB = (-2.5, 0.5)

	start_time = time.time()

	jul = julia(c, xB, yB, magnetic_1, args='', width=5, height=5, maxiter=1000, dpi=300)
	image(jul, cmap=plt.cm.gist_ncar, filename='mag_julia_ex0', gamma=0.5)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

def mag_julia_ex1():

	c = 2.0 + 0.80j
	xB = (-1.0, 1.0)
	yB = (-1.0, 1.0)

	start_time = time.time()

	jul = julia(c, xB, yB, magnetic_2, args='', width=5, height=5, maxiter=1000, dpi=300)
	image(jul, cmap=plt.cm.gist_ncar, filename='mag_julia_ex1', gamma=1.0)

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

#----- Animation -----#

def julia_animation_ex():

	start_time = time.time()

	c_vals = np.array([complex(i,0.75) for i in np.linspace(0.05, 3.0, 100)])
	s=julia_series(c_vals, [-1,1], [-0.75,1.25], magnetic_2, args=2, maxiter=100)
	animate(s, gamma=0.9, cmap=plt.cm.gist_ncar, filename='julia_animation_ex')

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

#----- random walk -----#
# this will take awhile, the code is not fast

def random_walk_ex():

	start_time = time.time()

	rw = random_walk(5,5,300)
	rw.lattice_walk_2D(10000000, start=[0,0], static=True, udlr_weights=[1,1,1,1], diag_weights=[1,1,1,1])
	rw.lattice_walk_2D_image(color='temporal_distance', cmap=plt.cm.gist_gray, filename='random_walk_ex')

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

#----- buddhabrot -----#
# this will take awhile

def buddhabrot_ex():

	xB = np.array([-1.50, -0.75])
	yB = np.array([-0.70, 0.70])
	
	start_time = time.time()

	cvals = compute_cvals(1000000, xB, yB, power, args=2, width=3, height=4, dpi=100)
	bud0 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=100, width=3, height=4, dpi=100)
	bud1 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=1000, width=3, height=4, dpi=100)
	bud2 = buddhabrot(xB, yB, cvals, power, args=2, horizon=1.0E6, maxiter=10000, width=3, height=4, dpi=100)
	
	nebula_image(bud0, bud1, bud2, gammas=(0.35,0.4,0.4), filename='buddhabrot_ex')

	print('calculation took %s seconds ' % np.round((time.time() - start_time), 3))

