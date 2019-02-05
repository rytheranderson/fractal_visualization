import numpy as np
import random

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors

class random_walk(object):
	def __init__(self, width, height, dpi):
		self.lattice = np.zeros((int(width*dpi), int(height*dpi)), dtype=np.float64)
		self.width   = width
		self.height  = height
		self.dpi     = dpi

	def lattice_walk_2D(self, nmoves, start=[0,0], udlr_weights=[1,1,1,1], diag_weights=[1,1,1,1], static=True):

		start = np.asarray(start)
		self.start_indices = start

		weights = udlr_weights + diag_weights
		weights = [float(i)/sum(weights) for i in weights]

		moves = [
				np.array([ 0, 1]), 
				np.array([ 0,-1]),
			    np.array([-1, 0]), 
			    np.array([ 1, 0]),
				np.array([ 1, 1]), 
				np.array([ 1,-1]), 
				np.array([-1,-1]),
				np.array([-1, 1])
				]

		move_indices = range(len(moves))

		indices  = start
		naccept  = 0
		nattempt = 0

		xindices    = []
		yindices    = []
		index_value = []

		if static:
			width,height = np.shape(self.lattice)

		while naccept < nmoves:

			nattempt += 1
			move_index = np.random.choice(move_indices, p=weights)
			move = moves[move_index]
			attempt_indices = indices + move

			if static:
				
				if (attempt_indices[0]+1 > width or attempt_indices[1]+1 > height
					or attempt_indices[0] < 0 or attempt_indices[1] < 0):
					continue
				else:
					naccept += 1
					indices = attempt_indices

					self.lattice[indices[0],indices[1]] = naccept

			else:

				naccept += 1
				indices = attempt_indices

				index_value.append((indices[0],indices[1],naccept))

				xindices.append(indices[0])
				yindices.append(indices[1])

		if not static:

			minx   = min(xindices)
			miny   = min(yindices)
			width  = max(xindices) - minx
			height = max(yindices) - miny

			self.width  = width /self.dpi
			self.height = height/self.dpi
			self.lattice = np.zeros((width + 20, height + 20))

			for l in index_value:
				self.lattice[l[0] - minx + 10, l[1] - miny + 10] = l[2]
		
	def lattice_walk_2D_image(self, filename='2D_walk', color='temporal_distance', 
		image_type='png', cmap=plt.cm.jet, ticks='off', alpha=1.0):

		lattice = self.lattice
		start = self.start_indices

		filename = filename + '_' + color

		if color == 'temporal_distance':
			pass
		elif color == 'euclidean_distance':
			for i,j in np.ndindex(lattice.shape):
				if lattice[i,j] != 0:
					lattice[i,j] = np.linalg.norm(np.array([i,j]) - start)
		elif color == 'x_distance':
			for i,j in np.ndindex(lattice.shape):
				if lattice[i,j] != 0:
					lattice[i,j] = abs(i - start[0])
		elif color == 'y_distance':
			for i,j in np.ndindex(lattice.shape):
				if lattice[i,j] != 0:
					lattice[i,j] = abs(j - start[1])
		elif color == 'manhattan_distance':
			for i,j in np.ndindex(lattice.shape):
				if lattice[i,j] != 0:
					lattice[i,j] = abs(i - start[0]) + abs(j - start[1])

		lattice = lattice.T

		w,h = plt.figaspect(lattice)
		fig, ax0 = plt.subplots(figsize=(w,h), dpi=self.dpi)
		fig.subplots_adjust(0,0,1,1)
		plt.axis(ticks)
		ax0.imshow(lattice, origin='lower', cmap=cmap, alpha=alpha) 

		F = plt.gcf()
		F.set_size_inches(self.width, self.height)

		fig.savefig(filename + '.' + image_type)

pm = random_walk(10,5,72)
pm.lattice_walk_2D(100000, start=[0,0], static=True, udlr_weights=[1,1,1,1], diag_weights=[1,1,1,1])
pm.lattice_walk_2D_image(color='temporal_distance' , cmap=plt.cm.gnuplot2)

#pm.lattice_walk_2D_image(color='euclidean_distance', cmap='gnuplot2', udlr_weights=[1,1,1,1], diag_weights=[0,0,0,0])
#pm.lattice_walk_2D_image(color='manhattan_distance', cmap='gnuplot2', polar=True)

		