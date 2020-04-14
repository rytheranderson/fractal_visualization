import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib.animation as animation

def stack_cmaps(cmap, Nstacks):
	
	colors = np.array(cmap(np.linspace(0, 1, 100)))	
	
	for n in range(Nstacks - 1):
		colors = np.vstack((colors, cmap(np.linspace(0, 1, 100))))

	mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

	return mymap

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

def animate(series, fps=15, bitrate=1800, cmap=plt.cm.hot, filename='f', ticks='off', gamma=0.3, vert_exag=0, ls=[315,10]):

	Writer = animation.writers['Pillow']
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

