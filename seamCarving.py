import numpy as np
from energy import *
from visualize import *

INF = int(1e9)

def backTrack(src: int, edgeTo: np.ndarray)->np.ndarray: 
	"""
	Back-tracking shortest path from src to the top
	"""
	h, _ = edgeTo.shape
	seam = np.empty(h, dtype=np.uint32)
	index = src
	for i in range(h-1, -1, -1):
		seam[i] = index
		index = edgeTo[i, index]
	return seam
	
def findSeamSlow(energy: np.ndarray, k: int)->np.ndarray:
	"""
	Compute the seam coordinates of the image
	Using dynamic programing approach, traverse all vertices
	in a DAG in topological order
	"""
	assert len(energy.shape) == 2, \
		'Dimensions of `energy` must be two'
	h, w = energy.shape
	k = min(w-1, k)
	edgeTo = np.tile(np.arange(0,w,dtype=np.int32), (h, 1))
	cost = energy[0]
	for i in range(1,h):
		_cost_tmp = cost.copy()
		for j in range(w):
			for dj in range(-k, k+1):
				prev = np.clip(dj+j, 0, w-1)
				if cost[j] > _cost_tmp[prev]:
					cost[j] = _cost_tmp[prev]
					edgeTo[i, j] = prev
		cost += energy[i]
	
	return backTrack(np.argmin(cost), edgeTo)
	
def findSeamFast(energy: np.ndarray, k: int)->np.ndarray:
	"""
	Vectorization version of `findSeamSlow`
	"""
	assert len(energy.shape) == 2, \
		'Dimensions of `energy` must be two'
	h, w = energy.shape
	k = min(w-1, k)
	edgeTo = np.empty((h, w),dtype=np.int32)
	cost = energy[0]
	base = np.arange(-k, w-k, dtype=np.int32)
	
	for i in range(1,h):
		_cost = np.lib.stride_tricks.as_strided(\
			np.pad(cost, (k,k), 'constant',constant_values=INF),\
			(w, 2*k+1), (4, 4), writeable=False)
		indx = np.argmin(_cost, axis=-1).squeeze() + base
		edgeTo[i] = indx
		cost = cost[indx] + energy[i]
	
	return backTrack(np.argmin(cost), edgeTo)
	
	
def removeSeam(img: np.ndarray, seam: np.ndarray)->np.ndarray:
	"""
	Remove seam of the image
	"""
	assert len(img.shape) <= 3 and len(seam.shape) == 1
	assert img.shape[0] == len(seam)
	
	h, w = img.shape[:2]
	new_img = np.empty((h, w-1), dtype=img.dtype)
	mask = ~np.eye(w, dtype=np.bool)[seam]
	return img[mask, ...].reshape(h, w-1,-1).squeeze()
	
			
def seamCarve(img: np.ndarray, n: int, \
				k: int, findSeam=findSeamFast, \
				energyFunc=gradientEnergySobel,\
				removeMask=None, keepMask=None)\
				->(np.ndarray, np.ndarray):
	"""
	Get the minimum n seams and carved result of the image
	"""
	h, w = img.shape[:2]
	assert n < w
	seamMask = np.zeros((h, w), dtype=np.bool)
	indxMap = np.tile(np.arange(0, w), (h, 1))
	energy = energyFunc(img, keepMask, removeMask)
	rows = np.arange(0, h)
	for _ in range(n):
		seam = findSeam(energy, k)
		seamMask[rows, indxMap[rows, seam]] = 1
		
		img = removeSeam(img, seam)
		indxMap = removeSeam(indxMap, seam)
		if removeMask is not None:
			removeMask = removeSeam(removeMask, seam)
		elif keepMask is not None:
			keepMask = removeSeam(keepMask, seam)
		energy = energyFunc(img, keepMask, removeMask)

	return seamMask, img
	
def insertSeamGray(gray: np.ndarray, seamMask: np.ndarray, n: int)\
				->np.ndarray:
	"""
	Expand n seams to a grayscale image
	Sub-routine of `seamExpand`
	"""
	assert len(gray.shape) == 2
	h, w = gray.shape
	output = np.empty((h, w+n), dtype=gray.dtype)
	for i in range(h):
		output[i] = np.insert(gray[i], \
				np.squeeze(np.argwhere(seamMask[i])), 
				gray[i, seamMask[i]])
				
	return output
	
def seamExpand(img: np.ndarray, n: int, *args, **kwargs)\
				->(np.ndarray, np.ndarray):
	"""
	Expand n seams to the original image
	Newly inserted pixels get values from seams' pixel values 
	"""
	seamMask, _ = seamCarve(img, n, *args, **kwargs)
	h, w = seamMask.shape
	if len(img.shape) == 2:
		output = insertSeamGray(img, seamMask, n)
	else:
		c = img.shape[2]
		output = [np.expand_dims(insertSeamGray(
				img[... ,i],seamMask, n), axis=-1) for i in range(c)]
		output = np.concatenate(output, axis=-1)
	
	return seamMask, output		
	
def seamExpandNaive(img: np.ndarray, n: int, *args, **kwargs)\
				->np.ndarray:
	"""
	Expand n seams to the original image by 
	finding and inserting the optimum seam
	"""
	output = img
	for _ in range(n):
		_, output = seamExpand(output, 1, *args, **kwargs)
		
	return output
	
def removeObject(img: np.ndarray, mask: np.ndarray)\
				->np.ndarray:
	"""
	Remove object in image, given mask of that object
	"""
	n, h, w = 0, *mask.shape
	for i in range(h):
		n = max(n, np.sum(mask[i]))
	_, output = seamCarve(img, n, 1, removeMask=mask)
	_, output = seamExpand(output, n, 1)
	return output

if __name__ == '__main__':
	import time
	img = cv2.imread('images/cats.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]
	k = 1
	n = 70
	
	mask = cv2.imread('images/mask.jpg',0)/255
	mask = np.array(np.round(mask), dtype=np.bool)
	print(mask.shape)
	print(img.shape)
	startTime = time.time()
	new_img = removeObject(img, mask)
	res = [img, new_img]
	elapsedTime = time.time() - startTime
	
	
	# optimal pixel removal
	# startTime = time.time()
	# mask, new_img = seamCarve(img, n, k, 
		# energyFunc=gradientEnergyLaplace)
	# res = [drawSeamMask(img, mask), new_img]
	# elapsedTime = time.time() - startTime


	print('Elapsed time :{} s'.format(elapsedTime))
	showImgs(res)
	