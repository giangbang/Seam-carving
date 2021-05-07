import numpy as np
from energy import *
from visualize import *

INF = int(1e9)

def backTrack(src: int, edgeTo: np.ndarray)->np.ndarray: 
	"""
	Back-tracking shortest path from src to top
	"""
	h, _ = edgeTo.shape
	seam = np.empty(h, dtype=np.uint32)
	index = src
	for i in range(h-1, -1, -1):
		seam[i] = index
		index = edgeTo[i, index]
	return seam
	
def findSeamSlow(energy: np.ndarray, k: int) -> np.ndarray:
	"""
	Compute the seam coordinates of the image
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
	
def removeSeamGray(gray: np.ndarray, seam: np.ndarray)->np.ndarray:
	"""
	Remove seam of a grayscale image, given seam coordinates
	"""
	h, w = gray.shape
	new_gray = np.empty((h, w-1),dtype=gray.dtype)
	for i in range(h):
		new_gray[i, :seam[i]] = gray[i, :seam[i]]
		new_gray[i, seam[i]:] = gray[i, seam[i]+1:]
	return new_gray
	
def removeSeam(img: np.ndarray, seam: np.ndarray)->np.ndarray:
	"""
	Remove seam of the image
	sub-routine `removeSeamGray` for each image channel
	"""
	assert len(img.shape) <= 3 and len(seam.shape) == 1
	assert img.shape[0] == len(seam)
	
	if len(img.shape) == 2: # grayscale image
		return removeSeamGray(img, seam)
	else:
		assert img.shape[-1] == 3 # rgb img
		return np.concatenate(\
			[np.expand_dims(removeSeamGray(img[...,i], seam),-1) for i in range(3)],\
			axis=-1)
			
def seamCarve(img: np.ndarray, n: int, \
				k: int, findSeam=findSeamFast, \
				energyFunc=gradientEnergySobel)->\
				(np.ndarray, np.ndarray):
	"""
	Get the minimum n seams and carved result of the image
	"""
	h, w = img.shape[:2]
	assert n < w
	seamMask = np.zeros((h, w), dtype=np.bool)
	indxMap = np.tile(np.arange(0, w), (h, 1))
	energy = energyFunc(img)
	rows = np.arange(0, h)
	for _ in range(n):
		seam = findSeam(energy, k)
		seamMask[rows, indxMap[rows, seam]] = 1
		
		img = removeSeam(img, seam)
		indxMap = removeSeam(indxMap, seam)
		energy = energyFunc(img)

	return seamMask, img
	
def insertSeamGray(gray: np.ndarray, seamMask: np.ndarray, n: int)\
				->np.ndarray:
	assert len(gray.shape) == 2
	h, w = gray.shape
	output = np.empty((h, w+n), dtype=gray.dtype)
	for i in range(h):
		output[i] = np.insert(gray[i], \
				np.squeeze(np.argwhere(seamMask[i])), 
				gray[i, seamMask[i]])
				
	return output
	
def seamExpand(img: np.ndarray, n: int, *args, **kwargs)->\
				(np.ndarray, np.ndarray):
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
	
if __name__ == '__main__':
	import time
	img = cv2.imread('images/cat.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]
	k = 1
	startTime = time.time()
	
	mask, new_img = seamExpand(img, 80, k)
	res = [img, gradientEnergySobel(img), 
				drawSeamMask(img.copy(), mask), new_img]
	elapsedTime = time.time() - startTime
	print('Elapsed time :{} s'.format(elapsedTime))
	showImgs(res)
	