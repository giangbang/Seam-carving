import numpy as np
from energy import *
from visualize import *
	
def findSeamSlow(energy: np.ndarray, k: int) -> np.ndarray:
	"""
	Compute the seam coordinates of the image
	"""
	assert len(energy.shape) == 2, 'Dimensions of `energy` must be two'
	h, w = energy.shape
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
	
	seam = np.empty(h, dtype=np.uint32)
	index = np.argmin(cost)
	for i in range(h-1, -1, -1):
		seam[i] = index
		index = edgeTo[i, index]
	return seam
	

	
def removeSeamGray(gray: np.ndarray, seam: np.ndarray)->np.ndarray:
	"""
	
	"""
	h, w = gray.shape
	new_gray = np.empty((h, w-1),dtype=gray.dtype)
	for i in range(h):
		new_gray[i, :seam[i]] = gray[i, :seam[i]]
		new_gray[i, seam[i]:] = gray[i, seam[i]+1:]
	return new_gray
	
def removeSeamSlow(img: np.ndarray, seam: np.ndarray)->np.ndarray:
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
				k: int, energyFunc=gradientEnergySobel)->np.ndarray:
	h, w = img.shape[:2]
	seamMask = np.zeros((h, w), dtype=np.bool)
	indxMap = np.tile(np.arange(0, w), (h, 1))
	energy = energyFunc(img)
	rows = np.arange(0, h)
	for _ in range(n):
		seam = findSeamSlow(energy, k)
		seamMask[rows, indxMap[rows, seam]] = 1
		
		img = removeSeamSlow(img, seam)
		indxMap = removeSeamSlow(indxMap, seam)
		energy = energyFunc(img)
	
	return seamMask
	
if __name__ == '__main__':
	import time
	img = cv2.imread('images/cat.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]
	k = 1
	startTime = time.time()
	
	res = [img.copy(), drawSeamMask(img.copy(),\
				seamCarve(img, 3, 1)), img]
	showImgs(res)
	
	elapsedTime = time.time() - startTime
	print('Elapsed time :{} s'.format(elapsedTime))
	