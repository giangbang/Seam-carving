import energy
import seamCarving
import matplotlib.pyplot as plt
import numpy as np

RED = [255,0, 0]

def showImgs(imgs):
	n = len(imgs)
	for i, img in enumerate(imgs):
		plt.subplot(1,n,i+1)
		if len(img.shape) == 2:
			plt.imshow(img, cmap='gray')
		else:
			plt.imshow(img)
		plt.xticks([]), plt.yticks([])
	
	plt.show()
	
def drawSeam(img: np.ndarray, seam: np.ndarray)->np.ndarray:
	_, w = img.shape[:2]
	mask = np.eye(w, dtype=np.bool)[seam]
	img[mask, ...] = RED
	return img
	
def drawSeamMask(img: np.ndarray, \
				seamMask: np.ndarray)->np.ndarray:
	assert len(img.shape) == 3 and img.shape[-1] == 3
	h, w = img.shape[:2]
	img[seamMask,...] = RED
	return img