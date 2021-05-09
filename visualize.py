import matplotlib.pyplot as plt
import numpy as np

RED = [255, 0, 0]

def showImgs(imgs):
	"""
	show list of images
	"""
	n = len(imgs)
	for i, img in enumerate(imgs):
		plt.subplot(1,n,i+1)
		if len(img.shape) == 2:
			plt.imshow(img, cmap='gray')
		else:
			plt.imshow(img)
		plt.xticks([]), plt.yticks([])
	f = plt.gcf()
	f.set_figheight(6)
	f.set_figwidth(14)
	plt.show()
	
def drawSeam(img: np.ndarray, seam: np.ndarray)->np.ndarray:
	'''
	Draw seam as a red path onto the image, given seam coordinates
	'''
	_, w = img.shape[:2]
	mask = np.eye(w, dtype=np.bool)[seam]
	img[mask, ...] = RED
	return img
	
def drawSeamMask(img: np.ndarray, \
				seamMask: np.ndarray)->np.ndarray:
	"""
	Draw seam as red paths onto the image, given seam masks
	"""
	assert len(img.shape) == 3 and img.shape[-1] == 3
	h, w = img.shape[:2]
	img[seamMask,...] = RED
	return img