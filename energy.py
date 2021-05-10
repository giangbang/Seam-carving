import numpy as np
import cv2
import matplotlib.pyplot as plt
from visualize import *

SOBEL_KERNEL_SIZE = 1
MASK_ENERGY = int(1e6)

def gradientEnergyGray(img: np.ndarray)->np.ndarray:
	sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,\
				ksize=SOBEL_KERNEL_SIZE)
	sobely = cv2.Sobel(img,cv2.CV_32F,0,1,\
				ksize=SOBEL_KERNEL_SIZE)

	return np.square(sobelx) + np.square(sobely)
	
def LaplaceGray(img: np.ndarray)->np.ndarray:
	res = np.array(cv2.Laplacian(img,cv2.CV_32F))
	return res*res

def gradientEnergyLaplace(img: np.ndarray, \
				keep_mask=None, remove_mask=None)->np.ndarray:
	if len(img.shape) == 2:
		result = np.sqrt(LaplaceGray(img))
	else:
		assert len(img.shape) == 3 and img.shape[-1] == 3
		grad = [LaplaceGray(img[...,i]) for i in range(3)]
		grad = np.array(grad)
		result = np.sqrt(grad.sum(axis=0))
		
	if keep_mask is not None:
		result[keep_mask] += MASK_ENERGY
	elif remove_mask is not None:
		result[remove_mask] -= MASK_ENERGY
		
	return result.astype(np.int32)
	
def gradientEnergySobel(img: np.ndarray, \
				keep_mask=None, remove_mask=None)->np.ndarray:
	if len(img.shape) == 2:
		result = np.sqrt(gradientEnergyGray(img))
	else:
		assert len(img.shape) == 3 and img.shape[-1] == 3
		grad = [gradientEnergyGray(img[...,i]) for i in range(3)]
		grad = np.array(grad)
		result = np.sqrt(grad.sum(axis=0))
		
	if keep_mask is not None:
		result[keep_mask] += MASK_ENERGY
	elif remove_mask is not None:
		result[remove_mask] -= MASK_ENERGY
		
	return result.astype(np.int32)
	
if __name__ == '__main__':
	img = cv2.imread('images/cat.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.array(img)
	grad = gradientEnergyLaplace(img)
	print(grad.max(), grad.min())
	showImgs([img, grad])
