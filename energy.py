import numpy as np
import cv2
import matplotlib.pyplot as plt

SOBEL_KERNEL_SIZE = 3

def gradientEnergyGray(img: np.ndarray)->np.ndarray:
	sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,\
				ksize=SOBEL_KERNEL_SIZE).astype(np.int32)
	sobely = cv2.Sobel(img,cv2.CV_8U,0,1,\
				ksize=SOBEL_KERNEL_SIZE).astype(np.int32)
	return np.square(sobelx) + np.square(sobely)
	
def gradientEnergySobel(img: np.ndarray)->np.ndarray:
	if len(img.shape) == 2:
		result = np.sqrt(gradientEnergyGray(img))
	else:
		assert len(img.shape) == 3 and img.shape[-1] == 3
		grad = [gradientEnergyGray(img[...,i]) for i in range(3)]
		grad = np.array(grad)
		result = np.sqrt(grad.sum(axis=0))
				
	return result.astype(np.int32)
	
if __name__ == '__main__':
	img = cv2.imread('images/cat.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	grad = gradientEnergy(img)
	
	plt.subplot(1,2,1),plt.imshow(img)
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.subplot(1,2,2),plt.imshow(grad,cmap='gray')
	plt.title('Gradient'), plt.xticks([]), plt.yticks([])
	plt.show()
