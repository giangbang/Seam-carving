from seamCarving import *
from energy import *
from visualize import *
import cv2
import numpy as np
import time


img = cv2.imread('images/cat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
k = 1
n = 65

# show gradient
res = [img, gradientEnergySobel(img)]
showImgs(res)

# show seam expand
startTime = time.time()
mask, new_img = seamExpand(img, n, k)
res = [drawSeamMask(img.copy(), mask), new_img]
elapsedTime = time.time() - startTime

print('Elapsed time :{} s'.format(elapsedTime))
showImgs(res)

# show seam carve
startTime = time.time()
mask, new_img = seamCarve(img, n, k)
res = [drawSeamMask(img.copy(), mask), new_img]
elapsedTime = time.time() - startTime


print('Elapsed time :{} s'.format(elapsedTime))
showImgs(res)

# show seam expand naive
# startTime = time.time()
# new_img = seamExpandNaive(img, n, k)
# res = [img, new_img]
# elapsedTime = time.time() - startTime


# print('Elapsed time :{} s'.format(elapsedTime))
# showImgs(res)

# object removal
# mask = cv2.imread('images/mask.jpg',0)/255
# mask = np.array(np.round(mask), dtype=np.bool)

# startTime = time.time()
# new_img = removeObject(img, mask)
# res = [img, new_img]
# elapsedTime = time.time() - startTime


# print('Elapsed time :{} s'.format(elapsedTime))
# showImgs(res)