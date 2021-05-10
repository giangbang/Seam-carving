# Seam carving
Python implementation of seam carving.

## Libraries
* OpenCV
* numpy

## Seam carving
The idea of seam carving is to remove unnoticeable pixels that blend with their surroundings.
=> Need an energy function, e.g gradient, to measure importance of pixels.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/gradient.png">

Given an energy function, the first strategy is to remove an equal number of low energy pixels from every row. However, this will destroy the image content by zig zag effect.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/optimal%20pixel%20removal.jpeg">

Another possible strategy is to remove whole columns with the lowest energy.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/optimal%20column.jpeg">


Seam carving is a resizing strategy somewhat in between the two above, it's less restrictive than column removal and preserve content better than pixel removal.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/carve.png">

## Seam expanding
To enlarge an image, one can insert new seams instead of removing them as in seam carving. Naturally, optimal seams are inserted until the desired size is reached.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/naiveExpand.png">

Unfortunately, this approach does not work as it creates a stretching artifact by choosing the same seam. Another idea is to find the first `k` seams for removal, and then duplicate them instead of removing.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/expand.png">

## Object removal
Given the mask of the object, the algorithm firstly uses seam carving to remove the mask region and expands it to the orginal size. Energy of pixels within mask region are subtracted to attract seams to travel through them.

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/object%20removal.jpeg">

## More examples
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_grad.jpeg">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat%20expand.jpeg">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat%20remove.jpeg">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_on_pav_grad.jpeg">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_on_pav_carve.jpeg">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_on_pav_expand.jpeg">

Other energy functions:

<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_grad_laplace.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_expand_laplace.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_shrink_laplace.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cats_laplace_grad.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cats_laplace_expand.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cats_laplace_shrink.png">
<img src="https://github.com/giangbang/Seam-carving/blob/master/demo/cat_op_exp.png">


## Code structure
`seamCarving.py`: all the main code are put in here.
* `findSeamSlow`: finding coordinates of one optimal seam. Using dynamic programing approach, traverse all vertices in a DAG in topological order.
* `findSeamFast`: Vectorization version of `findSeamSlow`.
* `seamCarve`: remove `n` optimal seams from image, return removed pixel mask and resulted image.
* `seamExpand`: expand `n` optimal seams from image, values of the inserted seams equal to values of pixels on the seams.
* `seamExpandNaive`: expand the optimal seam `n` times.
* `removeObject`: remove object from image, given mask of that image.
