# Image Alignment Thingy
TODO: Make a better name

The goal of this project is to automatically detect the straightness and skewedness of images.
Primarily for scanning comics, but it could be used for other things. There is a program out there
 that does this, but it is in Python, slow, and gosh dang I hate Python dependency manegment. I 
have to fix it every dang time I install it on a new computer.

## Problem Statement

Physical comics are not ever straight. Anyone who tells you otherwise is either ignorant, or lying 
to your face on purpose. The naive would just use Photoshop's ruler tool on a straight line, hit 
the "align" button, and call it a day. But it is trivially easy to spot that this is not enough. 
In addition to comics not being straight, they are also not skew. This means that the lines that 
should be 90 degrees are in fact not. The other tricky part is that they are usually not aligned 
with the physical pages themselves! So you have lost your longest edge for alignment. Rather than
guessing with Photoshop's excessively mediocre tools for fixing this problem, this program aims to 
calculate it with some mathy stuff.

## Goals

* 🚀Blazing fast🚀
* Learn some GPU programming stuff
* Make my life easier when adjusting comic pages.

## Algorithm

The primary algorthim is the Radon transform. This is used in medical imaging for CAT scans. It is 
also good at analyzing images for different rotational properties. The steps are as follows:

1. Load image in to program memory
2. Pad the image (s.t. it is centered in the resulting buffer) to meet the following requirments:
    * Square
    * Minimum of `ceiling( sqrt( h^2 + w^2 ) )` large
    * Exactly an multiple of the workgroup size (currently 16)

Square means that the uv coordinates will be the same basis.

The minimum size means that the image can rotate freely without any clipping on edges.

The exact multiple of workgroup size means that bounds checks can be elided.

3. N = max(h, w)
4. dTheta = N / 180
5. Dispatch workgroups over the entire padded input image to a depth of N - 1.

Essentially pretend the input image is a 3D buffer depth N - 1.

6. For each pixel, rotate the coordinates by `dTheta * global_id.z`

Exactly how this is done is TBD. Options:
    * Calculate N - 1 rotation matrices and bind them to an array
    * Calculate the matrix in each shader invocation. But I believe `sin()` and `cos()` are slow
    so I'd like to avoid that.
    * Use a push constant that sets the rotation matrix and the depth of each dispatch.

7. Sample the rotated coordinates
8. `AtomicAdd` the sampled data to the output buffer at index (global_id.z, global_id.y)

That's all that's designed so far. The rest needs more R&D.

## TODO

[x] Make a shader that rotates an image
[] The rest of the owl


