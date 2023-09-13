# Example

This is an example of how the bvhsdk fits in a pipeline for a deep learning project. 

## bvhsdk in deep learning project

Imagine a project such as the [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model). The proposed models receives as input motion capture data, in this case: human poses represented by either joint rotations or positions x<sup>i</sup> ∈ R<sup>J×D</sup>, where J is the number of joints and D is the dimension of the joint representation.

Let's convert our bvh dataset to the representation described.

First we are going to create a function that receives an Animation object and converts the data to the representation that we want.

Notice that the rotation and the position are being computed differently. For the rotation, we have an outer loop over the joints and a inner loop over the frames.

However, to get the positions, we'll iterate over the frames and then over the joints. This is faster for computing joints because we are leveraging one aspect of the getPosition() function.

Remember that, to get a joint's position, we need to compute the transformation matrix of every parent joint in the hierarchy. For a given frame, the bvhsdk will store the transformation matrices computed. So, consider joints of the index finger and the pinky, they share most of the joints in their kinematic tree and the transformation matrices of these joints in the hierarchy do not need to be computed again when calling getPosition() for one finger after another. However, bvhsdk only stores computed transformation matrices for a given frame, once getPosition() is called for another frame, it will erase all transformation matrices from the previous frame and begin to store new ones for the current frame.


```python
def bvh2representations(anim: bvhsdk.Animation):
    # Converts bvh to two representations: 3d rotations (local euler angles) and 3d positions (global)
    njoints = len(anim.getlistofjoints()) # Get the number of joints in the hierarchy (skeleton)
    npyrotpos = np.empty(shape=(anim.frames, 6*njoints)) # Create an array with shape number of frames by number of joints times the representation length
    for i, joint in enumerate(anim.getlistofjoints()): # For every joint in the hierarchy
        npyrotpos[:,i*6:i*6+3] = [ joint.rotation[frame] for frame in range(anim.frames) ] # Get the local rotation of every joint
        #npyrotpos[:,i*6:i*6+3] = [ joint.translation[frame] for frame in range(anim.frames) ] # REMEMBER: this is not the position, this is the translation (or local position)
        #npyrotpos[:,i*6+3:i*6+6] = [ joint.getPosition(frame) for frame in range(anim.frames) ] # DO NOT COMPUTE POSITIONS LIKE THIS

    for frame in range(anim.frames): # For every frame
        for i, joint in enumerate(anim.getlistofjoints()): # For every joint
            pos = joint.getPosition(frame) # Get the position and store it in the array
            npyrotpos[frame, i*6+3:i*6+6] = pos
    
    return npyrotpos
```

Next, we simply need to create a function to go through all files in our dataset, read, convert, and save:

```python
def process_bvh(sourcepath, savepath):
    for file in os.listdir(sourcepath):
        anim = bvhsdk.ReadFile(os.path.join(sourcepath, file))
        rotpos = bvh2representations(anim)
        np.save(os.path.join(savepath, file[:-4]+'.npy'), rotpos)
```

## Bibliography

[1] Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2022). Human motion diffusion model. arXiv preprint arXiv:2209.14916.