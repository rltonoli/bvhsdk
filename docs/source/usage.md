# Usage

## Installation

git clone the bvhsdk repository from [https://github.com/rltonoli/bvhsdk](https://github.com/rltonoli/bvhsdk) or install it using pip

```python
>>> conda create -n myenv python=3.7
>>> conda activate myenv
>>> pip install bvhsdk
```

## Open BVH file

```python
>>> import bvhsdk
>>> anim = bvhsdk.ReadFile('pathtobvh.bvh')
>>> anim
<bvhsdk.anim.Animation object at 0x0000022B79B914C8>
```

## The Animation Object

The bvh.ReadFile() returns an [anim.Animation](#bvhsdk.anim.Animation) object. An Animation object contains information about the animation data stored within. In particular, it serves as a container for global information relating to the entire motion sequence. Perhaps the most important property of an Animation object is the Animation.root, that holds a reference to a Joint object representing the topmost or the root joint of the skeleton hierarchy.

```python
>>> anim.frames
480
>>> anim.frametime
0.03333
>>> anim.root
<bvhsdk.anim.Joints object at 0x0000022B79B9A248>
```


## The Joint Object

Every joint in the skeleton is an [anim.Joint](#bvhsdk.anim.Joint) object. You can iterate to every joint in the hierarchy by calling the getlistofjoints() method of an Animation object (through depth first search):

```python
>>> for joint in anim.getlistofjoints():
...   print(joint.name)
...
body_world
b_root
b_spine0
b_spine1
b_spine2
b_spine3
b_neck0
b_head
(...)
```

You can grab the reference to a Joint using its name the method getJoint() from the Animation object. The Joint object also stores its children in list. It is possible to get a child from the children list or using the getChildren() method and passing its index as input:

```python
>>> right_wrist = anim.getJoint('b_l_wrist')
>>> right_wrist.name
'b_l_wrist'
>>> [joint.name for joint in right_wrist.children]
['b_l_thumb0', 'b_l_index1', 'b_l_middle1', 'b_l_ring1', 'b_l_pinky1']
>>> child_joint = right_wrist.getChildren(3)
>>> child_joint.name
'b_l_ring1'
```

## Where to go from here

Check out other useful functions of the bvhsdk to get your project going:

> ### anim.Animation
>
> - Animation.insertFrame()
> - Animation.removeFrame()
> - Animation.getlistofjoints()
>
> ### anim.Joint
>
> - Joint.getPosition()
> - Joint.getChildren()
> - Joint.getLocalTransform()
> - Joint.getGlobalTransform()
>
> ### mathutils
>
> - alignVectors()
> - angleBetween()
> - eulerFromMatrix()
> - matrixR()
> - matrixRotation()
> - matrixTranslation()