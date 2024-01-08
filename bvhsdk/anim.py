import numpy as np
from . import mathutils

class Animation:

    def __init__(self, filename, root):
        self.name = filename
        self.root = root
        self.listofjoints = []
        self.surfaceinfo = []
        self.skeletonmap = []
        self.frames = None
        self.frametime = None

    def printlist(self):
        """
        Calls Animation.gestlistofjoints() and prints the name of each joint
        """
        for joint in self.getlistofjoints():
            print(joint.name)

    def getlistofjoints(self):
        """
        Get list of joints in the animation in a Depth First Search order.

        :returns: List of joints.
        :rtype: List[anim.Joints]
        """
        if not self.listofjoints:
            self.listofjoints = self.__auxgetlist(None, [])
        return self.listofjoints

    def __auxgetlist(self, joint=None, listofjoints=[], skip=[]):
        """
        Create and return list of joints in the animation object recursively.

        :param str joint: anim.Joints object to start the search. If None, start from the root.

        :param list listofjoints: List of joints to be returned.

        :param list skip: List of joint names to not include them and their children in the list.
        
        :returns: List of joints.
        :rtype: List[anim.Joints]
        """
        if not joint:
            joint = self.root
        listofjoints.append(joint)
        for child in joint.children:
            if child.name not in skip:
                self.__auxgetlist(child, listofjoints, skip)
        if joint == self.root:
            return listofjoints

    # def getskeletonmap(self, mapfile=None):
    #     """
    #     Pega o mapeamento do esqueleto. Procura juntas como hips, spine, arm,
    #     hand, leg, etc. na instância de animação comparando os nomes das juntas
    #     da animação com os nomes correspondentes no mapeamento descrito em skeletonmap
    #     """
    #     if not self.skeletonmap:
    #         self.skeletonmap = skeletonmap.SkeletonMap(self, mapfile)
    #     return self.skeletonmap

    def RootReferences(self):
        """
        Get references points for the root. Used to calculate the egocentric coordinates (*).

        :returns: List of root joint's position projected on the XY plane normalized by the root's height in the first frame; list of the root's height normalized by the root's height in the first frame.
        :rtype: List[float, float], List[float]
        """
        root = self.root
        self.rootNormalizedReferencePoint = [[root.position[frame][0]/root.position[0][2], root.position[frame][1]/root.position[0][2]] for frame in range(len(root.position))]
        self.rootNormalizedHeight = [root.position[frame][2]/root.position[0][2] for frame in range(len(root.position))]
        return self.rootNormalizedReferencePoint, self.rootNormalizedHeight

    def getJoint(self, jointname):
        """
        Find a joint in the animation by its name.

        :param str jointname: Name of the joint to be found.

        :returns: Joint object.
        :rtype: anim.Joints
        """
        return self.root.getByName(jointname)

    def __erasepositions(self):
        """
        Erase all global position and orientation information from the joints. Note that it will note erase the local rotation and translation information, contained in the original BVH file.
        """
        for joint in self.getlistofjoints():
            joint.position=[]
            joint.orientation=[]
            joint.endsiteposition=[]

    def checkExtraRoot(self):
        """
        Legacy function, may be removed in the future. Some BVH files have an extra root joint that is not on the base of the spine, some times it is on the floor or behind the hips.
        If you application assumes that the root joint is at the base of the spine, you may want to remove this extra joints.
        Remebember to also recompute the global positions and orientations of the "new" root joint after removing the extra root joint.
        This function used the name of the desired root joint defined in skeletonmap.SkeletonMap to check it and set it as the root of the animation.
        """
        if not skeletonmap.isroot(self.root.name):
            if not self.root.children:
                print('The animation seems to have an extra root joint without a child joint.')
                return None
            else:
                if type(self.root.children)==list:
                    for child in self.root.children:
                        child.parent = None
                    self.root = self.root.children[0]
                else:
                    self.root = self.root.children

    def RecalculatePositions(self):
        """
        Legacy function, may be removed in the future. It used to recalculate the global positions and orientations of the joints based on the local rotations and translations for all frames.
        It used the deprecated function bvh.GetPositions() to do so.
        """
        self.__erasepositions()
        start = time.time()
        # if self.surfaceinfo:
        for frame in range(self.root.translation.shape[0]):
            if self.surfaceinfo:
                GetPositions(self.root, frame=frame, surfaceinfo=self.surfaceinfo)
            else:
                GetPositions(self.root, frame=frame)
            if np.mod(frame+1,100) == 0:
                print('%i frames done. %s seconds.' % (int((frame+1)/100)*100, time.time()-start))
                start = time.time()


    def checkPose(self):
        """
        Legacy function, may be removed in the future. 
        Check the first frame of the animation for a valid T Pose, that is, 
        hips facing the positive direction of the Z axis, the arms parallel to
        the X axis (left arm along positive X) and standing in the positive
        direction of Y.
        """
        root, head, lforearm, larm = None, None, None, None
        for joint in self.getlistofjoints():
            print(joint.name)
            if skeletonmap.isroot(joint.name):
                root = joint
            elif skeletonmap.ishead(joint.name):
                head = joint
            elif skeletonmap.isleftforearm(joint.name):
                lforearm = joint
            elif skeletonmap.isleftarm(joint.name):
                larm = joint
        if not (root or head or lforearm or larm):
            print('One or more joints could not be found')
            return None
        else:
            standvector = head.position[0]-root.position[0]
            print(standvector)
            larmvector = lforearm.position[0]-larm.position[0]
            print(larmvector)
            if max(standvector)!=standvector[1]:
                #if (np.abs(standvector[1]) < np.abs(standvector[0])) or (np.abs(standvector[1]) < np.abs(standvector[2]))
                #root.rotation=np.asarray([[0,0,0] for _ in range(root.rotation.shape[0])],dtype='float')
                self.RecalculatePositions()

    def PlotPose(self, 
                 frame=0, 
                 surface=None):
        """
        Legacy function, a similar version should be available in bvhsdk.plotanimation.
        """
        if not surface:
            totaljoints = len(self.getlistofjoints())
            joints = np.zeros([totaljoints,3])
            for joint,i in zip(self.getlistofjoints(),range(totaljoints)):
                joints[i] = [joint.position[frame,0],joint.position[frame,1],joint.position[frame,2]]
            joints = np.asarray(joints)
            bones = self.getBones(frame)
            plotanimation.PosePlotBones(joints, bones)
        else:

            plotanimation.PlotPoseAndSurface(self, surface, frame)
#            totaljoints = len(self.getlistofjoints())
#            joints = np.zeros([totaljoints,3])
#            for joint,i in zip(self.getlistofjoints(),range(totaljoints)):
#                joints[i] = [joint.getPosition(frame)[0],joint.getPosition(frame)[1],joint.getPosition(frame)[2]]
#            joints = np.asarray(joints)
#            bones = self.getBones(frame)
#            plotanimation.PlotPoseAndSurface(self,surface, frame)
#
#            for triangle in surface.headmesh:
#                vertices = [[vert.getPosition(self,0)[0],vert.getPosition(self,0)[1],vert.getPosition(self,0)[2]] for vert in triangle]

    def PlotAnimation(self, 
                      surface=None):
        """
        Legacy function, a similar version should be available in bvhsdk.plotanimation.
        """
        if surface:
            bones = []
            for frame in range(self.frames):
                bones.append(self.getBones(frame))
            bones = np.asarray(bones).transpose(1,2,0)
            listofpoints = []
            for point in surface.points:
                if point.pointtype == 'mesh':
                    listofpoints.append(point)
            plotanimation.AnimationSurface(self, bones, listofpoints)
        else:
            plotanimation.PlotBVH(self)

    def getBonesInFrames(self, *args):
        """
        Legacy function, it will be removed in the next release.
        """
        raise Exception('This method is no longer available, please use getBones()')


    def getBones(self, 
                 frame = 0, 
                 bonesPositions=[], 
                 joint=None, 
                 include_endsite=False):
        """
        Create a list of bones useful for plotting the skeleton in the frame provided.
        Each entry of the list contains six values, the first three are the coordinates of the parent joint and the last three are the coordinates of the child joint.
        Can be thought as a list of lines, each line is a bone starting at p0 (first three values) end ending at p1 (last three values).

        :param int frame: Frame to get the bones. Default is 0.

        :param list bonesPositions: Not intended for user input. List of bones to be returned.

        :param anim.Joints joint: Not intended for user input. Current joint being evaluated.

        :param bool include_endsite: If True, include endsites in the list of bones. Default is False.

        :returns: List of bones' begin and end.
        :rtype: List[[float, float, float, float, float, float]]
        """
        if joint == None:
            bonesPositions = []
            joint = self.root
        else:
            pp = joint.parent.getPosition(frame)
            cp = joint.getPosition(frame)
            bonesPositions.append([pp[0], pp[1], pp[2], cp[0], cp[1], cp[2]])
            if len(joint.endsite)>0 and include_endsite:
                es = joint.getEndSitePosition(frame)
                bonesPositions.append([cp[0], cp[1], cp[2], es[0], es[1], es[2]])
        for child in joint.children:
            self.getBones(frame,bonesPositions,child, include_endsite)
        if joint == self.root:
            return np.asarray(bonesPositions)


    def plotPoseSurfFrame(self, 
                          surfaceinfo,
                          frame=0):
        """
        Legacy function, a similar version should be available in bvhsdk.plotanimation.
        """
        totaljoints = len(self.getlistofjoints())
        joints = np.zeros([totaljoints,3])
        for joint,i in zip(self.getlistofjoints(),range(totaljoints)):
            joints[i] = [joint.position[frame,0],joint.position[frame,1],joint.position[frame,2]]
        joints = np.asarray(joints)
        bones = self.getBones(frame)

        #Getting surface info, testing if there is frames information or just
        #the points
        surface = []
        for point in surfaceinfo:
            if len(point.position)>0:
                surface.append(point.position[frame])
#            try:
#                if (point.position.shape[1] > 0):
#                    surface.append(point.position[frame])
#            except:
#                if (point.position.shape[0] == 3):
#                    surface.append(point.position[frame])

        plotanimation.PosePlotBonesSurface(joints, bones, np.asarray(surface))

    def expandFrames(self,
                     frames,
                     set_empty=False):
        """
        Expand the number of frames of the animation by coping the rotation and translation of frame zero several times.

        :param int frames: Total of frames after expansion.

        :param bool set_empty:  If False, stacks the rotation and translation of frame zero (default). If True, set the translation and rotation of the joints to empty arrays.
        """
        self.frames = frames
        if set_empty:
            for joint in self.getlistofjoints():
                joint.translation = np.empty(shape=(frames, 3))
                joint.rotation = np.empty(shape=(frames, 3))
        else:
            for joint in self.getlistofjoints():
                joint.translation = np.asarray([joint.translation[0] for _ in range(frames)])
                joint.rotation = np.asarray([joint.rotation[0] for _ in range(frames)])

    def insertFrame(self, 
                    index, 
                    rotation=None, 
                    translation=None):
        """
        Insert a frame in the animation at the index position

        :param int index: Index position to insert the frame. If -1, insert at the end of the animation.

        :param numpy.ndarray rotation: Rotation of the joints in the new frame. If None (default), set the rotation to zero.

        :param numpy.ndarray translation: Local translation of the joints in the new frame. If None (default), use the same value store in anim.Joints.offset.
        """
        if index == -1:
            index = self.frames
        else:
            if index > self.frames or index < 0:
                print('Index out of range. Please select an index between 0 and %i' % self.frames)
                return None
        if rotation is None:
            rotation = np.zeros(shape=(1,3))
        self.frames = self.frames + 1
        for joint in self.getlistofjoints():
            t = translation if translation is not None else joint.offset
            joint.translation = np.insert(joint.translation, index, t, axis=0)
            joint.rotation = np.insert(joint.rotation, index, rotation, axis=0)

    def removeFrame(self, index):
        """
        Remove a frame in the animation at the index position

        :param int index: Index position to remove the frame. If -1, remove the last frame.
        """
        if index == -1:
            index = self.frames
        else:
            if index > self.frames or index < 0:
                print('Index out of range. Please select an index between 0 and %i' % self.frames)
                return None
        self.frames = self.frames - 1
        for joint in self.getlistofjoints():
            joint.translation = np.delete(joint.translation, index, axis=0)
            joint.rotation = np.delete(joint.rotation, index, axis=0)



    def downSample(self, 
                   target_fps):
        """
        Downsample the animation to the target frame rate. If the target frame rate is higher than the current frame rate, it will return False and print a warning.
        Useful for reducing the size of the animation file from 120 fps to 30 fps, for example.

        :param int target_fps: Target frame rate.
        """
        current_fps = np.round(1/self.frametime)
        if target_fps == current_fps:
            print('Animation is already at the target frame rate.')
            return False
        elif target_fps > current_fps:
            print('Can\'t perform upsampling.')
            return False
        else:
            total_time = self.frames/current_fps
            target_frametime = 1/target_fps
            target_frames = int(np.floor(total_time/target_frametime))
            extra_frames = self.frames - target_frames
        step_to_remove = np.round(self.frames/extra_frames)
        indexes_to_remove = [i for i in range(1, self.frames, int(step_to_remove))]
        for joint in self.getlistofjoints():
            joint.rotation = np.delete(joint.rotation, indexes_to_remove, axis=0)
            joint.translation = np.delete(joint.translation, indexes_to_remove, axis=0)
        self.frames = joint.rotation.shape[0]
        self.frametime = target_frametime
        print(str.format('Animation downsampled to %i fps. %i frames.' % (target_fps, self.frames)))
        return True

    def doubleSample(self):
        """
        Up sample the animation to the double of the current frame rate.
        TODO: Create an upsample method that uses interpolation.
        """
        self.frames = self.frames*2
        for joint in self.getlistofjoints():
            aux = np.empty(shape=(self.frames, 3))
            joint.rotation = mathutils.multiInterp(np.arange(self.frames), np.arange(0, self.frames, 2), joint.rotation)
            joint.translation = mathutils.multiInterp(np.arange(self.frames), np.arange(0, self.frames, 2), joint.translation)
        self.frametime = self.frametime/2

    def MLPreProcess(self, 
                     skipjoints=[],
                     root_translation=False,
                     root_rotation=False,
                     local_rotation=False):
        """
        An example of how to preprocess the animation to be used in a machine learning algorithm.
        Returns two numpy arrays with the global position and local rotation of each joint in each frame. The shape of the arrays are (number of joints, 3, number of frames).

        :param list skipjoints: List of joints (and their children) to skip in the preprocessing. Default is [] and will account for every joint.

        :param bool root_translation: If True, the root joint will be forced to the origin in **every frame**. Default is False.

        :param bool root_rotation: If True, the root joint will be forced to have zero rotation in **every frame**. Default is False.

        :param bool local_rotation: If True, the local rotation of each joint will be returned in a second array. Default is False.

        :returns: Numpy array with the global position of each joint in each frame; Numpy array with the local rotation of each joint in each frame.
        :rtype: numpy.ndarray; numpy.ndarray
        """
        root_position = self.root.getGlobalTranslation(frame=0)
        list_of_joints = self.__auxgetlist(None, [], skip=skipjoints)
        dim1 = len(list_of_joints)  # Number of joints
        dim2 = 3  # Three values for position
        dim3 = self.frames  # Number of frames
        np_array = np.empty(shape=(dim1, dim2, dim3))
        if local_rotation:
            np_array_rot = np.empty(shape=(dim1, dim2, dim3))
        for frame in range(self.frames):
            # Center root
            if not root_translation:
                aux_trans = self.root.getLocalTranslation(frame)
                self.root.setTranslation(frame=frame, translation=np.array((0, 0, 0)))
            # Fix root rotation
            if not root_rotation:
                aux_rot = self.root.getLocalRotation(frame)
                self.root.setLocalRotation(frame=frame, rotation=np.array((0, 0, 0)))
            np_array[:,:,frame] = np.asarray([joint.getPosition(frame) for joint in list_of_joints])
            if local_rotation:
                np_array_rot[:,:,frame] = np.asarray([joint.getLocalRotation(frame) for joint in list_of_joints])
            self.root.setTranslation(frame=frame, translation=aux_trans)
            self.root.setLocalRotation(frame=frame, rotation=aux_rot)
        if not local_rotation:
            return np_array
        else:
            return np_array, np_array_rot

    def newRotationOrder(self, 
                         newOrder):
        """
        Change the rotation order of all joints in the animation to the newOrder provided. This method assumes that every joint has the same rotation order.

        Note: This method was only tested for changing rotation order from ZXY to ZYX. It will not work for other rotation orders.
        To add another convertion, you will need to update the mathutils.matrixR(), mathutils.eulerFromMatrix() and Joint.getLocalTransform() methods.
        You'll also need to make sure that the bvh.ReadFile() and bvh.WriteBVH() methods are updated to read and write the new rotation order.
        Additionally, that does not mean that the new order will work with the rest of the code.

        :param str newOrder: New rotation order. Can be "XYZ", "ZYX" or "ZXY".
        """
        assert self.root.order == "ZXY", "This method was only tested for changing rotation order from ZXY to ZYX. It will not work for other rotation orders."
        for joint in self.getlistofjoints():
            for frame in range(self.frames):
                transformMatrix = joint.getLocalTransform(frame)
                newrot, _ = mathutils.eulerFromMatrix(transformMatrix, newOrder)
                joint.setLocalRotation(frame, newrot)
            joint.order = newOrder


class Joints:

    def __init__(self, 
                 name, 
                 depth=0, 
                 parent=None):
        self.name = name
        self.depth = depth
        self.children = []
        self.parent = parent
        self.endsite = []
        self.translation = []
        self.rotation = []
        self.order = []
        self.n_channels = 0
        self.position = []
        self.orientation = []
        self.egocoord = []
        self.length = []
        self.baserotation = []
        self.tposerot = []
        self.tposetrans = []
        if self.parent:
            self.parent.addChild(self)
        # Used for debuggin purposes
        self.frameswarning = []

        # Check when it is needed to recompute
        self.flagGlobalTransformComputed = None
        self.currentGlobalTransform = []

    def __iter__(self):
        for child in self.children:
            yield child

    def getChildren(self,
                    index=-1):
        """
        Returns a list of references to the childrens of the joint. If index is equal or bigger
        than 0, return the index-th child

        :param int index: Index-th child to be returned

        :returns: List of childrens of the joint if index is -1, otherwise return the index-th child
        :rtype: List[anim.Joints] or anim.Joints
        """
        if not self.children:
#            print('The joint %s does not have children.' % self.name)
            return []
        else:
            if index >= 0:
                try:
                    if len(self.children) < index:
                        print('Index greater than the amount of children.')
                        return None
                    else:
                        return self.children[index]
                except:
                    print('Something went wrong in readbvh->getChildren(%s, %i)' %(self.name, index))
                    return None
            else:
                return self.children

    def getJointsBelow(self, 
                       first=True):
        """
        Generator for all joints below the hierarchy

        :param bool first: Internal control, do not change. Default is True.

        :returns: Yields joints below the hierarchy.
        :rtype: anim.Joints
        """
        for child in self:
            yield child
            yield from child.getJointsBelow(first=False)

    def __reversed__(self):
        while self.parent:
            yield self.parent
            self = self.parent

    def pathFromRootToJoint(self):
        """
        Generator of the path between the root and the joint.

        Example: Given the subtree root, node1, node2, node3, node4, node5 and node6. node5.pathFromRootToJoint() will
        yield the joints root, node1, node2, node3, node4 and node5.

        :returns: Yields joints from the root to the current joint.
        :rtype: anim.Joints
        """
        # path = list(reversed([joint for joint in reversed(self)]))
        path = ([joint for joint in reversed(self)])[::-1]
        for joint in path:
            yield joint
        yield self

    def pathFromDepthToJoint(self,
                             depth=-1):
        """
        Generator of the path between the joint located depth nodes up the hierarchy to the joint.

        Example: Given the subtree root, node1, node2, node3, node4, node5 and node6. node5.pathFromDepthToJoint(2) will
        return the joints node3, node4 and node5.

        :param int depth: Position above the current joint in the hierarchy

        :returns: Yields joints from the joint located depth nodes up the hierarchy to the current joint.
        :rtype: anim.Joints
        """
        # path = list(reversed([joint for joint in reversed(self)]))
        path = ([joint for joint in reversed(self)])[::-1]
        if depth > len(path) or depth == -1:
            depth = len(path)
        path = path[-depth:]
        for joint in path:
            yield joint
        yield self

    def pathFromJointToJoint(self,
                             parentjoint):
        """
        Returns the kinematic path between a parentjoint up in the hierarchy to the
        current joint. If they are not in the same kinematic path (same subtree), raises error. Note: Returns a list, not a generator.

        Example: Given the subtree root, node1, node2, node3, node4, node5 and node6. node5.pathFromJointToJoint(node2) will
        return the joints node2, node3, node4 and node5.

        :param anim.Joints parentjoint: Joint to start the path

        :returns: List of joints from the parentjoint to the current joint.
        :rtype: List[anim.Joints]
        """
        path = []
        found = False
        for joint in self.pathFromRootToJoint():
            if joint == parentjoint or found:
                path.append(joint)
                found = True
        if not found:
            print('Warning at mathutils.pathFromJointToJoint(): the joints are not in the same subtree.')
            return None
        return path

    def getRoot(self):
        """
        Returns the root of the hierarchy

        :returns: Root of the hierarchy
        :rtype: anim.Joints
        """
        return [joint for joint in reversed(self)][-1]

    def addChild(self,
                 item):
        """
        Called after initialization of every joint except root

        :param anim.Joints items: Joint to be added as a child
        """
        item.parent = self
        self.children.append(item)

    def addOffset(self, 
                  offset):
        """
        Add the offset of the joint. This offset is the local translation of the joint from its parent joint defined at the header of the BVH files.

        :param numpy.ndarray offset: Offset of the joint
        """
        self.offset = offset

    def addEndSite(self,
                   endsite=None):
        """
        Add the endsite of the joint. The endsite is the local translation of the endsite from the current anim.Joints object defined at the header of the BVH files.

        :param numpy.ndarray endsite: Endsite of the joint
        """
        self.endsite = endsite
        self.endsiteposition = []

    def __addBoneLength(self,
                        bonelength):
        """
        The BVH file format does not inform the length of the bone directly.
        The length can be calculated using the offset of its child. Note that, in the
        case that the joint has multiple children, the first one is used. This is a huge assumption.

        :param numpy.ndarray bonelength: Length of the bone
        """
        self.length = bonelength

    def addPosition(self,
                    pos,
                    frame):
        """
        In the first time, instantiate the global position variable of a joint.
        Then fill in values at the frame provided.
        **Note:** manually changing the (global) position **does not** update the local rotation of the joint and it **will not** change the animation if saved to a BVH file.
        The (global) position values stored in Joints.position only exists for easily accessing the position of the joint in the animation.
        If you want to change the position of the joint, you need to compute and change its local rotation.

        :param numpy.ndarray pos: Position of the joint in the frame
        :param int frame: Frame to add the position
        """
        if len(self.position)==0:
            totalframes = self.translation.shape[0]
            self.position = np.zeros([totalframes,3])
            if len(self.endsite) > 0:
                self.endsiteposition = np.zeros([totalframes,3])
        self.position[frame,:] = pos.ravel()

    def addOrientation(self,
                       ori,
                       frame):
        """
        In the first time, instantiate the global orientation variable of a joint.
        Then fill in values at the frame provided. Similar to Joints.addPosition().

        :param numpy.ndarray ori: Global orientation of an endeffector joint.
        :param int frame: Frame to add the orientation
        """
        if len(self.orientation)==0:
            totalframes = self.translation.shape[0]
            self.orientation = np.zeros([totalframes,3])
        self.orientation[frame,:] = ori.ravel()

    def addEndSitePosition(self,
                           pos,
                           frame):
        """
        Add the position of the endsite of an endeffector joint.

        :param numpy.ndarray pos: Position of the endsite of the joint in the frame
        :param int frame: Frame to add the position
        """
        self.endsiteposition[frame,:] = pos.ravel()

    def getLocalTransform(self,
                          frame=0):
        """
        Get joint 4x4 local transformation matrix at the specified frame.

        :param int frame: Frame to get the local transformation matrix.

        :returns: Local transformation matrix (4x4).
        :rtype: numpy.ndarray
        """
        trans = self.getLocalTranslation(frame)
        rot = self.getLocalRotation(frame)
        rotx = mathutils.matrixRotation(rot[0],1,0,0)
        roty = mathutils.matrixRotation(rot[1],0,1,0)
        rotz = mathutils.matrixRotation(rot[2],0,0,1)
        transform = mathutils.matrixTranslation(trans[0], trans[1], trans[2])
        if self.order == "ZXY":
            transform = np.dot(transform, rotz)
            transform = np.dot(transform, rotx)
            transform = np.dot(transform, roty)
        elif self.order == "XYZ":
            transform = np.dot(transform, rotx)
            transform = np.dot(transform, roty)
            transform = np.dot(transform, rotz)
        elif self.order == "ZYX":
            transform = np.dot(transform, rotz)
            transform = np.dot(transform, roty)
            transform = np.dot(transform, rotx)
        return transform

    def getLocalTransformBaseRotation(self,
                                      frame):
        """
        Deprecated. Please use getLocalTransform() instead. It should be removed in the next release.
        """
        print('Do not use this function!')
        localRotMatrix = mathutils.expandShape3ToShape4(self.getRecalcRotationMatrix(frame))
        translation = mathutils.matrixTranslation(0, self.getLength(), 0)
        localTransform = mathutils.matrixMultiply(translation, localRotMatrix)
        #return localTransform
        return None

    def __getCurrentGlobalTransform(self):
        """
        Auxiliary function to get the current global transformation matrix of the joint.

        :returns: Current 4x4 global transformation matrix of the joint.
        :rtype: numpy.ndarray
        """
        return self.currentGlobalTransform

    def __setCurrentGlobalTransform(self,
                                    globalTransform,
                                    frame):
        """
        Sets the current global transformation matrix of the joint. Used to avoid recomputing the global transformation matrix of the joint for the current frame.
        **Note:** This function does not update local rotation of the joint and it will not change the animation if saved to a BVH file.

        :param numpy.ndarray globalTransform: 4x4 global transformation matrix of the joint.
        :param int frame: Frame to set the global transformation matrix.
        """
        self.flagGlobalTransformComputed = frame
        self.currentGlobalTransform = globalTransform

    def getGlobalTransform(self,
                           frame=0,
                           includeBaseRotation=False):
        """
        Compute the 4x4 global transformation matrix of the joint at the specified frame.
        This is the function to be used if you want to get the global rotation or (global) position of the joint.
        If the global transformation matrix was already computed for the current frame, it will return it without recomputing.
        
        :param int frame: Frame to get the global transformation matrix.
        :param bool includeBaseRotation: Deprecated parameter, should be removed from the next release. Default is False.

        :returns: Global transformation matrix (4x4).
        :rtype: numpy.ndarray
        """
        #if the global transform of this joint was already computed for this frame return it
        if self.flagGlobalTransformComputed != frame:
            #Get the hierarchy from joint to root and reverse it
            # tree = list(reversed([joint for joint in reversed(self)]))
            # tree.append(self)
            #parentTransform = np.empty(0)
            for joint in self.pathFromRootToJoint():
                #if the global transform of this joint was already computed for this frame, move on to the next
                if joint.flagGlobalTransformComputed == frame:
                    pass
                else:
                    if includeBaseRotation:
                        transform = joint.getLocalTransformBaseRotation(frame)
                    else:
                        transform = joint.getLocalTransform(frame)
                    if joint.parent:
                        #If this joint is not the root, multiply its local transform by its parent global transform
                        transform = np.dot(joint.parent.__getCurrentGlobalTransform(),transform)
                    #parentTransform = np.copy(transform)
                    joint.__setCurrentGlobalTransform(np.copy(transform), frame)

            if np.sum(transform, axis=1)[3] > 1:
                print('getGlobalTransform: Something is wrong. Pleas check this case')
        return self.__getCurrentGlobalTransform()

    def getLocalRotation(self,
                         frame=0):
        """
        Get local rotation angles of the joint at the specified frame. Same as joint.rotation[frame].

        :param int frame: Frame to get the local rotation angles.

        :returns: Local rotation angles.
        :rtype: numpy.ndarray
        """
        return self.rotation[frame].copy()

    def getGlobalRotation(self,
                          frame=0,
                          includeBaseRotation=False):
        """
        Get global rotation angles (Euler angles) of the joint at the specified frame.
        It first computes the global transformation matrix, then extracts the euler angles from it.

        :param int frame: Frame to get the global rotation angles.
        :param bool includeBaseRotation: Deprecated parameter, should be removed from the next release. Default is False.

        :returns: Global rotation angles.
        :rtype: numpy.ndarray
        """
        return mathutils.eulerFromMatrix(self.getGlobalTransform(frame, includeBaseRotation), self.order)

    def getLocalTranslation(self,
                            frame=0):
        """
        Get local translation of the joint at the specified frame. Same as joint.translation[frame].

        :param int frame: Frame to get the local translation.

        :returns: Local rotation translation.
        :rtype: numpy.ndarray
        """
        if self.n_channels == 3:
            return self.offset
        try:
            return self.translation[frame]
        except:
            return self.translation[0]

    def getGlobalTranslation(self,
                             frame=0):
        """
        The same as Joints.getPosition().

        :returns: Global translation (position).
        :rtype: numpy.ndarray
        """
        transform = self.getGlobalTransform(frame)
        return np.asarray([transform[0,3],transform[1,3],transform[2,3]])

    def getPosition(self,
                    frame=0):
        """
        Computes the global position of the joint at the specified frame. It first computes the global transformation matrix, then extracts the position from it using the last column.

        :param int frame: Frame to get the global position.
        
        :returns: Global position.
        :rtype: numpy.ndarray
        """
        return self.getGlobalTransform(frame, False)[:-1,-1]

#    def setLocalRotation(self,
#                         frame,
#                         rotation):
#        """
#        Outdated. Please use setLocalRotation() instead
#
#        """
#        print("setLocalRotation() is outdated. Please use setLocalRotation() instead")
#        raise Exception
#        #use self.setLocalRotation(frame=frame, rotation=rotation)


    def setLocalRotation(self,
                         frame,
                         rotation):
        """
        Updates local rotation angles in the current frame and sets all global transform matrices down the hierarchy to
        not reliable. 
        That is, the global transformation matrix of the current joint and all its children will be recomputed if Joints.GetGlobalTransform() is called.

        :param int frame: Frame to update the rotation angles.
        :param numpy.ndarray rotation: Rotation angles.
        """
        self.flagGlobalTransformComputed = None
        for joint in self.getJointsBelow():
            joint.flagGlobalTransformComputed = None
        self.rotation[frame] = rotation

    def setGlobalRotation(self, R, frame):
        """
        Set the global rotation of the joint as the R array (rx, ry, rz rotations) and update the local rotation angles accordingly.

        :param numpy.ndarray R: Global rotation (rx, ry, rz).
        :param int frame: Frame to update the rotation angles.
        """
        if type(R)!=np.ndarray:
            try:
                R = np.asarray(R)
            except:
                print('Could not convert rotation vector to numpy array at Joint.setGlobalRotation()')
                return None
        if R.shape[0] != 3:
            print('Wrong rotation vector shape. Shape = %f and not 3' % R.shape[0])
            return None
        if self.parent:
            # New global rotation matrix
            matrix = mathutils.matrixR(R, self.order, shape=3)
            # Get new local rotation matrix
            parentGblRotationMat = mathutils.shape4ToShape3(self.parent.getGlobalTransform(frame))
            newLclRotationMat = np.dot(parentGblRotationMat.T, matrix)
            # Get new local rotation euler angles
            newAngle, warning = mathutils.eulerFromMatrix(newLclRotationMat, self.order)
            self.setLocalRotation(frame, newAngle[:])
        else:
            self.setLocalRotation(frame, R)

    def RotateGlobal(self, 
                     R, 
                     frame):
        """
        Rotates the joint by a rotation defined by the R array (rx, ry, rz rotations) and update the local rotation angles accordingly.
        **Note that R represents a global rotation and not a rotation relative to the parent joint.**

        :param numpy.ndarray R: Global rotation (rx, ry, rz).
        :param int frame: Frame to update the rotation angles.
        """
        if type(R)!=np.ndarray:
            try:
                R = np.asarray(R)
            except:
                print('Could not convert rotation vector to numpy array at Joint.setGlobalRotation()')
                return None
        if R.shape[0] != 3:
            print('Wrong rotation vector shape. Shape = %f and not 3' % R.shape[0])
            return None

        # New global rotation matrix
        matrix = mathutils.matrixR(R, self.order, shape=3)
        glbRotationMat = mathutils.shape4ToShape3(self.getGlobalTransform(frame))
        newGblRotationMat = np.dot(matrix, glbRotationMat)
        # Get new local rotation matrix
        if self.parent:
            parentGblRotationMat = mathutils.shape4ToShape3(self.parent.getGlobalTransform(frame))
            newLclRotationMat = np.dot(parentGblRotationMat.T, newGblRotationMat)
        else:  # root
            newLclRotationMat = newGblRotationMat
            T = self.getPosition(frame)
            trnsMatrix = mathutils.matrixTranslation(T[0], T[1], T[2])
            matrix = mathutils.matrixR(R, self.order, shape=4)
            newTrnsl = np.dot(matrix, trnsMatrix)
            self.setTranslation(frame, [newTrnsl[0, 3], newTrnsl[1, 3], newTrnsl[2, 3]])
        # Get new local rotation euler angles
        newAngle, warning = mathutils.eulerFromMatrix(newLclRotationMat, self.order)
        self.setLocalRotation(frame, newAngle[:])

    def setTranslation(self,
                       frame,
                       translation):
        """
        Updates translation values in the current frame and sets all global transform matrices down the hierarchy to
        not reliable.
        That is, the global transformation matrix of the current joint and all its children will be recomputed if Joints.GetGlobalTransform() is called.
        
        :param int frame: Frame to update the translation.
        :param numpy.ndarray translation: Translation values.
        """
        self.flagGlobalTransformComputed = None
        for joint in self.getJointsBelow():
            joint.flagGlobalTransformComputed = None
        self.translation[frame] = translation

    def getEndSitePosition(self, 
                           frame=0):
        """
        Returns the position of the EndSite of the joint at the specified frame if the joint has an EndSite (is an endeffector joint).

        :param int frame: Frame to get the EndSite position.

        :returns: EndSite position.
        :rtype: numpy.ndarray
        """
        if len(self.endsite) == 0:
            print('Unable to get joint\'s EndSite at readbvh.getEndSitePosition(joint, frame) because the joint %s does not have an EndSite.' % self.name)
            return None
        transform = mathutils.matrixTranslation(self.endsite[0], self.endsite[1], self.endsite[2])
        transform = np.dot(self.getGlobalTransform(frame),transform)
        # return np.dot(transform,[0,0,0,1])[:-1]
        return np.asarray(transform[:-1,-1])

    def getByName(self,
                  name):
        """
        Traverse the hierarchy using Depth First Search (DFS) and returns the joint object with the provided name.

        :param str name: Name of the joint to be returned.

        :returns: Joint object with the provided name.
        :rtype: anim.Joints
        """
        if self.name == name:
            return self
        for child in self.children:
            if child.name == name:
                return child
            else:
                found = child.getByName(name)
                if found:
                    return found

    def getDepth(self):
        """
        Computes the depth of the joint in the hierarchy

        :returns: Depth of the joint in the hierarchy
        :rtype: int
        """
        depth = 0
        joint = self
        if joint.parent:
            while joint.parent:
                depth = depth + 1
                joint = joint.parent
        return depth

    def getLastDepth(self,
                     depth,
                     jointsInDepth = []):
        """
        Returns the last joint initializated with the depth provided.
        This function is misplaced, it probably should be in the anim.Animation class.

        :param int depth: Depth of the joint in the hierarchy
        :param list jointsInDepth: Don't use, internal control

        :returns: Last joint initializated with the depth provided
        :rtype: anim.Joints
        """
        if depth==0:
            return self
        else:
            for child in self.children:
                if child.depth == depth:
                    jointsInDepth.append(child)
                child.getLastDepth(depth, jointsInDepth)
            return jointsInDepth[-1]

    def getLength(self):
        """
        Returns the length of the bone (distance between the joint to its first child)

        :returns: Length of the bone
        :rtype: float
        """
        if not self.length:
            if len(self.endsite)>0:
                value = np.linalg.norm(self.endsite)
            else:
                value = np.linalg.norm(self.getChildren(0).offset)
            self.__addBoneLength(value)
        return self.length

    def getBaseRotation(self):
        """
        Don't use it (it's not necessary). Deprecated function, should be removed in the next release along with everything related to baserotation.
        Function based on the description of "Motion Capture File Formats Explained." by M. Meredith S.Maddock

        Returns the base rotation matrix of the bone associated with this joint.
        The base rotation matrix rotates the vector [0 length 0] to the offset
        vector. "length" is computed through self.getLength and "offset" through
        self.offset.
        
        """
        if len(self.baserotation)==0:
            self.getLength()
            try:
                euler, warning = mathutils.eulerFromMatrix(mathutils.alignVectors(self.getChildren(0).offset,[0,1.0,0]), self.order)
            except:
                print('Endsite used in getBaseRotation(joint)')
                euler, warning = mathutils.eulerFromMatrix(mathutils.alignVectors(self.endsite,[0,1.0,0]), self.order)
            if warning:
                print('Instability found in getBaseRotation(%s)' % self.name)
            self.baserotation = euler
        return self.baserotation

    def getRecalcRotationMatrix(self,
                                frame):
        """
        Don't use it. Deprecated. Please use getLocalTransform() instead. It should be removed in the next release.
        Function based on Equation 4.9 from "Motion Capture File Formats Explained." by M. Meredith S.Maddock
        Returns the recalculated local rotation matrix.
        """
        #TODO: Eu preciso de um jeito de calcular a rotação global com a rotação base
        # pq minha rotação global atual não leva em conta a base, então pode estar
        # apontando para o lugar errado
        #TODO: Função obsoleta, testei e não deu certo, fui por outro caminho.
        #ESSA FUNçÂO NÂO DEVE SER UTILIZADA

        #Get the hierarchy from joint to root
        tree_parentjointTOroot = list([joint for joint in reversed(self)])
        stack = mathutils.matrixIdentity()

        for joint in tree_parentjointTOroot:
            #Creates base matrix
            try:
                basematrix = mathutils.alignVectors(joint.getChildren(0).offset,[0,1.0,0]).T
            except:
                print('Endsite used in getBaseRotation(joint)')
                basematrix = mathutils.alignVectors(joint.endsite,[0,1.0,0]).T
            stack = np.dot(stack, basematrix)

        matrix = np.dot(stack, mathutils.matrixR(self.getLocalRotation(frame), self.order))
        stack = stack.T
        try:
            basematrix = mathutils.alignVectors(self.getChildren(0).offset,[0,1.0,0])
        except:
            print('Endsite used in getBaseRotation(joint)')
            basematrix = mathutils.alignVectors(self.endsite,[0,1.0,0])
        stack = np.dot(stack, basematrix)
        matrix = np.dot(matrix, stack)
        return matrix


    def printHierarchy(self,
                       hierarchy=[],
                       endsite=False):
        """
        Print hierarchy of the joint.

        :param list hierarchy: Don't use, internal control
        :param bool endsite: Don't use, internal control
        """
        flag = False
        if len(hierarchy)==0:
            flag = True
        hierarchy.append(str.format("%s%s %s" % (' '*2*int(self.depth),self.name, self.offset)))
        #print("%s%s %s" % (' '*2*int(self.depth),self.name, self.offset))
        try:
            if len(self.endsite)>0 and endsite:
                hierarchy.append("%s%s %s" % (' '*2*(int(self.depth+1)),"End Site", self.endsite))
#                print("%s%s %s" % (' '*2*(int(self.depth+1)),"End Site", self.endsite))
        except:
            pass
        for child in self.children:
            child.printHierarchy(hierarchy, endsite=False)
        if flag:
            return hierarchy

    def PoseBottomUp(self, value=np.zeros(3)):
        #TODO: test and verify function
        aux = np.asarray([float(number) for number in self.offset])
        value = value + aux
        if self.parent:
            value = self.parent.PoseBottomUp(value)
        return value
