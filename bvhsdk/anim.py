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
        for joint in self.getlistofjoints():
            print(joint.name)

    def getlistofjoints(self):
        """
        Get list of joints in the animation
        """
        if not self.listofjoints:
            self.listofjoints = self.__auxgetlist(None, [])
        return self.listofjoints

    def __auxgetlist(self, joint=None, listofjoints=[], skip=[]):
        """
        Create and return list of joints in the animation

        skip: list of joint names to not include them and their children in the list.
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
        Get references points for the root
        """
        root = self.root
        self.rootNormalizedReferencePoint = [[root.position[frame][0]/root.position[0][2], root.position[frame][1]/root.position[0][2]] for frame in range(len(root.position))]
        self.rootNormalizedHeight = [root.position[frame][2]/root.position[0][2] for frame in range(len(root.position))]
        return self.rootNormalizedReferencePoint, self.rootNormalizedHeight

    def getJoint(self, jointname):
        """
        Find the joint with jointname in the animation hierarchy
        """
        return self.root.getByName(jointname)

    def __erasepositions(self):
        for joint in self.getlistofjoints():
            joint.position=[]
            joint.orientation=[]
            joint.endsiteposition=[]

    def checkExtraRoot(self):
        """
        O esqueleto da Talita exportado em bvh pelo MotionBuilder possui uma
        junta root extra chamada 'Talita' que atrapalha o algoritmo e precisa
        ser removida. Ela não foi removida durante a exportação pois as
        distâncias das juntas (tamanho dos ossos) são normalizados.
        Aqui verifico se a junta root do bvh possui um dos nomes válidos.
        """
        # TODO: Função obsoleta. O usuário tem que se certificar que não existe juntas indesejadas
        # REMOVER
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
        Work in progress
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

    def PlotPose(self, frame=0, surface=None):
        """
        Plot the pose in the frame
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

    def PlotAnimation(self, surface=None):
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
        raise Exception('This method is no longer available, please use getBones()')


    def getBones(self, frame = 0, bonesPositions=[], joint=None, include_endsite=False):
        """
        Return the bones to plot
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


    def plotPoseSurfFrame(self, surfaceinfo, frame=0):
        """
        Plot pose in the frame with surface points
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

    def expandFrames(self, frames, set_empty=False):
        """
        Expand the number of frames of the animation by coping the rotation and translation of frame zero several times.

        frames: desired amount of frames
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



    def downSample(self, target_fps):
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



    def MLPreProcess(self, skipjoints=[], root_translation=False, root_rotation=False, local_rotation=False):
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
        #if lowerbody is False:
            #list_of_joints = [joint if ['leg', 'Leg', 'Foot', 'foot', 'Toe', 'toe'] not in ]
        #np_array = numpy.empty()
        #for joint in self.getlistofjoints():
        #    if joint is self.root:


        #TODO Include translation



class Joints:
#    listofjointsnames = []
#    listofjoints = []

    def __init__(self, name, depth=0, parent=None):
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
        #variável para debbugar o retargeting
        self.frameswarning = []

        #Check when it is needed to recompute
        self.flagGlobalTransformComputed = None
        self.currentGlobalTransform = []

    def __iter__(self):
        for child in self.children:
            yield child

    def getChildren(self, index=-1):
        """
        Returns a list of references to the childrens of the joint. If index is equal or bigger
        than 0, return the index-th child

        :type index: int
        :param index: Index-th child to be returned
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

    def getJointsBelow(self, first=True):
        """
        Generator for all joints below the hierarchy
        Parameters
        ----------
        first : bool, optional
            Internal control, do not change. The default is True.

        Yields
        ------
        Joint
            joints below the hierarchy.

        """
        for child in self:
            yield child
            yield from child.getJointsBelow(first=False)

    def __reversed__(self):
        while self.parent:
            yield self.parent
            self = self.parent

    def pathFromRootToJoint(self):
        # path = list(reversed([joint for joint in reversed(self)]))
        path = ([joint for joint in reversed(self)])[::-1]
        for joint in path:
            yield joint
        yield self

    def pathFromDepthToJoint(self, depth=-1):
        """
        Generator of the path between the joint located depth nodes up the hierarchy to the joint.

        Example: Given the subtree node1, node2, node3, node4, node5 and node6. node5.pathFromDepthToJoint(2) will
        return the joints node3, node4 and node5.

        :type depth: int
        :param depth: Position above the current joint in the hierarchy
        """
        # path = list(reversed([joint for joint in reversed(self)]))
        path = ([joint for joint in reversed(self)])[::-1]
        if depth > len(path) or depth == -1:
            depth = len(path)
        path = path[-depth:]
        for joint in path:
            yield joint
        yield self

    def pathFromJointToJoint(self, parentjoint):
        """
        Returns the kinematic path between a parentjoint up in the hierarchy to the
        current joint. If they are not in the same kinematic path (same subtree), raises error

        Returns a list, not a generator!
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
        return [joint for joint in reversed(self)][-1]

    def addChild(self, item):
        """
        Called after initialization of every joint except root

        self: the parent joint to add child
        item: the child joint initialized
        """
        item.parent = self
        self.children.append(item)

    def addOffset(self, offset):
        self.offset = offset

    def addEndSite(self, endsite=None):
        self.endsite = endsite
        self.endsiteposition = []

    def __addBoneLength(self, bonelength):
        """
        The BVH file format does not inform the length of the bone directly.
        The length can be calculated using the offset of its child, but in the
        case that the joint has multiple children, the first one is used.
        """
        self.length = bonelength

    def addPosition(self, pos, frame):
        """
        In the first time, instantiate the global position variable of a joint.
        Then fill in values at the frame provided

        self: joint to fill position
        pos: position of the joint in the frame
        frame: current frame
        """
        if len(self.position)==0:
            totalframes = self.translation.shape[0]
            self.position = np.zeros([totalframes,3])
            if len(self.endsite) > 0:
                self.endsiteposition = np.zeros([totalframes,3])
        self.position[frame,:] = pos.ravel()

    def addOrientation(self, ori, frame):
        """
        In the first time, instantiate the global orientation variable of a joint.
        Then fill in values at the frame provided

        self: joint to fill orientation
        ori: orientation of the joint in the frame
        frame: current frame
        """
        if len(self.orientation)==0:
            totalframes = self.translation.shape[0]
            self.orientation = np.zeros([totalframes,3])
        self.orientation[frame,:] = ori.ravel()

    def addEndSitePosition(self, pos, frame):
        self.endsiteposition[frame,:] = pos.ravel()

    def getLocalTransform(self, frame=0):
        """
        Get joint local transform
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
        return transform

    def getLocalTransformBaseRotation(self, frame):
        #OBSOLETE
        print('Do not use this function!')
        localRotMatrix = mathutils.expandShape3ToShape4(self.getRecalcRotationMatrix(frame))
        translation = mathutils.matrixTranslation(0, self.getLength(), 0)
        localTransform = mathutils.matrixMultiply(translation, localRotMatrix)
        #return localTransform
        return None

    def __getCurrentGlobalTransform(self):
        return self.currentGlobalTransform

    def __setCurrentGlobalTransform(self, globalTransform, frame):
        self.flagGlobalTransformComputed = frame
        self.currentGlobalTransform = globalTransform

    def getGlobalTransform(self, frame=0, includeBaseRotation=False):
        """
        Get joint global transform
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

    def getLocalRotation(self, frame=0):
        return self.rotation[frame].copy()

    def getGlobalRotation(self, frame=0,includeBaseRotation=False):
        return mathutils.eulerFromMatrix(self.getGlobalTransform(frame, includeBaseRotation), self.order)

    def getLocalTranslation(self, frame=0):
        if self.n_channels == 3:
            return self.offset
        try:
            return self.translation[frame]
        except:
            return self.translation[0]

    def getGlobalTranslation(self, frame=0):
        """
        Doesn't include base rotation
        """
        transform = self.getGlobalTransform(frame)
        return np.asarray([transform[0,3],transform[1,3],transform[2,3]])

    def getPosition(self, frame=0):
        # return np.dot(self.getGlobalTransform(frame, False), [0,0,0,1])[:-1]
        return self.getGlobalTransform(frame, False)[:-1,-1]

    def setLocalRotation(self, frame, rotation):
        """
        Outdated. Please use setLocalRotation() instead

        """
        print("setLocalRotation() is outdated. Please use setLocalRotation() instead")
        raise Exception
        #use self.setLocalRotation(frame=frame, rotation=rotation)


    def setLocalRotation(self, frame, rotation):
        """
        Updates local rotation angles in the current frame and sets all global transform matrices down the hierarchy to
        not reliable.
        Parameters
        ----------
        frame : int
            Current frame.
        rotation : numpy.ndarray
            Rotation angles.

        Returns
        -------
        None.

        """
        self.flagGlobalTransformComputed = None
        for joint in self.getJointsBelow():
            joint.flagGlobalTransformComputed = None
        self.rotation[frame] = rotation

    def setGlobalRotation(self, R, frame):
        """
        Set the global rotation of the joint as the R array

         Parameters
        ----------
        R : numpy array
            Size 3 vector containing the rotation over the x-, y- and z-axis

        frame: int
            Current frame

        Returns
        -------
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

    def RotateGlobal(self, R, frame):
        """
        Apply a global rotation of R on the joint

         Parameters
        ----------
        R : numpy array
            Size 3 vector containing the rotation over the x-, y- and z-axis

        frame: int
            Current frame

        Returns
        -------
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

    def setTranslation(self, frame, translation):
        """
        Updates translation values in the current frame and sets all global transform matrices down the hierarchy to
        not reliable.
        Parameters
        ----------
        frame : int
            Current frame.
        translation : numpy.ndarray
            Rotation angles.

        Returns
        -------
        None.

        """
        self.flagGlobalTransformComputed = None
        for joint in self.getJointsBelow():
            joint.flagGlobalTransformComputed = None
        self.translation[frame] = translation

    def getEndSitePosition(self, frame=0):
        if len(self.endsite) == 0:
            print('Unable to get joint\'s EndSite at readbvh.getEndSitePosition(joint, frame) because the joint %s does not have an EndSite.' % self.name)
            return None
        transform = mathutils.matrixTranslation(self.endsite[0], self.endsite[1], self.endsite[2])
        transform = np.dot(self.getGlobalTransform(frame),transform)
        # return np.dot(transform,[0,0,0,1])[:-1]
        return np.asarray(transform[:-1,-1])

    def getByName(self, name):
        """
        Returns the joint object with the provided name

        self: first joint in the hierarchy
        name: name of the joint
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
        depth = 0
        joint = self
        if joint.parent:
            while joint.parent:
                depth = depth + 1
                joint = joint.parent
        return depth

    def getLastDepth(self, depth, jointsInDepth = []):
        """
        Returns the last joint initializated with the depth provided

        self: root of the hierarchy
        depth: hierarchy level
        jointsInDepth: list of joints at depth level
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
        Returns the length of the bone (distance between the joint to its first
        child)
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
        DON'T USE (it's not necessary)

        Returns the base rotation matrix of the bone associated with this joint.
        The base rotation matrix rotates the vector [0 length 0] to the offset
        vector. "length" is computed through self.getLength and "offset" through
        self.offset.

        Motion Capture File Formats Explained. M. Meredith S.Maddock
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

    def getRecalcRotationMatrix(self, frame):
        """
        ESSA FUNçÂO NÂO DEVE SER UTILIZADA
        Returns the recalculated local rotation matrix

        Equation 4.9 from Motion Capture File Formats Explained. M. Meredith S.Maddock
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


    def printHierarchy(self, hierarchy=[]):
        """
        Print hierarchy below self

        self: first joint in the hierarchy
        hierarchy: formatted hierarchy list
        """
        flag = False
        if len(hierarchy)==0:
            flag = True
        hierarchy.append(str.format("%s%s %s" % (' '*2*int(self.depth),self.name, self.offset)))
        #print("%s%s %s" % (' '*2*int(self.depth),self.name, self.offset))
        try:
            if len(self.endsite)>0:
                hierarchy.append("%s%s %s" % (' '*2*(int(self.depth+1)),"End Site", self.endsite))
#                print("%s%s %s" % (' '*2*(int(self.depth+1)),"End Site", self.endsite))
        except:
            pass
        for child in self.children:
            child.printHierarchy(hierarchy)
        if flag:
            return hierarchy

    #TODO: testar e verificar se é necessário
    def PoseBottomUp(self, value=np.zeros(3)):
        aux = np.asarray([float(number) for number in self.offset])
        value = value + aux
        if self.parent:
            value = self.parent.PoseBottomUp(value)
        return value
