import numpy as np
from . import plotanimation, mathutils, skeletonmap
from .anim import *
from os.path import basename as getfilename
from os.path import join as pathjoin
import time

def ReadFile(path,
             surfaceinfo=None,
             skipmotion=False):
    """
    Read BVH file and returns an anim.Animation object. The anim.Animation holds the information about the animation file and contain reference to anim.Joint objects. The global position of each joint is **not** computed when calling bvh.ReadFile().

    :param surface.Surface surfaceinfo: Attach the surface information of the corresponding character to the anim.Animation object. The surface information is only required for computing the egocentric coordinates. Default set to None. (*)

    :param bool skipmotion: If set to True, the motion of the BVH file will not be read (skip everything after "Frame Time"), only the skeleton specification is stored. Default set to False.
    
    :returns: Animation object containing the information from the bvh file.
    :rtype: anim.Animation
    """
    animation = GetBVHDataFromFile(path, skipmotion=skipmotion)
    animation.surfaceinfo = surfaceinfo

    return animation

def WriteBVH(animation, 
             path, 
             name = '_export', 
             frametime = 0.00833333, 
             writeTranslation = True,
             refTPose = True):
    """
    Create a bvh file with the motion contained in the anim.Animation object using the information contained in joint.rotation and joint.translation.

    :param anim.Animation animation: anim.Animation containing the motion

    :param str path: Full path to save the file

    :param str name: Filename without the '.bvh' extension.

    :param float frametime: 1/(frame per second), the time duration of each frame. Default set to 120 fps.

    :param bool writeTranslation: If set to True (default), translations (local positions) are written for every joint. Each joint will have six channels, with the first three representing X, Y, and Z translations. If set to False, only the translation (global position) for the root joint will be written.

    :param bool refTPose: If set to True, the first frame of the BVH file will be the input TPose reference (default). Note that the TPose reference must be set manually for each joint and stored in joint.tposetrans and joint.tposerot as in retarget.MotionRetargeting().
    """
    if name:
        path = pathjoin(path, name)
    endsiteflag = False
    depth = 0
    with open(str.format("%s.bvh" % path), "w") as file:
        file.write('HIERARCHY\n')
        for section in ['header', 'content']:
            if section == 'header':
                for joint in animation.getlistofjoints():
                    if joint==animation.root:
                        file.write(str.format('ROOT %s\n' % joint.name))
                        file.write('{\n')
                        file.write(str.format('\tOFFSET %.5f %.5f %.5f\n' % (joint.offset[0],joint.offset[1],joint.offset[2])))
                        if joint.order == 'XYZ':
                            file.write(str.format("\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"))
                        elif joint.order == 'ZXY':
                            file.write(str.format("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"))
                        elif joint.order == 'ZYX':
                            file.write(str.format("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"))
                    else:
                        if endsiteflag:
                            endsiteflag = False
                            next_depth = joint.getDepth()
                            while depth >= next_depth:
                                file.write('%s}\n' % ((depth)*'\t'))
                                depth = depth-1
                        depth = joint.getDepth()
                        file.write(str.format('%sJOINT %s\n' % (depth*'\t', joint.name)))
                        file.write('%s{\n' % (depth*'\t'))
                        file.write(str.format('%sOFFSET %.5f %.5f %.5f\n' % ((depth+1)*'\t',joint.offset[0],joint.offset[1],joint.offset[2])))
                        if writeTranslation:
                            aux_string = str.format("%sCHANNELS 6 Xposition Yposition Zposition " % ((depth+1)*"\t"))
                        else:
                            aux_string = str.format("%sCHANNELS 3 " % ((depth+1)*"\t"))
                        if joint.order == 'XYZ':
                            file.write(aux_string + "Xrotation Yrotation Zrotation\n")
                        elif joint.order == 'ZXY':
                            file.write(aux_string + "Zrotation Xrotation Yrotation\n")
                        elif joint.order == 'ZYX':
                            file.write(aux_string + "Zrotation Yrotation Xrotation\n")
                        else:
                            print('Order not implemented')
                            raise NotImplementedError
                        if len(joint.endsite) > 0:
                            endsiteflag = True
                            file.write(str.format('%sEnd Site\n' % ((depth+1)*'\t')))
                            file.write('%s{\n' % ((depth+1)*'\t'))
                            file.write(str.format('%sOFFSET %.5f %.5f %.5f\n' % ((depth+2)*'\t',joint.endsite[0],joint.endsite[1],joint.endsite[2])))
                            file.write('%s}\n' % ((depth+1)*'\t'))
            elif section == 'content':
                while depth > 0:
                    file.write('%s}\n' % ((depth)*'\t'))
                    depth = depth-1
                file.write('}\n')
                file.write(str.format('MOTION\n'))
                totalframes = animation.frames
                #Check if the TPose should be added and check if at least the root has this information
                #It assumes that if the root has this information, so do the other joints

                if refTPose == True and len(animation.root.tposerot)>0 and len(animation.root.tposetrans)>0:
                     totalframes = totalframes + 1
                else:
                    refTPose = False
                file.write(str.format('Frames: %i\n' % totalframes))
                file.write(str.format('Frame Time: %.5f\n' % frametime))

                # Write the reference TPose line
                if refTPose == True:
                    line = []
                    for joint in animation.getlistofjoints():
                        if writeTranslation or joint==animation.root:
                            line = line + [joint.tposetrans[0], joint.tposetrans[1], joint.tposetrans[2]]
                        if joint.order=='XYZ':
                            line = line + [joint.tposerot[0], joint.tposerot[1], joint.tposerot[2]]
                        elif joint.order=='ZXY':
                            line = line + [joint.tposerot[2], joint.tposerot[0], joint.tposerot[1]]
                        elif joint.order=='ZYX':
                            line = line + [joint.tposerot[2], joint.tposerot[1], joint.tposerot[0]]
                    string = " ".join(str.format("%.2f"%number) for number in line)
                    file.write(string+'\n')

                #Write the rest of the file
                for frame in range(animation.frames):
                    line = []
                    for joint in animation.getlistofjoints():
                        if writeTranslation or joint==animation.root:
                            line = line + [joint.translation[frame,0], joint.translation[frame,1], joint.translation[frame,2]]
                        if joint.order=='XYZ':
                            line = line + [joint.rotation[frame,0], joint.rotation[frame,1], joint.rotation[frame,2]]
                        elif joint.order=='ZXY':
                            line = line + [joint.rotation[frame,2], joint.rotation[frame,0], joint.rotation[frame,1]]
                        elif joint.order=='ZYX':
                            line = line + [joint.rotation[frame,2], joint.rotation[frame,1], joint.rotation[frame,0]]
                    string = " ".join(str.format("%.2f"%number) for number in line)
                    file.write(string+'\n')
    print('File Saved: %s' % (path+'.bvh'))


def GetBVHDataFromFile(path, 
                       skipmotion=False):
    """
    Auxiliary function to bvh.ReadFile(), it is not intended to be used by the user. It is the parser of the bvh file.

    :param str path: Full path to the bvh file

    :param bool skipmotion: If set to True, skip everything after "Frame Time", only the skeleton specification is stored. Default set to False.

    :returns: Animation object containing the information from the bvh file.
    :rtype: anim.Animation
    """
    # TODO: Account for BVH files without translation
    frame = 0
    bvhfile = None
    with open(path) as file:
        flagEndSite = False
        flagMotionDataBegin = False
        for line in file:
            if not flagMotionDataBegin:
                # Starts here (first joint)
                if line.find("ROOT") >= 0:
                    # Creates root joint
                    root = Joints(name=line[5:-1])
                    lastJoint = root

                    # Create the Animation object of this file
                    filename = getfilename(path)[:-4]
                    bvhfile = Animation(filename, root)

                # Every other joint goes through here
                # Identention should be tabular or with pairs of spaces
                elif line.find("JOINT") >= 0:
                    depth = line.count('\t')
                    if depth == 0:
                        depth = line[:line.find('JOINT')].count(' ')/2
                    parent = root.getLastDepth(depth-1)
                    # Creates joint
                    lastJoint = Joints(name=line[line.find("JOINT")+6:-1],
                                       depth=depth, parent=parent)

                elif line.find("End Site") >= 0:
                    flagEndSite = True

                elif (line.find("OFFSET") >= 0) and (not flagEndSite):
                    lastJoint.addOffset(np.asarray(line[line.find("OFFSET")+7:-1].split(' '),float))
                elif (line.find("OFFSET") >= 0) and (flagEndSite):
                    lastJoint.addEndSite(np.asarray(line[line.find("OFFSET")+7:-1].split(' '),float))
                    flagEndSite = False

                elif (line.find("CHANNELS")) >= 0:
                    # TODO: Improve this part
                    aux = line.replace('\t','').replace('\n','').split(" ")
                    aux = [item for item in aux if item]
                    lastJoint.n_channels = int(aux[1])
                    lastJoint.channels = {key: value for key, value in zip(aux[2:], np.arange(lastJoint.n_channels))}
                    if lastJoint.n_channels != 3 and lastJoint.n_channels != 6:
                        print("Number of channels must be 3 or 6")
                        raise NotImplementedError
                    X, Y, Z = lastJoint.channels["Xrotation"], lastJoint.channels["Yrotation"], lastJoint.channels["Zrotation"]
                    if Z < X and X < Y: lastJoint.order = "ZXY"
                    else:
                        if X < Y and Y < Z: 
                            lastJoint.order = "XYZ"
                        elif Z < Y and Y < X:
                            lastJoint.order = "ZYX"
                        else:
                            print("Invalid Channels order.")
                            raise NotImplementedError
                        print("WARNING: Channels order %s for Joint %s is not fully implemented yet." % (lastJoint.order, lastJoint.name))
                        print("bvhsdk only fully supports ZXY order. Use it with caution.")

                elif (line.find("Frames")) >= 0:
                    bvhfile.frames = int(line[8:])
                    for joint in bvhfile.getlistofjoints():
                        joint.translation = np.empty(shape=(bvhfile.frames, 3))
                        joint.rotation = np.empty(shape=(bvhfile.frames, 3))
                elif (line.find("Frame Time")) >= 0:
                    bvhfile.frametime = float(line[12:])
                    flagMotionDataBegin = True
                else:
                    # Lines only with { and } pass through here
                    pass
            elif flagMotionDataBegin and not skipmotion:
                if frame >= bvhfile.frames:
                    print("WARNING: Number of frames in file is different from declared in file.")
                    break
                line = [float(item) for item in line.replace('\n', '').split(' ') if item]
                i = 0
                for joint in bvhfile.getlistofjoints():
                    values = line[i:i+len(joint.channels)]
                    joint.rotation[frame] = np.array( [values[joint.channels['Xrotation']], values[joint.channels['Yrotation']], values[joint.channels['Zrotation']]] )
                    if len(joint.channels) == 6:
                        joint.translation[frame] = np.array( [values[joint.channels['Xposition']], values[joint.channels['Yposition']], values[joint.channels['Zposition']]] )
                    i+=len(joint.channels)
                frame += 1


    return bvhfile

def GetPositions(joint,
                 frame = 0,
                 parentTransform=[],
                 surfaceinfo=None,
                 calibrating=None):
    """
    Recursevely compute the global position of each joint for the given frame. The global position is stored in joint.position. This function is not used anymore and could be removed or moved to the anim.Animation class in the future.

    :param anim.Joint joint: Joint to calculate the global position

    :param int frame: Frame to calculate the global position, default set to 0.

    :param list parentTransform: Parameter not intended for user input, used in recursion. List of 4x4 matrices, each matrix is the local transformation of the parent joint.

    :param surface.Surface surfaceinfo: Legacy parameter, not used anymore. Should be removed in the future.

    :param bool calibrating: Legacy parameter, not used anymore. Should be removed in the future.
    """
    rot = joint.rotation[frame]
    transform = joint.getLocalTransform(frame)

    if len(parentTransform) == 0:
        # If it is the root joint, the global position is the same as the local position
        positionfinal = np.dot(transform, [0,0,0,1])
        orientation = np.asarray([rot[0], rot[1], rot[2]])
    else:
        # If it is not the root joint, the global position is the joint's local transformation matrix multiplied by the parent joint's global transformation matrx
        transform = np.dot(parentTransform,transform)
        positionfinal = np.dot(transform,[0,0,0,1])
        orientation = joint.parent.orientation[frame,:] + np.asarray([rot[0], rot[1], rot[2]])

    if not calibrating:
        joint.addPosition(np.asarray(positionfinal[:-1]), frame)
        joint.addOrientation(orientation, frame)

    # If joint have an endsite (it is an end effector)
    if len(joint.endsite)>0:
        ee_transform = mathutils.matrixTranslation(joint.endsite[0], joint.endsite[1], joint.endsite[2])
        ee_transform = np.dot(transform,ee_transform)
        endsitepos = np.dot(ee_transform,[0,0,0,1])
        if not calibrating:
            # Save endsite position in the current joint
            joint.addEndSitePosition(np.asarray(endsitepos[:-1]), frame)


    parentTransform = np.copy(transform)
    for child in joint.children:
        GetPositions(child, frame, parentTransform, surfaceinfo, calibrating)
    parentTransform=[]