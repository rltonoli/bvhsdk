import numpy as np
from . import plotanimation, mathutils, skeletonmap
from .anim import *
from os.path import basename as getfilename
from os.path import join as pathjoin
import time

def WriteBVH(animation, path, name='_export', frametime = 0.00833333, refTPose = True, writeTranslation=True):
    """
    Create a bvh file with the motion contained in the animation.

    :type animation: pyanimation.Animation
    :param animation: Animation containing the motion

    :type path: str
    :param path: Full path to save the file

    :type name: str
    :param name: Filename

    :type frametime: float
    :param frametime: 1/(Frame per second), the time duration of each frame

    :type refTPose: bool
    :param refTPose: If True, the first frame of the animation is the input TPose reference

    :type writeTranslation: bool
    :param writeTranslation: If True, write translations for every joint. If False, only write translation for the root joint
    """
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
                        else:
                            print('Order is not implemented')
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
                    string = " ".join(str.format("%.2f"%number) for number in line)
                    file.write(string+'\n')
    print('File Saved: %s' % (path+'.bvh'))


def GetBVHDataFromFile(path, skipmotion=False):
    """
    Read a bvh file.

    :type path: string or path
    :param path: Complete path to the bvh file

    :type skipmotion: bool
    :param skipmotion: Whether to read the motion of the file (False) or to
    read only the skeleton definition.

    :rtype bvhfile: Animation
    :rparam bvhfile: An Animation object containing the information from the
    bvh file.
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
                    elif X < Y and Y < Z: lastJoint.order = "XYZ"
                    else:
                        lastJoint.order("XYZ")
                        print("Invalid Channels order. XYZ chosen.")
                        raise NotImplementedError

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

    # Check channels' order
    if not skipmotion:
        for joint in bvhfile.getlistofjoints():
            if joint.order == "ZXY":
                # joint.rotation = joint.rotation[:, [2, 0, 1]]
                joint.rotation = joint.rotation[:, [1, 2, 0]]

    return bvhfile

def GetPositions(joint, frame=0, parentTransform=[], surfaceinfo=None, calibrating=None):
    # Recebe o frame e a junta root, recursivamente calcula a posição de todos
    # os filhos para esse frame
    # Salva a posição em cada frame dentro de cada instância junta

    # Caso precise recalcular as posições das juntas, os dados antigos precisam
    # ser apagados

    #TODO: Orientation não significa nada, arrumar

    rot = joint.rotation[frame]
    transform = joint.getLocalTransform(frame)

    if len(parentTransform) == 0:
        #Se for root apenas calcula a posição
        positionfinal = np.dot(transform, [0,0,0,1])
        orientation = np.asarray([rot[0], rot[1], rot[2]])
    else:
        #Nos outros casos, multiplica pela transformada da junta pai
        transform = np.dot(parentTransform,transform)
        positionfinal = np.dot(transform,[0,0,0,1])
        orientation = joint.parent.orientation[frame,:] + np.asarray([rot[0], rot[1], rot[2]])

    if not calibrating:
        joint.addPosition(np.asarray(positionfinal[:-1]), frame)
        joint.addOrientation(orientation, frame)

    #Caso a junta tenha um endsite (é um end effector)
    if len(joint.endsite)>0:
        ee_transform = mathutils.matrixTranslation(joint.endsite[0], joint.endsite[1], joint.endsite[2])
        ee_transform = np.dot(transform,ee_transform)
        endsitepos = np.dot(ee_transform,[0,0,0,1])
        if not calibrating:
            #Salva a posição do endsite da junta no frame atual
            joint.addEndSitePosition(np.asarray(endsitepos[:-1]), frame)


    parentTransform = np.copy(transform)
    for child in joint.children:
        GetPositions(child, frame, parentTransform, surfaceinfo, calibrating)
    parentTransform=[]


def ReadFile(path, surfaceinfo=None):
    """
    Read BVH file, create animation a joints instances and compute joint positions
    """
    animation = GetBVHDataFromFile(path)
    animation.surfaceinfo = surfaceinfo

    return animation
