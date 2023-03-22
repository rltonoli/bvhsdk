# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:12:57 2018

@author: Rodolfo Luis Tonoli
"""

import os
import numpy as np
from . import bvh
from . import anim
from . import surface
from . import mathutils
from . import skeletonmap
import time
from . import egocentriccoord
from . import ik


def PostureInitialization(tgtMap, srcMap, heightRatio, frame, getpositions=False, headAlign=True, spineAlign=False, handAlign=True):
    """
    Copy the rotation from the mocap joints to the corresponding avatar joint.

    ani_ava: Avatar animation
    ani_mocap: Mocap animation
    """
    ground_normal = np.array([0, 1, 0])
    # Adjust roots/hips height
    # Eray Molla Eq 1
    srcPosHips = srcMap.hips.getPosition(frame)
    srcGroundHips = np.asarray([srcPosHips[0], 0, srcPosHips[2]])
    tgtGroundHips = srcGroundHips*heightRatio
    srcHHips = np.dot(srcMap.hips.getPosition(frame), ground_normal)
    tgtHHips = srcHHips*heightRatio
    tgtMap.root.translation[frame] = [0, tgtHHips, 0] + tgtGroundHips

    if frame == 0:
        mocapbones = []
        avabones = []
        if spineAlign:
            mocapbones.append([srcMap.hips, srcMap.spine3])
            avabones.append([tgtMap.hips, tgtMap.spine3])
        if headAlign:
            mocapbones.append([srcMap.neck, srcMap.head])
            avabones.append([tgtMap.neck, tgtMap.head])
        mocapbones = mocapbones + [[srcMap.rarm, srcMap.rforearm],[srcMap.larm, srcMap.lforearm],[srcMap.rforearm, srcMap.rhand],[srcMap.lforearm, srcMap.lhand],[srcMap.rupleg, srcMap.rlowleg],[srcMap.lupleg, srcMap.llowleg],[srcMap.rlowleg, srcMap.rfoot],[srcMap.llowleg, srcMap.lfoot]]
        avabones = avabones + [[tgtMap.rarm, tgtMap.rforearm],[tgtMap.larm, tgtMap.lforearm],[tgtMap.rforearm, tgtMap.rhand],[tgtMap.lforearm, tgtMap.lhand],[tgtMap.rupleg, tgtMap.rlowleg],[tgtMap.lupleg, tgtMap.llowleg],[tgtMap.rlowleg, tgtMap.rfoot],[tgtMap.llowleg, tgtMap.lfoot]]
        if handAlign and srcMap.lhandmiddle and srcMap.rhandmiddle and tgtMap.lhandmiddle and tgtMap.rhandmiddle:
            mocapbones = mocapbones + [[srcMap.rhand, srcMap.rhandmiddle],[srcMap.lhand, srcMap.lhandmiddle]]
            avabones = avabones + [[tgtMap.rhand, tgtMap.rhandmiddle],[tgtMap.lhand, tgtMap.lhandmiddle]]
        for mocapbone, avabone in zip(mocapbones,avabones):
            #Get source and target global transform and rotation matrices from the start of the bone
            p0 = mocapbone[0].getPosition(0)
            p1 = mocapbone[1].getPosition(0)
            srcDirection = mathutils.unitVector(p1-p0)
            #Get source and target global transform and rotation matrices from the end of the bone
            p0 = avabone[0].getPosition(0)
            p1 = avabone[1].getPosition(0)
            tgtDirection = mathutils.unitVector(p1-p0)
            #Align vectors
            alignMat = mathutils.alignVectors(tgtDirection, srcDirection)
            #Get new global rotation matrix
            tgtGlbTransformMat = avabone[0].getGlobalTransform(frame)
            tgtGlbRotationMat = mathutils.shape4ToShape3(tgtGlbTransformMat)
            tgtNewGblRotationMat = np.dot(alignMat,tgtGlbRotationMat)
            #Get new local rotation matrix
            if not avabone[0] == tgtMap.root: #Does not have a parent, transform is already local
                tgtParentGblRotationMat = mathutils.shape4ToShape3(avabone[0].parent.getGlobalTransform(frame))
                tgtNewLclRotationMat = np.dot(tgtParentGblRotationMat.T, tgtNewGblRotationMat)
            else:
                tgtNewLclRotationMat = tgtNewGblRotationMat[:]
            #Get new local rotation euler angles
            tgtNewLclRotationEuler, warning = mathutils.eulerFromMatrix(tgtNewLclRotationMat, avabone[0].order)
            avabone[0].setLocalRotation(frame,tgtNewLclRotationEuler)

    else:
        for joint_ava, joint_mocap in zip(tgtMap.getJointsNoRootHips(), srcMap.getJointsNoRootHips()):
            if joint_ava is not None and joint_mocap is not None:
                previousframe = frame-1 if frame!= 0 else 0
                #Get source and target global transform and rotation matrices
                #Even if frame == 0 the matrices need to be recalculated
                srcGlbTransformMat = joint_mocap.getGlobalTransform(frame)
                srcGlbRotationMat = mathutils.shape4ToShape3(srcGlbTransformMat)
                tgtGlbTransformMat = joint_ava.getGlobalTransform(previousframe)
                tgtGlbRotationMat = mathutils.shape4ToShape3(tgtGlbTransformMat)
                #Get previous source global transform and rotation matrices
                srcPreviousGlbTransformMat = joint_mocap.getGlobalTransform(previousframe)
                srcPreviousGlbRotationMat = mathutils.shape4ToShape3(srcPreviousGlbTransformMat)
                #Get the transform of the source from the previous frame to the present frame
                transform = np.dot(srcGlbRotationMat, srcPreviousGlbRotationMat.T)
                #Apply transform
                tgtNewGblRotationMat = np.dot(transform, tgtGlbRotationMat)
                #Get new local rotation matrix
                tgtParentGblRotationMat = mathutils.shape4ToShape3(joint_ava.parent.getGlobalTransform(frame))
                tgtNewLclRotationMat = np.dot(tgtParentGblRotationMat.T, tgtNewGblRotationMat)
                #Get new local rotation euler angles
                tgtNewLclRotationEuler, warning = mathutils.eulerFromMatrix(tgtNewLclRotationMat, joint_ava.order)
                joint_ava.setLocalRotation(frame,tgtNewLclRotationEuler[:])


def checkName(name, currentpath=None):
    """
    Return True if the file in the path/name provided exists inside current path

    :type name: string
    :param name: Local path/name of the back file
    """
    if not currentpath:
        currentpath = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(currentpath, name)
    return os.path.isfile(fullpath)


def MotionRetargeting(sourceAnimationPath, sourceSurfacePath, targetSkeletonPath, targetSurfacePath, customSkeletomMap=None, computeEgo=True, computeIK=True, adjustOrientation=True, saveFile=True, saveInitAndFull=True, out_path=None):
    retargettime = time.time()

    # Surface Calibration #####################################################
    start = time.time()
    srcSurface = surface.GetMoCapSurfaceFromTXT(sourceSurfacePath, highpolymesh=False)
    print('Surface from file done. %s seconds.' % (time.time()-start))

    # Read mocap bvh file #####################################################
    source_filename = os.path.basename(sourceAnimationPath)
    start = time.time()
    # TODO: I think that ReadFile does not need surfaceinfo anymore. Test it and remove it.
    srcAnimation = bvh.ReadFile(sourceAnimationPath, surfaceinfo=srcSurface)
    #srcMap = srcAnimation.getskeletonmap()
    srcMap = skeletonmap.SkeletonMap(srcAnimation)
    print('MoCap BVH read done. %s seconds.' % (time.time()-start))

    # Read TPose bvh file #####################################################
    start = time.time()
    tgtAnimation = bvh.ReadFile(targetSkeletonPath)
    # Get skeleton map
    # TODO: Acho que não é necessário pegar o mapa aqui, remover e testar
    #tgtMap = tgtAnimation.getskeletonmap(mapfile=customSkeletomMap)
    tgtMap = skeletonmap.SkeletonMap(tgtAnimation, mapfile=customSkeletomMap)

    # Get the avatar surface data
    tgtSurface = surface.GetAvatarSurfaceFromCSV(targetSurfacePath, highpolymesh=False)
    # Scale the avatar surface data accordingly to the TPose bvh data
    tgtSurface.NormalizeSurfaceData(hipsjoint=tgtMap.hips)
    surface.GetAvatarSurfaceLocalTransform(tgtAnimation, tgtSurface)
    print('Avatar BVH read done. %s seconds.' % (time.time()-start))

    # Initialize pose #########################################################
    tgtAnimation.frames = srcAnimation.frames
    tgtAnimation.frametime = srcAnimation.frametime
    # Save the reference TPose
    for joint in tgtAnimation.root:
        joint.tposerot = joint.rotation[0]
        joint.tposetrans = joint.translation[0]
    tgtAnimation.expandFrames(srcMap.root.translation.shape[0])

    # Get the Height of the root in the base position (Frame = 0)
    # If the source animation (mocap) is not in the TPose, it will fail
    ground_normal = np.array([0, 1, 0])
    srcHHips = np.dot(srcMap.hips.getPosition(0), ground_normal)
    tgtHHips = np.dot(tgtMap.hips.getPosition(0), ground_normal)
    heightRatio = tgtHHips/srcHHips



    #JacRHand = ik.SimpleJacobian(tgtAnimation, tgtAnimation.getskeletonmap().rhand, depth=5)
    #JacLHand = ik.SimpleJacobian(tgtAnimation, tgtAnimation.getskeletonmap().lhand, depth=5)
    JacRHand = ik.SimpleJacobian(tgtAnimation, tgtAnimation.skeletonmap.rhand, depth=5)
    JacLHand = ik.SimpleJacobian(tgtAnimation, tgtAnimation.skeletonmap.lhand, depth=5)

    iklogRHand = []
    iklogLHand = []

    print('Starting Motion Retargeting')
    start = time.time()
    for frame in range(srcAnimation.frames):

        # Perform Simple Retargeting ##################################################
        PostureInitialization(tgtMap, srcMap, heightRatio, frame, getpositions=False, headAlign=True, spineAlign=False)

        # Calculate egocentric coordinates ############################################
        egocoord = egocentriccoord.GetEgocentricCoordinatesTargets(srcAnimation, srcSurface, tgtAnimation, tgtSurface, frame)

        # TODO: I can't remember why I'm doing this. Probably to get some info for my thesis. Remove it later
        MotionRetargeting.importance.append(egocoord[0].importance)

        # Applies Inverse Kinematics ###################################################
        logR = JacRHand.jacobianTranspose(frame=frame, target=egocoord[0].getTarget(frame))
        logL = JacLHand.jacobianTranspose(frame=frame, target=egocoord[1].getTarget(frame))
        iklogRHand.append(logR)
        iklogLHand.append(logL)

        # Adjust Limb Extremities ##################################################
        egocentriccoord.AdjustExtremityOrientation(tgtAnimation, tgtSurface, egocoord, srcAnimation, frame)
        # I was working on some different adjustment but it did not work. Remove it later:
        # egocentriccoord.AdjustExtremityOrientation2(tgtAnimation, srcAnimation)

        if np.mod(frame+1, 100) == 0:
            print('%i frames done. %s seconds.' % (int((frame+1)/100)*100, time.time()-start))
            start = time.time()

    if not saveFile:
        # TODO: This method used to have different returns (thus the 'None'). Clean it up later
        return tgtAnimation, tgtSurface, srcAnimation, srcSurface, None, egocoord

    # Save File ###################################################
    if not out_path:
        currentpath = os.path.dirname(os.path.realpath(__file__))
    else:
        currentpath = out_path
    output_filename = source_filename[:-4] + '_retarget'
    of_aux = output_filename
    i = 0
    while checkName(output_filename+'.bvh', currentpath):
        i = i + 1
        output_filename = of_aux + str(i)

    bvh.WriteBVH(tgtAnimation, currentpath, output_filename, refTPose=True)

    # if saveInitAndFull:
    #     output_filename = source_filename[:-4] + '_initialRetarget'
    #     of_aux = output_filename
    #     i = 0
    #     while checkName(output_filename+'.bvh'):
    #         i = i + 1
    #         output_filename = of_aux + str(i)
    #     bvh.WriteBVH(tgtAnimation_onlyInitial, currentpath, output_filename,refTPose=True)
    # return tgtAnimation, tgtSurface, srcAnimation, srcSurface, tgtAnimation_onlyInitial, egocoord

    print('Done! Total time: %s seconds.' % (time.time()-retargettime))
    return tgtAnimation, tgtSurface, srcAnimation, srcSurface, None, egocoord


def SimpleMotionRetargeting(srcAnimationPath,
                            tgtAnimationPath,
                            outputPath,
                            srcSkeletonMap=None,
                            tgtSkeletonMap=None,
                            frameStop=False,
                            trackProgress=False,
                            forceFaceZ=False):
    """
    Perform a simple motion retargeting. The bones of the target skeleton is aligned with the bones of the source animation

     Parameters
    ----------
    srcAnimationPath : string
        Path to the source animation (.bvh)
    tgtAnimationPath : string, optional
        Path to the source animation (.bvh). The animation file may have one or several frames, but only the first take will be used.

    Returns
    -------
    tgtAnimation : Animation object
        Retargeted animation.
    """
    srcAnimation = bvh.ReadFile(srcAnimationPath)
    tgtAnimation = bvh.ReadFile(tgtAnimationPath)
    #srcMap = srcAnimation.getskeletonmap(mapfile=srcSkeletonMap)
    #tgtMap = tgtAnimation.getskeletonmap(mapfile=tgtSkeletonMap)
    srcMap = skeletonmap.SkeletonMap(srcAnimation, mapfile=srcSkeletonMap)
    tgtMap = skeletonmap.SkeletonMap(tgtAnimation, mapfile=tgtSkeletonMap)

    if trackProgress:
        print('Retargeting start.')

    if frameStop:
        stop = np.min([srcAnimation.frames, frameStop])
    else:
        stop = srcAnimation.frames

    # TODO: NÃO É COM O LOCAL,MAS SIM COM O GLOBAL EM Y QUE TENHO QUE GIRAR
    if forceFaceZ:

        initrot = srcAnimation.root.getLocalRotation(frame=0)

        for frame in range(stop):
            srcAnimation.root.RotateGlobal([0,180,0], frame)

        ############## DEBUG: #########
        srcAnimation.frames = stop
        output_name = os.path.basename(tgtAnimationPath)[:-4] + '_ForceZTest'
        i = 0
        of_aux = output_name
        while checkName(output_name+'.bvh', outputPath):
            i = i + 1
            output_name = of_aux + str(i)
        bvh.WriteBVH(srcAnimation, path=outputPath, name=output_name, refTPose=True)
        return None

    tgtAnimation.expandFrames(stop)
    # Get the Height of the root in the base position (Frame = 0)
    # If the source animation (mocap) is not in the TPose, it will fail
    ground_normal = np.array([0, 1, 0])
    srcHHips = np.dot(srcMap.hips.getPosition(0), ground_normal)
    tgtHHips = np.dot(tgtMap.hips.getPosition(0), ground_normal)
    heightRatio = tgtHHips/srcHHips

    for frame in range(stop):
        # Perform Simple Retargeting ##################################################
        if trackProgress and np.mod(frame, 100) == 0:
            print('%i / %i done.' % (frame, stop))
        PostureInitialization(tgtMap, srcMap, heightRatio, frame, getpositions=False, headAlign=True, spineAlign=False)

    #srcMap = srcAnimation.getskeletonmap()
    #tgtMap = tgtAnimation.getskeletonmap()
    #srcMap = skeletonmap.SkeletonMap(srcAnimation)
    #tgtMap = skeletonmap.SkeletonMap(tgtAnimation)

    output_name = os.path.basename(tgtAnimationPath)[:-4] + '_SimpleRetarget'
    i = 0
    of_aux = output_name
    while checkName(output_name+'.bvh', outputPath):
        i = i + 1
        output_name = of_aux + str(i)
    bvh.WriteBVH(tgtAnimation, path=outputPath, name=output_name, refTPose=False)

    return tgtAnimation
