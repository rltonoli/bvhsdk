# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:44:27 2019

@author: Rodolfo L. Tonoli
"""

import numpy as np
from . import mathutils
import time


class EgocentricCoordinate:
    """
    Egocentric coordinates for the right and left hand (may be also extended to support feet)
    Objects of this class holds the egocentric coordinates of a joint. The objects contain the respective joint, its name,
    and a list of reference (length = frame) for the coordinates data of that joint for every frame.
    """
    egolist = []

    def __init__(self, joint, frame):
        self.joint = joint
        self.name = joint.name
        self.egolist.append(self)
        self.target = []

        self.frame = frame
        self.importance = []  # lambda
        self.refpoint = []  # x
        self.dispvector = []  # v
        self.normcoef = []  # C
        self.angle = []  # B
        self.distroot = []  # path distance to root
        self.triangle = []  # triangulo associado a essa coordenada
        self.normal = []
        self.targets = []

        self.tau = []  # debbug tau
        self.ortho = []  # debbug importance
        self.proxi = []  # debbug importance

    def getTarget(self, frame):
        return self.importance.dot(self.targets)

    @classmethod
    def clean(cls):
        cls.egolist = []


def getVectors(animation, frame):
    """
    Get vectors to calculate the kinematic path

    :type animation: pyanimation.Animation
    :param animation: Animation (skeleton) to get the distance between mapped joints
    """
    skmap = animation.skeletonmap

    lvec_fore = skmap.vecLForearm(frame)
    rvec_fore = skmap.vecRForearm(frame)

    lvec_arm = skmap.vecLArm(frame)
    rvec_arm = skmap.vecRArm(frame)

    lvec_clavicle = skmap.vecLClavicle(frame)
    rvec_clavicle = skmap.vecRClavicle(frame)

    vec_neck = skmap.vecNeck(frame)

    vec_spine = skmap.vecSpine(frame)

    lvec_femur = skmap.vecLFemur(frame)
    rvec_femur = skmap.vecRFemur(frame)

    lvec_upleg = skmap.vecLUpleg(frame)
    rvec_upleg = skmap.vecRUpleg(frame)

    lvec_lowleg = skmap.vecLLowleg(frame)
    rvec_lowleg = skmap.vecRLowleg(frame)

    return lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg


def getJointsPositions(animation, frame):
    skmap = animation.skeletonmap
    jointlist = skmap.getJointsNoRoot()
    positions = []
    for joint in jointlist:
        if joint:
            positions.append(joint.getPosition(frame))
        else:
            positions.append(None)
    # pos_hips, pos_spine, pos_spine1, pos_spine2, pos_spine3, pos_neck, pos_neck1, pos_head, pos_lshoulder,pos_larm, pos_lforearm, pos_lhand, pos_rshoulder, pos_rarm, pos_rforearm, pos_rhand, pos_lupleg, pos_llowleg, pos_lfoot, pos_rupleg, pos_rlowleg, pos_rfoot
    # print(positions)
    return positions

def getMeshPositions(animation, surface, frame):
    mesh = [[triangle[0].getPosition(animation, frame) ,triangle[1].getPosition(animation, frame),triangle[2].getPosition(animation, frame)] for triangle in surface.headmesh+surface.bodymesh]
    return mesh




def AdjustExtremityOrientation(animation, surface, ego, sourceanim, frame):
#    TODO: NOT WORKING
    #O calculo da superficie parece estar OK, então acredito que o erro esteja aqui
    lhand, rhand = animation.skeletonmap.lhand, animation.skeletonmap.rhand
    lfoot, rfoot = animation.skeletonmap.lfoot, animation.skeletonmap.rfoot
    headmesh = surface.headmesh
    bodymesh = surface.bodymesh

    start=time.time()
    #print('Adjusting extremities orientation')
    #for frame in range(animation.frames):
    vectors = getVectors(animation, frame)
    jointpositions = getJointsPositions(animation, frame)
    lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors
    # if np.mod(frame+1,100) == 0:
    #     print('%i frames done. %s seconds.' % (int((frame+1)/100)*100,time.time()-start))
    #     start=time.time()

    for joint,egoindex in zip([rhand, lhand], range(2)):
        #Get the ego coordinates of the srcAnim animation joint
        # aux_jointname = skeletonmap.getmatchingjoint(joint.name, sourceanim).name
        # ego = EgocentricCoordinate.egolist[egoindex].getCoordFrame(frame)
        ego = EgocentricCoordinate.egolist[egoindex]
        currentJointSurfaceNormal = extremityNormal(animation, joint, frame)

#            if frame==170:
#                print('Current Joint Surface Normal:')
#                print(currentJointSurfaceNormal)
#                print('Components Surface Normal:')

        newJointSurfaceNormals = []
        for i in range(len(bodymesh)+len(headmesh)):
            if i<len(headmesh):
                _, componentSurfaceNormal = mathutils.getCentroid(headmesh[i][0].getPosition(animation, frame),headmesh[i][1].getPosition(animation, frame), headmesh[i][2].getPosition(animation, frame))
            else:
                j = i-len(headmesh)
                _, componentSurfaceNormal = mathutils.getCentroid(bodymesh[j][0].getPosition(animation, frame),bodymesh[j][1].getPosition(animation, frame), bodymesh[j][2].getPosition(animation, frame))

            #Get the axis of rotation to align the component surface normal
            axis = np.cross(componentSurfaceNormal,currentJointSurfaceNormal)
            axis_norm = axis/np.linalg.norm(axis)
            #Rotate the component surface normal and get a joint surface normal regarding that component
            matrix = mathutils.matrixRotation(ego.angle[i]*180/np.pi, axis_norm[0],axis_norm[1],axis_norm[2], shape=3)
            newJointSurfaceNormals.append(np.dot(matrix, componentSurfaceNormal))

#                if frame==170:
#                    print(newJointSurfaceNormals[-1])

        # for values in DenormEgoLimb(joint, animation, surface, frame, vectors, jointpositions, ego, i+1):
        #     _, _, _, componentSurfaceNormal = values
        #     i = i+1
        #     #Get the axis of rotation to align the component surface normal
        #     axis = np.cross(componentSurfaceNormal,currentJointSurfaceNormal)
        #     axis_norm = axis/np.linalg.norm(axis)
        #     #Rotate the component surface normal and get a joint surface normal regarding that component
        #     matrix = mathutils.matrixRotation(ego.angle[i]*180/np.pi, axis_norm[0],axis_norm[1],axis_norm[2], shape=3)
        #     newJointSurfaceNormals.append(np.dot(matrix, componentSurfaceNormal))
        if joint == rfoot or joint == lfoot:
            #Handle foot contact
            componentSurfaceNormal = [0,1,0]
            #Get the axis of rotation to align the component surface normal
            axis = np.cross(componentSurfaceNormal,currentJointSurfaceNormal)
            axis_norm = axis/np.linalg.norm(axis)
            #Rotate the component surface normal and get a joint surface normal regarding that component
            matrix = mathutils.matrixRotation(ego.angle[-1]*180/np.pi, axis_norm[0],axis_norm[1],axis_norm[2], shape=3)
            newJointSurfaceNormals.append(np.dot(matrix, componentSurfaceNormal))

#            if frame == 170:
#                print('Soma:')
#                print((np.asarray(newJointSurfaceNormals)*ego.importance[:,None]).sum(axis=0))

        #Get the mean of the new joint surface  normals
        normals = np.asarray(newJointSurfaceNormals)
        importance = ego.importance[:len(normals),None]/ego.importance[:len(normals),None].sum()
        newJointSurfaceNormal = (normals*importance).sum(axis=0)
        #Get the matrix to rotate the current joint surface normal to the new one
        matrix = mathutils.alignVectors(currentJointSurfaceNormal, newJointSurfaceNormal)
        #Apply this rotation to the joint:
        #Get global rotation matrix
        glbRotationMat = mathutils.shape4ToShape3(joint.getGlobalTransform(frame))
        #Rotate joint
        newGblRotationMat = np.dot(matrix, glbRotationMat)
        #Get new local rotation matrix
        parentGblRotationMat = mathutils.shape4ToShape3(joint.parent.getGlobalTransform(frame))
        newLclRotationMat = np.dot(parentGblRotationMat.T, newGblRotationMat)
        #Get new local rotation euler angles
        newAngle, warning = mathutils.eulerFromMatrix(newLclRotationMat, joint.order)
        #joint.rotation[frame] = newAngle[:]
        joint.setLocalRotation(frame, newAngle[:])


def AdjustExtremityOrientation2(animation, sourceanim):
#    TODO: NOT WORKING
    #O calculo da superficie parece estar OK, então acredito que o erro esteja aqui
    lhand, rhand = animation.skeletonmap.lhand, animation.skeletonmap.rhand
    lfoot, rfoot = animation.skeletonmap.lfoot, animation.skeletonmap.rfoot
    srclhand, srcrhand = sourceanim.skeletonmap.lhand, sourceanim.skeletonmap.rhand
    start=time.time()
    print('Adjusting extremities orientation')
    for frame in range(animation.frames):
        if np.mod(frame+1,100) == 0:
            print('%i frames done. %s seconds.' % (int((frame+1)/100)*100,time.time()-start))
            start=time.time()
        for joint, srcjoint in zip([rhand, lhand], [srcrhand, srclhand]):
            srcNormal = extremityNormal(sourceanim, srcjoint, frame)
            currentNormal = extremityNormal(animation, joint, frame)
            matrix = mathutils.alignVectors(currentNormal, srcNormal)
            #Apply this rotation to the joint:
            #Get global rotation matrix
            glbRotationMat = mathutils.shape4ToShape3(joint.getGlobalTransform(frame))
            #Rotate joint
            newGblRotationMat = np.dot(matrix, glbRotationMat)
            #Get new local rotation matrix
            parentGblRotationMat = mathutils.shape4ToShape3(joint.parent.getGlobalTransform(frame))
            newLclRotationMat = np.dot(parentGblRotationMat.T, newGblRotationMat)
            #Get new local rotation euler angles
            newAngle, warning = mathutils.eulerFromMatrix(newLclRotationMat, joint.order)
            #joint.rotation[frame] = newAngle[:]
            joint.setLocalRotation(frame, newAngle[:])


def DenormEgoLimb(joint, animation, surface, frame, vectors, jointpositions, egocoord, index):
        """
        Denormalize egocentric coordinates for the Limbs
        """
        assert joint is not None
        assert animation is not None
        assert surface is not None
        assert frame is not None
        assert vectors is not None
        assert index is not None
        lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors
        p_hips, p_spine, p_spine1, p_spine2, p_spine3, p_neck, p_neck1, p_head, p_lshoulder,p_larm, p_lforearm, p_lhand, p_rshoulder, p_rarm, p_rforearm, p_rhand, p_lupleg, p_llowleg, p_lfoot, p_rupleg, p_rlowleg, p_rfoot = jointpositions
        if joint == animation.skeletonmap.rhand:
            #Right hand in respect to
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_arm, lvec_clavicle, rvec_clavicle, rvec_arm, rvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_clavicle, rvec_clavicle, rvec_arm, rvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT LOW LEG LIMB
            index += 1
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_upleg, rvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT UP LEG LIMB
            index += 1
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_upleg, lvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore]
            tau = 0
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal

        elif joint == animation.skeletonmap.lhand:
            #Left hand in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_arm, rvec_clavicle, lvec_clavicle, lvec_arm, lvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_clavicle, lvec_clavicle, lvec_arm, lvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT LOW LEG LIMB
            index += 1
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_upleg, rvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT UP LEG LIMB
            index += 1
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_upleg, lvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore]
            tau = 0
            for coef,vector in zip(egocoord.normcoef[index],path):
                tau += np.linalg.norm(vector)*coef
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal

        elif joint == animation.skeletonmap.rforearm:
            #Right elbow in respect to
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_arm, lvec_clavicle, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_clavicle, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT LOW LEG LIMB
            index += 1
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_upleg, rvec_femur, vec_spine, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT UP LEG LIMB
            index += 1
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_femur, vec_spine, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_upleg, lvec_femur, vec_spine, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_femur, vec_spine, rvec_clavicle, rvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal

        elif joint == animation.skeletonmap.lforearm:
            #Left elbow in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_arm, rvec_clavicle, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_clavicle, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT LOW LEG LIMB
            index += 1
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_upleg, rvec_femur, vec_spine, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT UP LEG LIMB
            index += 1
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [rvec_femur, vec_spine, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_upleg, lvec_femur, vec_spine, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = [lvec_femur, vec_spine, lvec_clavicle, lvec_arm]
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal

        elif joint == animation.skeletonmap.rfoot:
            #Right foot in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT FOREARM LIMB
            index += 1
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, rvec_femur, rvec_upleg, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_femur, rvec_femur, rvec_lowleg, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
        elif joint == animation.skeletonmap.lfoot:
            #Left foot in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT FOREARM LIMB
            index += 1
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_femur, lvec_femur, lvec_upleg, lvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
        elif joint == animation.skeletonmap.rlowleg:
            #Right knee in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT FOREARM LIMB
            index += 1
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, rvec_femur, rvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_femur, rvec_femur, rvec_lowleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
        elif joint == animation.skeletonmap.llowleg:
            #Left foot in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #RIGHT ARM LIMB
            index += 1
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT FOREARM LIMB
            index += 1
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT ARM LIMB
            index += 1
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT LOW LEG LIMB
            index += 1
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal
            #LEFT UP LEG LIMB
            index += 1
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightRight').radius
            de_refpoint, normal = mathutils.capsuleCartesian(egocoord.refpoint[index], p0, p1, r)
            path = np.asarray([- lvec_femur, lvec_femur, lvec_upleg])
            tau = (np.linalg.norm(path, axis=1)*egocoord.normcoef[index]).sum()
            de_displacement= egocoord.dispvector[index]*tau
            yield de_displacement, de_refpoint, tau, normal

def extremityNormal(animation, joint, frame):
        """
        Returns the surface normal

        Estimate the direction of a surface normal for the extrimity joints (hands and feet).
        Based on the TPose in frame = 0, the initial surface normal is computed through:
        Get the direction of the bone in the first frame (not the joint's orientation!)
        Set a rotation axis equal to the cross product of this direction and the Y-axis [0,1,0]
        The initial surface normal is the result of a 90 degrees rotation around this axis.

        With the initial surface normal computed, apply the same transforms of the joint
        in the initial surface normal, resulting in the current surface normal.
        """
        skmap = animation.skeletonmap
        try:
            initnormal = joint.initNormal
        except:
            #The joint still does not have a initial normal
            #Get the direction of the bone
            if joint == skmap.rhand:
                child = skmap.rhandmiddle
                if not child:
                    print('Right hand middle base not mapped, using bone direction = [-1,0,0]')
                    bonedirection = [-1,0,0]
            elif joint == skmap.lhand:
                child = skmap.lhandmiddle
                if not child:
                    print('Left hand middle base not mapped, using bone direction = [1,0,0]')
                    bonedirection = [1,0,0]
            elif joint == skmap.rfoot:
                child = skmap.rtoebase
                if not child:
                    print('Right toe base not mapped, using bone direction = [0,0,1]')
                    bonedirection = [0,0,1]
            elif joint == skmap.lfoot:
                child = skmap.ltoebase
                if not child:
                    print('Left toe base not mapped, using bone direction = [0,0,1]')
                    bonedirection = [0,0,1]
            else:
                raise Exception('This is not a extrimity joint.')
            if child:
                bonedirection = child.getPosition(frame=0) - joint.getPosition(frame=0)
                bonedirection = mathutils.unitVector(bonedirection)
            #Get the rotation axis
            axis = np.cross( [0,1,0], bonedirection )
            #Get rotation matrix
            matrix = mathutils.matrixRotation(90, axis[0], axis[1], axis[2], shape = 3)
            initnormal = np.dot( matrix, bonedirection )
            initnormal = mathutils.unitVector(initnormal)
            joint.initNormal = initnormal[:]
        if frame == 0:
            return initnormal
        else:
            #Get the rotation from frame zero from current frame of the joint
            glbTransformMat = joint.getGlobalTransform(frame)
            glbRotationMat = mathutils.shape4ToShape3(glbTransformMat)
            glbInitTransformMat = joint.getGlobalTransform(frame = 0)
            glbInitRotationMat = mathutils.shape4ToShape3(glbInitTransformMat)
            transform = np.dot(glbRotationMat, glbInitRotationMat.T)
            #Rotate initial surface normal
            currentnormal = np.dot( transform, initnormal )

            return currentnormal

def importanceCalc(dispvector, normal, handthick = 3.5):
        """
        Calcula a importância da contribuição desse triangulo para a posição da junta
        """
        epsilon = 0.01
        normdispvector = np.linalg.norm(dispvector)-handthick
        if normdispvector <= epsilon:
            proximity = 1/epsilon
        else:
            proximity = 1/normdispvector

        normal_unit = normal/np.linalg.norm(normal)
        dispvector_unit = dispvector/normdispvector
        orthogonality = np.clip(np.dot(normal_unit, dispvector_unit), -1.0, 1.0)
        #TODO: CHECK
        orthogonality = (orthogonality+1)/2

        #TODO: No artigo fala para substituir por cos(epsilon), mas isso
        #iria alterar o valor que estava chegando em zero para um.
        if orthogonality < epsilon:
            orthogonality = epsilon
        orthogonality = np.abs(orthogonality)

        return orthogonality*proximity, orthogonality, proximity

def importanceCalcLimb(vectors, limbname, dispvector, normal):
        """
        Compute the importance for the limbs (without the surface normal vector)
        """
        lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors
        if limbname == 'rarm':
            bone = rvec_arm
        elif limbname == 'larm':
            bone = lvec_arm
        elif limbname == 'rfore':
            bone = rvec_fore
        elif limbname == 'lfore':
            bone = lvec_fore
        elif limbname == 'rlowleg':
            bone = rvec_lowleg
        elif limbname == 'llowleg':
            bone = lvec_lowleg
        elif limbname == 'rupleg':
            bone = rvec_upleg
        elif limbname == 'lupleg':
            bone = lvec_upleg
        else:
            print('Unknown limb name')
            return None
#        dispvector_unit = dispvector/np.linalg.norm(dispvector)
        bone = bone/np.linalg.norm(bone)
        importance, orthogonality, proximity = importanceCalc(dispvector, normal)
        return importance, orthogonality, proximity

def pathnormCalc(joint, animation, mesh, frame, refpoint, vectors, jointpositions):
    """
    Calcula a normalização do caminho cinemático. Recebe a junta e sobe na
    hierarquia. Caminho cinemático utilizado: Mão - Cotovelo - Ombro -
    Espinha - Cabeça ou Quadris.
    Retorna o vetor de deslocamento normalizado e o vetor de cossenos
    """
    #TODO: Fazer o Ground
    #Por enquanto, se não for mão, não faz nada

    #Eray Molla Fig. 9
    #Get bone vectors
    lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors
    #Get pre-computed joint positions
    pos_hips, _, _, _, _, _, _, pos_head, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = jointpositions
    #Get mapped joints
    lhand, rhand, lforearm, rforearm = animation.skeletonmap.lhand, animation.skeletonmap.rhand, animation.skeletonmap.lforearm, animation.skeletonmap.rforearm
    lfoot, rfoot, llowleg, rlowleg = animation.skeletonmap.lfoot, animation.skeletonmap.rfoot, animation.skeletonmap.llowleg, animation.skeletonmap.rlowleg
    #Defines the kinematic path for each joint
    if joint == lhand:
        kinpath = np.asarray([lvec_clavicle, lvec_arm, lvec_fore])
    elif joint == rhand:
        kinpath = np.asarray([rvec_clavicle, rvec_arm, rvec_fore])
    elif joint == lforearm:
        kinpath = np.asarray([lvec_clavicle, lvec_arm])
    elif joint == rforearm:
        kinpath = np.asarray([rvec_clavicle, rvec_arm])
    elif joint == lfoot:
        kinpath = np.asarray([lvec_femur, lvec_upleg, lvec_lowleg])
    elif joint == rfoot:
        kinpath = np.asarray([rvec_femur, rvec_upleg, rvec_lowleg])
    elif joint == llowleg:
        kinpath = np.asarray([lvec_femur, lvec_upleg])
    elif joint == rlowleg:
        kinpath = np.asarray([rvec_femur, rvec_upleg])

    #Get vector displacement
    if joint == lhand or joint == rhand or joint == lforearm or joint == rforearm:
        cos = np.empty(len(kinpath)+1)
        #Upper limb
        if mesh == 'head':
            vec_displacement = -(refpoint - pos_head) + vec_neck
            vec_displacement = vec_displacement + kinpath.sum(axis = 0)
            cos[0] = mathutils.cosBetween(vec_displacement, vec_neck)
            tau = np.linalg.norm(vec_neck)*cos[0]
        elif mesh == 'body':
            vec_displacement = -(refpoint - pos_hips) + vec_spine
            vec_displacement = vec_displacement + kinpath.sum(axis = 0)
            cos[0] = mathutils.cosBetween(vec_displacement, vec_spine)
            tau = np.linalg.norm(vec_spine)*cos[0]
        else:
            raise Exception('Upper limb joints only accept meshes from the head and body.')
        #Get tau (Eray Molla Eq 5)
        for i in range(1,len(cos)):
            cos[i] = mathutils.cosBetween(vec_displacement, kinpath[i-1])
            tau = tau + np.linalg.norm(kinpath[i-1])*cos[i]
    else:
        #Lower limbs
        if mesh == 'head':
            cos = np.empty(len(kinpath)+2)
            vec_displacement = -(refpoint - pos_head) + vec_neck - vec_spine
            vec_displacement = vec_displacement + kinpath.sum(axis = 0)
            cos[0] = mathutils.cosBetween(vec_displacement, vec_neck)
            cos[1] = mathutils.cosBetween(vec_displacement, -vec_spine)
            tau = np.linalg.norm(vec_neck)*cos[0] + np.linalg.norm(-vec_spine)*cos[1]
            for i in range(2,len(cos)):
                cos[i] = mathutils.cosBetween(vec_displacement, kinpath[i-2])
                tau = tau + np.linalg.norm(kinpath[i-2])*cos[i]
        elif mesh == 'body':
            cos = np.empty(len(kinpath))
            vec_displacement = -(refpoint - pos_hips)
            vec_displacement = vec_displacement + kinpath.sum(axis = 0)
            tau = 0
            for i in range(len(cos)):
                cos[i] = mathutils.cosBetween(vec_displacement, kinpath[i])
                tau = tau + np.linalg.norm(kinpath[i])*cos[i]
        elif mesh == 'ground':
            assert joint == rfoot or joint == lfoot, 'Foot contact should only be randled with the right and left foot'
            hipsGround = np.asarray([pos_hips[0], 0, pos_hips[2]])
            hipsHeight = np.asarray([0, pos_hips[1], 0])
            vec_displacement = -(refpoint - hipsGround) + hipsHeight
            vec_displacement = vec_displacement+ kinpath.sum(axis = 0)
            cos = np.empty(len(kinpath)+1)
            cos[0] = mathutils.cosBetween(vec_displacement, hipsHeight)
            tau = 0
            for i in range(1,len(cos)):
                cos[i] = mathutils.cosBetween(vec_displacement, kinpath[i-1])
                tau = tau + np.linalg.norm(kinpath[i-1])*cos[i]

    return vec_displacement/tau, cos, tau

def pathnormCalcLimb(joint, animation, mesh, frame, vectors, jointpositions, surface):
        lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors
        p_hips, p_spine, p_spine1, p_spine2, p_spine3, p_neck, p_neck1, p_head, p_lshoulder,p_larm, p_lforearm, p_lhand, p_rshoulder, p_rarm, p_rforearm, p_rhand, p_lupleg, p_llowleg, p_lfoot, p_rupleg, p_rlowleg, p_rfoot = jointpositions
#            TODO: Fazer para cada junta para cada um dos membros
        jointPosition = joint.getPosition(frame)
        if joint == animation.skeletonmap.rhand:
            #Right hand in respect to
            #LEFT FOREARM LIMB
            p1 = p_lhand
            p0 = p_lforearm
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p1 = p0[:]
            p0 = p_larm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - lvec_clavicle, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #RIGHT LOW LEG LIMB
            p1 = p_rfoot
            p0 = p_rlowleg
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_upleg, - rvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0)  + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rlowleg', normal, cylindric, refpoint
            #RIGHT UP LEG LIMB
            p1 = p0[:]
            p0 = p_rupleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0)  + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rupleg', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p1 = p_lfoot
            p0 = p_llowleg
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_upleg, - lvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p1 = p0[:]
            p0 = p_lupleg
            r = surface.getPoint('thightLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_femur, vec_spine, rvec_clavicle, rvec_arm, rvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint

        elif joint == animation.skeletonmap.lhand:
            #Left hand in respect to
            #RIGHT FOREARM LIMB
            p1 = p_rhand
            p0 = p_rforearm
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p1 = p0[:]
            p0 = p_rarm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rarm', normal, cylindric, refpoint
            #RIGHT LOW LEG LIMB
            p1 = p_rfoot
            p0 = p_rlowleg
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_upleg, - rvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rlowleg', normal, cylindric, refpoint
            #RIGHT UP LEG LIMB
            p1 = p0[:]
            p0 = p_rupleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rupleg', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p1 = p_lfoot
            p0 = p_llowleg
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - lvec_upleg, - lvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p1 = p0[:]
            p0 = p_lupleg
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            r = surface.getPoint('thightLeft').radius
            path = np.asarray([- lvec_femur, vec_spine, lvec_clavicle, lvec_arm, lvec_fore])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.rforearm:
            #Right elbow in respect to
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, rvec_clavicle, rvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_clavicle, rvec_clavicle, rvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #RIGHT LOW LEG LIMB
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_upleg, - rvec_femur , vec_spine , rvec_clavicle , rvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rlowleg', normal, cylindric, refpoint
            #RIGHT UP LEG LIMB
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, vec_spine, rvec_clavicle, rvec_arm])
            vec_displacement = -(refpoint -p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rupleg', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_upleg, - lvec_femur, vec_spine, rvec_clavicle, rvec_arm])
            vec_displacement = -(refpoint - p0)+ path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_femur, vec_spine, rvec_clavicle, rvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.lforearm:
            #Left elbow in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rarm', normal, cylindric, refpoint
            #RIGHT LOW LEG LIMB
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_upleg, - rvec_femur, vec_spine, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rlowleg', normal, cylindric, refpoint
            #RIGHT UP LEG LIMB
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, vec_spine, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rupleg', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, vec_spine, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_femur, vec_spine, lvec_clavicle, lvec_arm])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.rfoot:
            #Right foot in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rfore', normal, cylindric, refpoint
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, rvec_femur, rvec_upleg, rvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_femur, rvec_femur, rvec_lowleg, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.lfoot:
            #Left foot in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rfore', normal, cylindric, refpoint
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #RIGHT LOW LEG LIMB
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - rvec_upleg,- rvec_femur, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #RIGHT UP LEG LIMB
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.rlowleg:
            #Right knee in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rfore', normal, cylindric, refpoint
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_clavicle, - vec_spine, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p0 = p_llowleg
            p1 = p_lfoot
            r = surface.getPoint('shinLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - lvec_upleg,- lvec_femur, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p0 = p_lupleg
            p1 = p_llowleg
            r = surface.getPoint('thightLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_femur, rvec_femur, rvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint
        elif joint == animation.skeletonmap.llowleg:
            #Left knee in respect to
            #RIGHT FOREARM LIMB
            p0 = p_rforearm
            p1 = p_rhand
            r = surface.getPoint('foreRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_arm, - rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement,cos, tau, 'rfore', normal, cylindric, refpoint
            #RIGHT ARM LIMB
            p0 = p_rarm
            p1 = p_rforearm
            r = surface.getPoint('armRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'rfore', normal, cylindric, refpoint
            #LEFT FOREARM LIMB
            p0 = p_lforearm
            p1 = p_lhand
            r = surface.getPoint('foreLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_arm, - lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lfore', normal, cylindric, refpoint
            #LEFT ARM LIMB
            p0 = p_larm
            p1 = p_lforearm
            r = surface.getPoint('armLeft').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- lvec_clavicle, - vec_spine, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'larm', normal, cylindric, refpoint
            #LEFT LOW LEG LIMB
            p0 = p_rlowleg
            p1 = p_rfoot
            r = surface.getPoint('shinRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([ - rvec_upleg,- rvec_femur, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'llowleg', normal, cylindric, refpoint
            #LEFT UP LEG LIMB
            p0 = p_rupleg
            p1 = p_rlowleg
            r = surface.getPoint('thightRight').radius
            cylindric, refpoint, normal = mathutils.capsuleCollision(jointPosition,p0,p1,r)
            path = np.asarray([- rvec_femur, lvec_femur, lvec_upleg])
            vec_displacement = -(refpoint - p0) + path.sum(axis=0)
            cos = np.asarray([mathutils.cosBetween(vec_displacement, path[i]) for i in range(len(path))])
            tau = (np.linalg.norm(path, axis = 1)*cos).sum()
            yield vec_displacement, cos, tau, 'lupleg', normal, cylindric, refpoint

def GetEgocentricCoordinatesTargets(srcAnim, surfacesrcAnim, tgtAnim, surfacetgtAnim, frame, checkLimbDistanceFlag=True):
    headmesh = surfacesrcAnim.headmesh
    bodymesh = surfacesrcAnim.bodymesh
    headmesh_tgtAnim = surfacetgtAnim.headmesh
    bodymesh_tgtAnim = surfacetgtAnim.bodymesh
    ego = None
    EgocentricCoordinate.clean()

    #Get source skeleton map
    srcAnim_skmap = srcAnim.skeletonmap
    lhand, rhand = srcAnim_skmap.lhand, srcAnim_skmap.rhand
    lforearm, rforearm = srcAnim_skmap.lforearm, srcAnim_skmap.rforearm
    larm, rarm = srcAnim_skmap.larm, srcAnim_skmap.rarm
    lupleg, rupleg = srcAnim_skmap.lupleg, srcAnim_skmap.rupleg
    llowleg, rlowleg = srcAnim_skmap.llowleg, srcAnim_skmap.rlowleg
    lfoot, rfoot = srcAnim_skmap.lfoot, srcAnim_skmap.rfoot

    #Get target skeleton map
    ava_skmap = tgtAnim.skeletonmap
    lhand_ava, rhand_ava = ava_skmap.lhand, ava_skmap.rhand
    lforearm_ava, rforearm_ava = ava_skmap.lforearm, ava_skmap.rforearm
    larm_ava, rarm_ava = ava_skmap.larm, ava_skmap.rarm
    lupleg_ava, rupleg_ava = ava_skmap.lupleg, ava_skmap.rupleg
    llowleg_ava, rlowleg_ava = ava_skmap.llowleg, ava_skmap.rlowleg
    lfoot_ava, rfoot_ava = ava_skmap.lfoot, ava_skmap.rfoot

    start=time.time()

    ground_normal = np.asarray([0,1,0])

    # EgocentricCoordinate(rhand, frame)
    # EgocentricCoordinate(lhand, frame)
    # EgocentricCoordinate(rforearm, frame)
    # EgocentricCoordinate(lforearm, frame)
    # EgocentricCoordinate(rfoot, frame)
    # EgocentricCoordinate(lfoot, frame)
    # EgocentricCoordinate(rlowleg, frame)
    # EgocentricCoordinate(llowleg, frame)

    #Para cada frame
    #for frame in range(srcAnim.frames):
    # if np.mod(frame+1,100) == 0:
    #     print('%i frames done. %s seconds.' % (int((frame+1)/100)*100,time.time()-start))
    #     start=time.time()

    vectors = getVectors(srcAnim, frame)
    jointpositions = getJointsPositions(srcAnim, frame)
    mesh = getMeshPositions(srcAnim, surfacesrcAnim, frame)

    #Para cada junta
    #for joint in [rhand, lhand, rforearm, lforearm, rfoot, lfoot, rlowleg, llowleg]:
    #for joint in [rhand, lhand]:
    start = time.time()
    for joint in [rhand, lhand, rfoot, lfoot]:
        # ego = EgocentricCoordinate.getCoord(joint.name).addCoordFrame(frame)
        ego = EgocentricCoordinate(joint, frame)
        jointPosition = joint.getPosition(frame)

        #Eray Molla Equation 3
        #Get the surface normal of extrimities joints
        if joint == rhand or joint == lhand or joint == rfoot or joint == lfoot:
            jointSurfaceNormal = extremityNormal(srcAnim, joint, frame)

        start_aux = time.time()
        #Mesh components
        for i in range(len(bodymesh)+len(headmesh)):
            if i<len(headmesh):
                #refpoint, dispvector, normal = mathutils.distFromCentroid(jointPosition, mesh[i][0], mesh[i][1], mesh[i][2])
                normal, refpoint, dispvector, refpoint_cartesian, _ = mathutils.clampedBarycentric(jointPosition, mesh[i][0], mesh[i][1], mesh[i][2])
                #dispvector_norm, normcoef, tau = pathnormCalc(joint, srcAnim, 'head', frame, refpoint, vectors, jointpositions)
                dispvector_norm, normcoef, tau = pathnormCalc(joint, srcAnim, 'head', frame, refpoint_cartesian, vectors, jointpositions)
            else:
                j = i-len(headmesh)
                #refpoint, dispvector, normal = mathutils.distFromCentroid(jointPosition, mesh[i][0], mesh[i][1], mesh[i][2])
                normal, refpoint, dispvector, refpoint_cartesian, _ = mathutils.clampedBarycentric(jointPosition, mesh[i][0], mesh[i][1], mesh[i][2])
                #dispvector_norm, normcoef, tau = pathnormCalc(joint, srcAnim, 'body', frame, refpoint, vectors, jointpositions)
                dispvector_norm, normcoef, tau = pathnormCalc(joint, srcAnim, 'body', frame, refpoint_cartesian, vectors, jointpositions)

            importance, ortho, proxi = importanceCalc(dispvector, normal)
            #Importance
            ego.ortho.append(ortho)
            ego.proxi.append(proxi)
            ego.importance.append(importance)
            #Reference point (triangle mesh)
            ego.refpoint.append(refpoint)
            #Displacement Vector (distance from refpoint to the joint position)
            ego.dispvector.append(dispvector_norm)
            #Cosines between each bone and the displacement vector Eray Molla Eq 4
            ego.normcoef.append(normcoef)
            #Normalization factor Eray Molla Eq 5
            ego.tau.append(tau)
            ego.normal.append(normal)

            #Eray Molla Equation 3
            if joint == rhand or joint == lhand or joint == rfoot or joint == lfoot:
                angle,_ = mathutils.angleBetween(normal, jointSurfaceNormal)
                ego.angle.append(angle)

        #TODO: DEBUG
        #print('    mesh: %.4f seconds.' % (time.time()-start_aux))

        start_aux = time.time()
        #Limbs components
        for values_returned in pathnormCalcLimb(joint, srcAnim, 'limb', frame, vectors, jointpositions, surfacesrcAnim):
            dispvector, normcoef, tau, limbname, normal, refpoint, refpoint_aux = values_returned
            importance, ortho, proxi = importanceCalcLimb(vectors, limbname, dispvector, normal)
            ego.ortho.append(ortho)
            ego.proxi.append(proxi)
            ego.importance.append(importance)
            ego.refpoint.append(refpoint)
            ego.dispvector.append(dispvector/tau)
            ego.normcoef.append(normcoef)
            ego.tau.append(tau)
            ego.normal.append(normal)
            #Eray Molla Equation 3
            if joint == rhand or joint == lhand or joint == rfoot or joint == lfoot:
                angle,_ = mathutils.angleBetween(normal, jointSurfaceNormal)
                ego.angle.append(angle)
        #TODO: DEBUG
        # print('    limb: %.4f seconds.' % (time.time()-start_aux))

        #Add the ground projection as a reference point
        if joint == rfoot or joint == lfoot:
            refpoint = np.asarray([jointPosition[0], 0,jointPosition[2]])
            dispvector_norm, normcoef, tau = pathnormCalc(joint, srcAnim, 'ground', frame, refpoint, vectors, jointpositions)
            importance, ortho, proxi = importanceCalc(dispvector, ground_normal)
            ego.ortho.append(ortho)
            ego.proxi.append(proxi)
            ego.importance.append(importance)
            ego.refpoint.append(refpoint)
            ego.dispvector.append(dispvector_norm)
            ego.normcoef.append(normcoef)
            ego.tau.append(tau)
            ego.normal.append(normal)
            angle,_ = mathutils.angleBetween(ground_normal, jointSurfaceNormal)
            ego.angle.append(angle)

        #distance between point p0=jointPosition and line passing through p1 and p2:
        # d = |(p0 - p1) x (p0 - p2)|/|p2-p1|
#            distance = np.linalg.norm(np.cross(jointPosition - p1,jointPosition - p2))/np.linalg.norm(p2 - p1)
#            dispvector = distance - surfacesrcAnim.getPoint('foreRight').radius

        #Normaliza a importancia
        sumimp = sum(ego.importance)
        ego.importance = np.asarray([ego.importance[element]/sumimp for element in range(len(ego.importance))])

    #TODO: DEBUG
    # print('  get: %.4f seconds.' % (time.time()-start))

    #####################################################################################
    # Desnormalizando a cada frame
    #####################################################################################


    vectors = getVectors(tgtAnim, frame)
    jointpositions = getJointsPositions(tgtAnim, frame)
    mesh = getMeshPositions(tgtAnim, surfacetgtAnim, frame)
    lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_spine, lvec_femur, rvec_femur, lvec_upleg, rvec_upleg, lvec_lowleg, rvec_lowleg = vectors

    start = time.time()
    #For each EE (each hand)
    #for joint,egoindex in zip([rhand_ava, lhand_ava, rforearm_ava, lforearm_ava, rfoot_ava, lfoot_ava, rlowleg_ava, llowleg_ava],range(6)):
    # for joint,egoindex in zip([rhand_ava, lhand_ava],range(2)):
    for egoindex,joint in enumerate([rhand_ava, lhand_ava, rfoot_ava, lfoot_ava]):
        #Get the ego coordinates of the srcAnim animation joint
        # aux_jointname = skeletonmap.getmatchingjoint(joint.name, srcAnim).name
        # ego = EgocentricCoordinate.egolist[egoindex].getCoordFrame(frame)
        ego = EgocentricCoordinate.egolist[egoindex]

        #For each mesh triangle
        vec_displacement = []
        de_refpoint = []
        position = []
        taulist = []
        normallist = []
        for i in range(len(bodymesh_tgtAnim)+len(headmesh_tgtAnim)):
            if i<len(headmesh_tgtAnim):
                #de_refpoint_aux, normal = mathutils.getCentroid(mesh[i][0], mesh[i][1], mesh[i][2])
                de_refpoint_aux, normal = mathutils.barycentric2cartesian(ego.refpoint[i], mesh[i][0], mesh[i][1], mesh[i][2])
                if joint == lhand_ava: kinpath = np.asarray([vec_neck, lvec_clavicle, lvec_arm, lvec_fore])
                elif joint == rhand_ava: kinpath = np.asarray([vec_neck, rvec_clavicle, rvec_arm, rvec_fore])
                elif joint == lforearm_ava: kinpath = np.asarray([vec_neck, lvec_clavicle, lvec_arm])
                elif joint == rforearm_ava: kinpath = np.asarray([vec_neck, rvec_clavicle, rvec_arm])
                elif joint == lfoot_ava: kinpath = np.asarray([vec_neck, vec_spine, lvec_femur, lvec_upleg, lvec_lowleg])
                elif joint == rfoot_ava: kinpath = np.asarray([vec_neck, vec_spine, rvec_femur, rvec_upleg, rvec_lowleg])
                elif joint == llowleg_ava: kinpath = np.asarray([vec_neck, vec_spine, lvec_femur, lvec_upleg])
                elif joint == rlowleg_ava: kinpath = np.asarray([vec_neck, vec_spine, rvec_femur, rvec_upleg])
            else:
                j = i-len(headmesh_tgtAnim)
                #de_refpoint_aux, normal = mathutils.getCentroid(mesh[i][0], mesh[i][1], mesh[i][2])
                de_refpoint_aux, normal = mathutils.barycentric2cartesian(ego.refpoint[i], mesh[i][0], mesh[i][1], mesh[i][2])
                if joint == lhand_ava: kinpath = np.asarray([vec_spine, lvec_clavicle, lvec_arm, lvec_fore])
                elif joint == rhand_ava: kinpath = np.asarray([vec_spine, rvec_clavicle, rvec_arm, rvec_fore])
                elif joint == lforearm_ava: kinpath = np.asarray([vec_spine, lvec_clavicle, lvec_arm])
                elif joint == rforearm_ava: kinpath = np.asarray([vec_spine, rvec_clavicle, rvec_arm])
                elif joint == lfoot_ava: kinpath = np.asarray([lvec_femur, lvec_upleg, lvec_lowleg])
                elif joint == rfoot_ava: kinpath = np.asarray([rvec_femur, rvec_upleg, rvec_lowleg])
                elif joint == llowleg_ava: kinpath = np.asarray([lvec_femur, lvec_upleg])
                elif joint == rlowleg_ava: kinpath = np.asarray([rvec_femur, rvec_upleg])

            # if  joint == rfoot_ava or joint == lfoot_ava:
            #     tau = (np.linalg.norm(kinpath, axis = 1)*ego.normcoef[i][:-1]).sum()
            #     vec_displacement_aux = ego.dispvector[i][:-1]*tau
            # else:
            tau = (np.linalg.norm(kinpath, axis = 1)*ego.normcoef[i]).sum()
            vec_displacement_aux = ego.dispvector[i]*tau
            taulist.append(tau)
            vec_displacement.append(vec_displacement_aux)
            de_refpoint.append(de_refpoint_aux)
            position.append(vec_displacement_aux+de_refpoint_aux)
            normallist.append(normal)

        #Get limb coordinates
        for values_returned in DenormEgoLimb(joint, tgtAnim, surfacetgtAnim, frame, vectors, jointpositions, ego, i+1):
            vec_displacement_aux, de_refpoint_aux, tau, normal = values_returned
            taulist.append(tau)
            vec_displacement.append(vec_displacement_aux)
            de_refpoint.append(de_refpoint_aux)
            position.append(vec_displacement_aux+de_refpoint_aux)
            normallist.append(normal)


        if joint == rfoot_ava or joint == lfoot_ava:
            jointPosition = joint.getPosition(frame)
            hipsPosition = tgtAnim.skeletonmap.hips.getPosition(frame)
            hipsGround = np.asarray([hipsPosition[0], 0, hipsPosition[2]])
            hipsHeight = np.asarray([0, hipsPosition[1], 0])
            de_refpoint_aux = np.asarray([jointPosition[0], 0, jointPosition[2]])
            if joint == rfoot:
                kinpath = np.asarray([-de_refpoint_aux, -hipsGround, hipsHeight, rvec_femur, rvec_upleg, rvec_lowleg])
            else:
                kinpath = np.asarray([-de_refpoint_aux, -hipsGround, hipsHeight, lvec_femur, lvec_upleg, lvec_lowleg])
            vec_displacement_aux = kinpath.sum(axis = 0)
            cos = np.empty(len(kinpath))
            tau = 0
            for i in range(len(cos)):
                cos[i] = mathutils.cosBetween(vec_displacement_aux, kinpath[i])
                tau = tau + np.linalg.norm(kinpath[i])*cos[i]
            vec_displacement_aux = ego.dispvector[-1]*tau
            taulist.append(tau)
            vec_displacement.append(vec_displacement_aux)
            de_refpoint.append(de_refpoint_aux)
            position.append(vec_displacement_aux+de_refpoint_aux)
            normallist.append([0,1,0])


        ego.tgt_dispvector = np.asarray(vec_displacement)
        ego.tgt_tau = np.asarray(taulist)
        ego.tgt_refpoint = np.asarray(de_refpoint)
        ego.targets = np.asarray(position)
        ego.tgt_normal = np.asarray(normallist)

#        if frame>200:
#            return ego.egolist, targets#, taulist, vec_displacement

    #TODO: DEBUG
    # print('  set: %.4f seconds.' % (time.time()-start))
    return ego.egolist  # targets, taulist, vec_displacement
