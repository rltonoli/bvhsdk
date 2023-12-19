# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:13:53 2018

@author: Rodolfo Luis Tonoli
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
from . import anim
import importlib
from itertools import combinations
from . import skeletonmap
from . import mathutils
#import plotanimation


class Surface:

    def __init__(self, name, highpolymesh=True):
        self.name = name
        self.points = []
        self.points.append(SurfacePoint('chestRight', 'mesh', 'Spine3'))
        self.points.append(SurfacePoint('chestLeft', 'mesh', 'Spine3'))
        self.points.append(SurfacePoint('abdomenRight', 'mesh', 'Spine'))
        self.points.append(SurfacePoint('abdomenLeft', 'mesh', 'Spine'))
        self.points.append(SurfacePoint('hipRight', 'mesh', 'Hips'))
        self.points.append(SurfacePoint('hipLeft', 'mesh', 'Hips'))
        self.points.append(SurfacePoint('thightRight', 'limb', 'RightUpLeg'))
        self.points.append(SurfacePoint('thightLeft', 'limb', 'LeftUpLeg'))
        self.points.append(SurfacePoint('shinRight', 'limb', 'RightLeg'))
        self.points.append(SurfacePoint('shinLeft', 'limb', 'LeftLeg'))
        self.points.append(SurfacePoint('abdomenUp', 'mesh', 'Spine2'))
        self.points.append(SurfacePoint('armRight', 'limb', 'RightArm'))
        self.points.append(SurfacePoint('foreRight', 'limb', 'RightForeArm'))
        self.points.append(SurfacePoint('armLeft', 'limb', 'LeftArm'))
        self.points.append(SurfacePoint('foreLeft', 'limb', 'LeftForeArm'))
        self.points.append(SurfacePoint('headRight', 'mesh', 'Head'))
        self.points.append(SurfacePoint('headLeft', 'mesh', 'Head'))
        self.points.append(SurfacePoint('earRight', 'mesh', 'Head'))
        self.points.append(SurfacePoint('earLeft', 'mesh', 'Head'))
        self.points.append(SurfacePoint('chinRight', 'mesh', 'Head'))
        self.points.append(SurfacePoint('chinLeft', 'mesh', 'Head'))
        self.points.append(SurfacePoint('cheekRight', 'mesh', 'Head'))
        self.points.append(SurfacePoint('cheekLeft', 'mesh', 'Head'))
        self.points.append(SurfacePoint('mouth', 'mesh', 'Head'))
        self.points.append(SurfacePoint('foreHead', 'mesh', 'Head'))
        self.points.append(SurfacePoint('backHeadRight', 'mesh', 'Head'))
        self.points.append(SurfacePoint('backHeadLeft', 'mesh', 'Head'))
        self.points.append(SurfacePoint('backHead', 'mesh', 'Head'))
        self.points.append(SurfacePoint('loinRight', 'mesh', 'Hips'))
        self.points.append(SurfacePoint('loinLeft', 'mesh', 'Hips'))
        self.points.append(SurfacePoint('loinUp', 'mesh', 'Spine1'))
        self.getSurfaceMesh(highpoly = highpolymesh)

    def getPoint(self, name):
        for point in self.points:
            if point.name == name:
                return point

    def NormalizeSurfaceData(self, hipsjoint):
        """
        Re-scale the avatar surface data to match the joint scale. It uses
        the global Y position of the hips to re-scale.

        hipsjoint: Class Joint object of the avatar hip (or the base of
        the spine)
        """
        try:
            hipsBasePosition = self.avatarHipsbaseposition
        except:
            print('avatarHipsbaseposition not found. It should be the last line of the avatar surface file')
        hipsposition = hipsjoint.getPosition(0)
        ratio = hipsposition[1]/hipsBasePosition[1]
        #Pode ser que o BVH esteja na origem mas o fbx utilizado não. Por exemplo, junta Hips no bvh = [0,100,0]
        #Junta hips no fbx = [-0.04,0.1,-0.02]
        #Pega o vetor da diferença da posição para transladar os pontos da superfície
        transX = hipsposition[0] - hipsBasePosition[0]*ratio
        transZ = hipsposition[2] - hipsBasePosition[2]*ratio
        for point in self.points:
            if point.pointtype == 'mesh':
                point.baseposition = point.baseposition*ratio
                point.baseposition[0] = point.baseposition[0] + transX
                point.baseposition[2] = point.baseposition[2] + transZ
            else:
                point.radius = point.radius*ratio


    def getCentroid(self, animation,meshtype, index, frame):
        if meshtype=='body':
            mesh = self.bodymesh[index]
        elif meshtype=='head':
            mesh = self.headmesh[index]
        centroid = np.mean([mesh[0].getPosition(animation,frame), mesh[1].getPosition(animation,frame), mesh[2].getPosition(animation,frame)], axis=0)
        return centroid[:-1]

    def getSurfaceMesh(self, highpoly=True):
        try:
            return self.headmesh, self.bodymesh
        except:
            headpoints = []
            bodypoints = []
            for point in self.points:
                if point.jointlock == 'Head':
                    headpoints.append(point)
                else:
                    if point.pointtype == 'mesh':
                        bodypoints.append(point)

            if highpoly:
                #21 Triangles
                self.headmesh = []
                comb = [[0,1,6],
                        [0,1,7],
                        [0,1,4],
                        [0,1,5],
                        [0,1,8],
                        [0,1,9],
                        [0,1,12],
                        [0,1,10],
                        [0,1,11],
                        [2,4,6],
                        [2,4,8],
                        [2,4,9],
                        [3,5,7],
                        [3,5,8],
                        [3,5,9],
                        [6,7,9],
                        [6,8,9],
                        [7,8,9],
                        [4,5,8],
                        [4,5,9],
                        [10,11,12]
                ]
            else:
                #13 Triangles
                self.headmesh = []
                comb = [[0,9,1],
                        [0,2,6],
                        [2,4,6],
                        [1,7,3],
                        [3,7,5],
                        [0,6,9],
                        [1,9,7],
                        [4,5,8],
                        [6,7,8],
                        [6,7,9],
                        [10,11,12],
                        [0,10,2],
                        [1,3,11]
                ]
            for tri in range(len(comb)):
                self.headmesh.append([headpoints[comb[tri][0]],headpoints[comb[tri][1]],headpoints[comb[tri][2]]])

            if highpoly:
                #15 Triangles
                self.bodymesh = []
                comb = [[0,1,2],
                        [0,1,3],
                        [0,1,5],
                        [0,4,5],
                        [0,6,1],
                        [0,6,2],
                        [0,6,3],
                        [1,2,3],
                        [1,4,5],
                        [2,3,6],
                        [2,3,4],
                        [2,3,5],
                        [7,8,9],
                        [4,5,9],
                        [6,7,8]
                ]
            else:
                #9 Triangles
                self.bodymesh = []
                comb = [[0,6,1],
                        [0,2,6],
                        [1,6,3],
                        [2,3,6],
#                        [2,3,6],
                        [2,4,5],
                        [3,4,5],
                        [9,8,7],
                        [2,7,4],
                        [3,5,8],
                ]

            #Total 36 Mesh Triangles
            for tri in range(len(comb)):
                self.bodymesh.append([bodypoints[comb[tri][0]],bodypoints[comb[tri][1]],bodypoints[comb[tri][2]]])

            return self.headmesh, self.bodymesh


class SurfacePoint:
    def __init__(self, pointname, pointtype, jointlock):
        self.name = pointname
        self.pointtype = pointtype
        self.jointlock = jointlock
        self.position = []
        self.baseposition = []
        self.calibrationLocalTransform = []

    def getPosition(self, animation, frame=0):
        """
        Estima a posição da superfície baseado na transformada local calculada em
        GetAvatarSurfaceLocalTransform()
        """
        if self.pointtype=='mesh':
            joint = skeletonmap.getmatchingjoint(self.jointlock, animation)
            assert joint, str.format('Could not find attached joint in animation: %s' % self.jointlock)
            globalPointTransform = np.dot(joint.getGlobalTransform(frame), self.calibrationLocalTransform)
            return np.dot(globalPointTransform,[0,0,0,1])[:-1]



def isNumber(s):
    """
    Return True if the string is a number

    :type s: string
    :param s: String to be tested as a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def checkName(name):
    """
    Return True if the file in the path/name provided exists inside current path

    :type name: string
    :param name: Local path/name of the back file
    """
    currentpath = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(currentpath, name)
    return os.path.isfile(fullpath)


def FindHandsZeroSpeedGaps(animation, threshold=None, plot=False, minrangewide = 20, minpeakwide=40):
    """
    Get the animation and  find the

    """
    rightHand = animation.getskeletonmap().rhandmiddle
    leftHand = animation.getskeletonmap().lhandmiddle

    #Get right and left hand position
    rightHandPos = np.asarray([rightHand.getPosition(frame) for frame in range(animation.frames)]).T
    leftHandPos = np.asarray([leftHand.getPosition(frame) for frame in range(animation.frames)]).T
    rightHandVel = np.zeros([rightHandPos.shape[1]-1])
    leftHandVel = np.zeros([rightHandPos.shape[1]-1])
    #Calculate velocity for both
    for frameCount in range(rightHandPos.shape[1]-1):
        deltaXsquare = np.square(rightHandPos[0,frameCount+1]-rightHandPos[0,frameCount])
        deltaYsquare = np.square(rightHandPos[1,frameCount+1]-rightHandPos[1,frameCount])
        deltaZsquare = np.square(rightHandPos[2,frameCount+1]-rightHandPos[2,frameCount])
        rightHandVel[frameCount] = np.sqrt(deltaXsquare + deltaYsquare + deltaZsquare)
        deltaXsquare = np.square(leftHandPos[0,frameCount+1]-leftHandPos[0,frameCount])
        deltaYsquare = np.square(leftHandPos[1,frameCount+1]-leftHandPos[1,frameCount])
        deltaZsquare = np.square(leftHandPos[2,frameCount+1]-leftHandPos[2,frameCount])
        leftHandVel[frameCount] = np.sqrt(deltaXsquare + deltaYsquare + deltaZsquare)

    #Calculate mean velocity
    rhVelmean = np.mean(rightHandVel[:])
    lhVelmean = np.mean(leftHandVel[:])
    if threshold:
        if threshold > 1:
            print('Threshold value not accepted, please choose 0 to 1')
            return None, None, None, None
        else:
            rhVelmean = rhVelmean*threshold
            lhVelmean = lhVelmean*threshold
    #where the velocity is lower than its mean for the right hand
    rhwhere = np.asarray(np.where(rightHandVel<rhVelmean)[0])
    #where the velocity is lower than its mean for the lef hand
    lhwhere = np.asarray(np.where(leftHandVel<lhVelmean)[0])

    #Locate the gaps in velocity and assume it is a calibration position
    rhSteadyFrames, lhSteadyFrames = [], []
    rhaux, lhaux = 0,0

    for i in range(rhwhere.shape[0]-1):
        #The minpeakwide prevents the detection of narrow peaks
        #if there is a gap in index, i.e., a peak:
        if (rhwhere[i]+1!=rhwhere[i+1] and rhwhere[i+1] - rhwhere[i]+1 > minpeakwide):
            #if it is NOT a narrow hole (noisy signal)
            if rhwhere[i]-rhaux > minrangewide:
                rhSteadyFrames.append([rhaux, rhwhere[i]])
                rhaux = rhwhere[i+1]
            else:
                rhaux = rhwhere[i+1]
        if i==rhwhere.shape[0]-2:
            rhSteadyFrames.append([rhaux, rhwhere[i]])

    #now for the left hand
    for i in range(lhwhere.shape[0]-1):
        if (lhwhere[i]+1!=lhwhere[i+1] and lhwhere[i+1] - lhwhere[i]+1 > minpeakwide):
            if lhwhere[i]-lhaux > minrangewide:
                lhSteadyFrames.append([lhaux, lhwhere[i]])
                lhaux = lhwhere[i+1]
            else:
                lhaux = lhwhere[i+1]
        if i==lhwhere.shape[0]-2:
            lhSteadyFrames.append([lhaux, lhwhere[i]])

    rightHandPositions = np.zeros((len(rhSteadyFrames),3))
    rightHandFrameRange = []
    leftHandPositions = np.zeros((len(lhSteadyFrames),3))
    leftHandFrameRange = []

    if plot:
        print('Right Hand:')
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rightHandVel[:], label='Right Hand', color = 'black')
#        plt.plot(leftHandVel[:], label='Left Hand')
        ax.minorticks_on()
        ax.set_ylim(bottom=0)
        ax.set_xlim([0,len(rightHandVel[:])])
        ax.set_title('Zero-Speed Right-Hand Frame Ranges', fontsize=16)
        ax.set_xlabel('Frame', fontsize=14)
        ax.set_ylabel('Speed (units/frame)', fontsize=14)
    for i in range(rightHandPositions.shape[0]):
        window = int(((rhSteadyFrames[i][1] - rhSteadyFrames[i][0])/2)*0.5)
        center = int((rhSteadyFrames[i][1] + rhSteadyFrames[i][0])/2)
        rightHandPositions[i,:] = np.mean([rightHand.getPosition(frame) for frame in range(center-window,center+window)],axis=0)
        rightHandFrameRange.append([center,window])
        if plot:
            #To add the legend
            if i ==0: ax.plot([center-window, center-window],[0,1], color='red', label='Frame Ranges')
            ax.plot([center-window, center-window],[0,1], color='red')
            ax.plot([center+window, center+window],[0,1], color='red')
            ax.text(center,1.05,str(i), horizontalalignment='center')


    if plot:
        ax.legend()
        #plt.show(fig)
        i = 0
        output_filename = 'rh'
        of_aux = output_filename
        while checkName(output_filename+'.png'):
            i = i + 1
            output_filename = of_aux + str(i)
        fig.savefig(output_filename+'.png', dpi=fig.dpi)

    if plot:
        print('Left Hand:')
        #plt.show()
        plt.clf()
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1, 1, 1)
#        plt.plot(rightHandVel[:], label='Right Hand')
        ax.plot(leftHandVel[:], label='Left Hand', color = 'black')
        ax.legend()
        ax.minorticks_on()
        ax.set_ylim(bottom=0)
        ax.set_xlim([0,len(leftHandVel[:])])
        ax.set_title('Zero-Speed Left-Hand Frame Ranges', fontsize=16)
        ax.set_xlabel('Frame', fontsize=14)
        ax.set_ylabel('Speed (units/frame)', fontsize=14)

    for i in range(leftHandPositions.shape[0]):
        window = int(((lhSteadyFrames[i][1] - lhSteadyFrames[i][0])/2)*0.5)
        center = int((lhSteadyFrames[i][1] + lhSteadyFrames[i][0])/2)
        leftHandPositions[i,:] = np.mean([leftHand.getPosition(frame) for frame in range(center-window,center+window)],axis=0)
        leftHandFrameRange.append([center,window])
        if plot:
            #To add the legend
            if i == 0:  ax.plot([center-window, center-window],[0,1], color='red', label='Frame Ranges')
            ax.plot([center-window, center-window],[0,1], color='red')
            ax.plot([center+window, center+window],[0,1], color='red')
            ax.text(center,1.05,str(i), horizontalalignment='center')


    if plot:
        ax.legend()
        #plt.show(fig)
        i = 0
        output_filename = 'lh'
        of_aux = output_filename
        while checkName(output_filename+'.png'):
            i = i + 1
            output_filename = of_aux + str(i)
        fig.savefig(output_filename+'.png', dpi=fig.dpi)

    return rightHandPositions, leftHandPositions, rightHandFrameRange, leftHandFrameRange



def GetCalibrationFromBVHS(frontAnimName, headAnimName, backAnimName, savefile=True, debugmode = False, minrangewide = 20, minpeakwide=40, handthreshold = 0.5):
    """
    Opens the calibration animations (bvh files) and get the calibration positions of the points.

    If savefile = True, saves a .txt file containing the calibration local transform of each point

    :type frontAnimName: string
    :param frontAnimName: Local path/name of the front calibration animation

    :type headAnimName: string
    :param headAnimName: Local path/name of the head calibration animation

    :type backAnimName: string
    :param backAnimName: Local path/name of the back calibration animation
    """

    def getLocalTransform(animation, jointlock, frame, pointpos, handthick = 3.5):
        """
        Return the local transform of the surface point in respect to its parent joint jointlock

        :type animation: bvhsdk.Animation
        :param animation: Calibration animation

        :type jointlock: bvhsdk.Joint
        :param jointlock: Joint to be used as parent of the surface point

        :type frame: int
        :param frame: The frame number of the pose to evaluate

        :type pointpos: numpy.ndarray
        :param pointpos: Position of the surface point
        """
        jointTransform = jointlock.getGlobalTransform(frame)
        if handthick:
            jointPosition = np.dot(jointTransform, [0, 0, 0, 1])[:-1]
            # Find parametric equation to remove hands surface (hand thickness)
            vec = jointPosition - pointpos
            distance = np.linalg.norm(vec)
            t = handthick/distance
            position = pointpos + t*vec
        else:
            position = pointpos
        jointInverse = mathutils.inverseMatrix(jointTransform)
        globalTransform = mathutils.matrixTranslation(position[0], position[1], position[2])
        localTransform = np.dot(jointInverse, globalTransform)
        return localTransform

    def getRadius(animation, jointlock, frame, pointpos, handthick=3.5):
        """
        Return the Radius of the limb, the distance between the surface point, p0,
        and the bone (vector from the jointlock, p1, to the jointlock's child, p2)

        Distance between point p0 and line passing through p1 and p2:
        d = |(p0 - p1) x (p0 - p2)|/|p2-p1| = |(p0p1) x (p0p2)|/|p2p1|

        :type animation: bvhsdk.Animation
        :param animation: Calibration animation

        :type jointlock: bvhsdk.Joint
        :param jointlock: Joint to be used as parent of the surface point

        :type frame: int
        :param frame: The frame number of the pose to evaluate

        :type pointpos: numpy.ndarray
        :param pointpos: Position of the surface point
        """
#       TODO: testar esse caso!!!
        p1 = jointlock.getPosition(frame)
        p2 = jointlock.getChildren(0).getPosition(frame)
        p0 = np.asarray(pointpos)
        p0p1 = p0-p1
        p0p2 = p0-p2
        p2p1 = p2-p1
        d = np.linalg.norm(np.cross(p0p1, p0p2))/np.linalg.norm(p2p1)
        if handthick:
            d = d-3.5
        return d

    if debugmode:
        np.set_printoptions(precision=3, suppress=True)
        log = []

    realpath = os.path.dirname(os.path.realpath(__file__))
#    arq = os.path.join(realpath, 'Superficie\Frente_2_mcp.bvh')
    arq = os.path.join(realpath, frontAnimName)
    print('Reading first BVH file')
    animation = bvhsdk.ReadFile(arq)
    print('Getting first surface data')
    right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, plot=debugmode, minrangewide=minrangewide, minpeakwide=minpeakwide)
    threshold = 0.9
    while len(right) < 14 and len(left) < 14:
        right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, threshold, plot=debugmode, minrangewide=minrangewide, minpeakwide=minpeakwide)
        threshold = threshold - 0.1
    localTransforms = []
    radius = []

    skipleft = 0
    skipright = 0

    # Starting at TPose, ignore index 0
    if debugmode:
        log.append('First BVH File')
        log.append('Right Hand Position in Gaps:')
        log = log + list(right)
        log.append('Left Hand Position in Gaps:')
        log = log + list(left)
        log.append('Right Hand Gaps Center:')
        log = log + list(np.asarray(rightrange)[:, 0])
        log.append('Left Hand Gaps Center:')
        log = log + list(np.asarray(leftrange)[:, 0])
        log.append('Right Hand Gaps Range:')
        log = log + list(np.asarray(rightrange)[:, 1])
        log.append('Left Hand Gaps Range:')
        log = log + list(np.asarray(leftrange)[:, 1])
        log.append('Starting analysis')

    # Check if performer lowered his arm after the TPose (Rest Pose)
    if max(right[1, :]) < max(animation.root.getPosition(frame=0)) or max(left[1, :]) < max(animation.root.getPosition(frame=0)):
        right = np.delete(right, 1, 0)
        left = np.delete(left, 1, 0)
        rightrange = np.delete(rightrange, 1, 0)
        leftrange = np.delete(leftrange, 1, 0)
        if debugmode:
            log.append('Rest Pose after TPose detected. Index 1 deleted.')

    # Chest Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine3, rightrange[1][0], right[1, :]))
    # Chest Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine3, leftrange[1][0], left[1, :]))
    # Abdomen Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine, rightrange[2][0], right[2, :]))
    # Abdomen Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine, leftrange[2][0], left[2, :]))
    # Hip Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().hips, rightrange[3][0], right[3, :]))
    # Hip Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().hips, leftrange[3][0], left[3, :]))
    # Thigh Right
    radius.append(getRadius(animation, animation.getskeletonmap().rupleg, rightrange[4][0], right[4, :]))
    # Thigh Left
    radius.append(getRadius(animation, animation.getskeletonmap().lupleg, leftrange[4][0], left[4, :]))
    # Shin Right
    radius.append(getRadius(animation, animation.getskeletonmap().rlowleg, rightrange[5][0], right[5, :]))
    # Shin Left
    radius.append(getRadius(animation, animation.getskeletonmap().llowleg, leftrange[5][0], left[5, :]))

    if debugmode:
        log.append('Chest Right Position: %s. Frame: %f.' % (right[1, :], rightrange[1][0]))
        log.append('Chest Left Position: %s. Frame: %f' % (left[1, :], leftrange[1][0]))
        log.append('Abdomen Right Position: %s. Frame: %f' % (right[2, :], rightrange[2][0]))
        log.append('Abdomen Left Position: %s. Frame: %f' % (left[2, :], leftrange[2][0]))
        log.append('Hip Right Position: %s. Frame: %f' % (right[3, :], rightrange[3][0]))
        log.append('Hip Left Position: %s. Frame: %f' % (left[3, :], leftrange[3][0]))
        log.append('Thigh Right Position: %s. Frame: %f' % (right[4, :], rightrange[4][0]))
        log.append('Thigh Left Position: %s. Frame: %f' % (left[4, :], leftrange[4][0]))
        log.append('Shin Right Position: %s. Frame: %f' % (right[5, :], rightrange[5][0]))
        log.append('Shin Left Position: %s. Frame: %f' % (left[5, :], leftrange[5][0]))

    # Rest Pose
    if max(right[6, :]) < max(animation.root.getPosition(frame=0)) or max(left[6, :]) < max(animation.root.getPosition(frame=0)):
        skipright = skipright + 1

    # Abdomen Up
    if len(right) > len(left):
        # Check which hand was used to touch the surface. The hand used will have more gaps
        # Right Hand used
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine2, rightrange[6+skipright][0], right[6+skipright, :]))
        if debugmode:
            log.append('Abdomen Up Position: %s. Frame: %f. Index: %i. Right Hand used.' % (right[6+skipright, :], rightrange[6+skipright][0], 6+skipright))
        skipright = skipright + 1
    else:
        # Left Hand used
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine2, leftrange[6+skipleft][0], left[6+skipleft, :]))
        if debugmode:
            log.append('Abdomen Up Position: %s. Frame: %f. Index: %i. Left Hand used.' % (left[6+skipleft, :], leftrange[6+skipleft][0], 6+skipleft))
        skipleft = skipleft + 1

    # Rest Pose
    # Arms Up
    skipright = skipright + 1
    skipleft = skipleft + 1

    if rightrange[7+skipright][1] > leftrange[7+skipleft][1]:
        # Check which hand was calibrated first, i.e., if the right hand has a bigger gap, it was steady
        # being calibrated (while the left hand was moving to touch the surface of the right arm)
        # Note: The right hand is used to calibrate the left arm and vice-versa
        # First Left Hand used to calibrate Right Arm
        skipright = skipright + 1
        if debugmode:
            log.append('Calibrating Right Arm first. Using Left Hand')
    else:
        # First Right Hand used to calibrate Left Arm
        if debugmode:
            log.append('Calibrating Left Arm first. Using Right Hand')
        skipleft = skipleft + 1

    # Arm Right (calibrated using left hand)
    radius.append(getRadius(animation, animation.getskeletonmap().rarm, leftrange[7+skipleft][0], left[7+skipleft,:]))
    if debugmode: log.append('Arm Right Position: %s. Frame: %f. Index: %i'% (left[7+skipleft,:], leftrange[7+skipleft][0], 7+skipleft))
    # ForeArm Right (calibrated using left hand)
    radius.append(getRadius(animation, animation.getskeletonmap().rforearm, leftrange[8+skipleft][0], left[8+skipleft,:]))
    if debugmode: log.append('ForeArm Right Position: %s. Frame: %f. Index: %i'% (left[8+skipleft,:], leftrange[8+skipleft][0], 8+skipleft))
    # Arm Left (calibrated using right hand)
    radius.append(getRadius(animation, animation.getskeletonmap().larm, rightrange[7+skipright][0], right[7+skipright,:]))
    if debugmode: log.append('Arm Left Position: %s. Frame: %f. Index: %i'% (right[7+skipright,:], rightrange[7+skipright][0], 7+skipright))
    # ForeArm Left (calibrated using right hand)
    radius.append(getRadius(animation, animation.getskeletonmap().lforearm, rightrange[8+skipright][0], right[8+skipright,:]))
    if debugmode: log.append('ForeArm Left Position: %s. Frame: %f. Index: %i'% (right[8+skipright,:], rightrange[8+skipright][0], 8+skipright))


    print('First file done')
    arq = os.path.join(realpath, headAnimName)
    print('Reading second BVH file')
    animation = bvhsdk.ReadFile(arq)
    print('Getting second surface data')
    # Starting with a lower threshold because movements in the head tend to be smaller
    threshold = 0.4
    right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, threshold,plot = debugmode, minrangewide = minrangewide, minpeakwide=minpeakwide)
    while len(right) < 11 and len(left) < 11:
        right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, threshold, plot = debugmode, minrangewide = minrangewide, minpeakwide=minpeakwide)
        threshold = threshold - 0.1

    skipleft = 0
    skipright = 0

    #Starting at TPose, ignore index 0
    if debugmode:
        log.append('\n\n')
        log.append('Second BVH File')
        log.append('Right Hand Position in Gaps:')
        log = log + list(right)
        log.append('Left Hand Position in Gaps:')
        log = log + list(left)
        log.append('Right Hand Gaps Center:')
        log = log + list(np.asarray(rightrange)[:,0])
        log.append('Left Hand Gaps Center:')
        log = log + list(np.asarray(leftrange)[:,0])
        log.append('Right Hand Gaps Range:')
        log = log + list(np.asarray(rightrange)[:,1])
        log.append('Left Hand Gaps Range:')
        log = log + list(np.asarray(leftrange)[:,1])
        log.append('Starting analysis')

    #Check if performer lowered his arm after the TPose
    if max(right[1,:]) < max(animation.root.getPosition(frame=0)) or max(left[1,:]) < max(animation.root.getPosition(frame=0)):
        right = np.delete(right, 1, 0)
        left = np.delete(left, 1, 0)
        rightrange = np.delete(rightrange, 1, 0)
        leftrange = np.delete(leftrange, 1, 0)
        if debugmode: log.append('Rest Pose after TPose detected. Index 1 deleted.')

    # Head Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[1][0], right[1, :]))
    if debugmode: log.append('Head Right Position: %s. Frame: %f.'% (right[1,:], rightrange[1][0]))
    # Head Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[1][0], left[1, :]))
    if debugmode: log.append('Head Left Position: %s. Frame: %f'% (left[1,:], leftrange[1][0]))
    # Ear Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[2][0], right[2, :]))
    if debugmode: log.append('Ear Right Position: %s. Frame: %f.'% (right[2,:], rightrange[2][0]))
    # Ear Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[2][0], left[2, :]))
    if debugmode: log.append('Ear Left Position: %s. Frame: %f'% (left[2,:], leftrange[2][0]))
    # Chin Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[3][0], right[3, :]))
    if debugmode: log.append('Chin Right Position: %s. Frame: %f.'% (right[3,:], rightrange[3][0]))
    # Chin Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[3][0], left[3, :]))
    if debugmode: log.append('Chin Left Position: %s. Frame: %f'% (left[3,:], leftrange[3][0]))
    # Cheek Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[4][0], right[4, :]))
    if debugmode: log.append('Cheek Right Position: %s. Frame: %f.'% (right[4,:], rightrange[4][0]))
    # Cheek Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[4][0], left[4, :]))
    if debugmode: log.append('Cheek Left Position: %s. Frame: %f'% (left[4,:], leftrange[4][0]))

    print( len(right) )
    print( len(left) )
    if len(right) >= len(left):
        print('ENTREI AQUI')
        #Right Hand used
        if debugmode: log.append('Right Hand used to calibrate Mouth and Forehead')
        #Mouth
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[5][0], right[5,:]))
        if debugmode: log.append('Mouth Position: %s. Frame: %f. Index: %i'% (right[5,:], rightrange[5][0], 5))
        skipright = skipright + 1
        #Forehead
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[5+skipright][0], right[5+skipright,:]))
        if debugmode: log.append('Forehead Position: %s. Frame: %f. Index: %i'% (right[5+skipright,:], rightrange[5+skipright][0], 5+skipright))

        if max(right[5+skipright+1,:]) < max(animation.root.getPosition(frame=0)):
            #Back to rest pose
            skipright = skipright + 1

    else:
        #Left Hand used
        if debugmode: log.append('Left Hand used to calibrate Mouth and Forehead')
        #Mouth
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[5][0], left[5,:]))
        if debugmode: log.append('Mouth Position: %s. Frame: %f. Index: %i'% (left[5,:], leftrange[5][0], 5))
        skipleft = skipleft + 1
        #Forehead
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[5+skipleft][0], left[5+skipleft,:]))
        if debugmode: log.append('Forehead Position: %s. Frame: %f. Index: %i'% (left[5+skipleft,:], leftrange[5+skipleft][0], 5+skipleft))
        if max(left[5+skipleft+1,:]) < max(animation.root.getPosition(frame=0)):
            #Back to rest pose
            skipleft = skipleft + 1

    #skipleft += 1
    #Back Head Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[6+skipright][0], right[6+skipright,:]))
    if debugmode: log.append('Backhead Right Position: %s. Frame: %f. Index: %i'% (right[6+skipright,:], rightrange[6+skipright][0], 6+skipright))
    #Back Head Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[6+skipleft][0], left[6+skipleft,:]))
    if debugmode: log.append('Backhead Left Position: %s. Frame: %f. Index: %i'% (left[6+skipleft,:], leftrange[6+skipleft][0], 6+skipleft))
    #Back Head Middle
    if len(right) >= len(left):
        #Right Hand used
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, rightrange[7+skipright][0], right[7+skipright,:]))
        if debugmode: log.append('Backhead Middle Position: %s. Frame: %f. Index: %i'% (right[7+skipright,:], rightrange[7+skipright][0], 7+skipright))
    else:
        #Left Hand used
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().head, leftrange[7+skipleft][0], left[7+skipleft,:]))
        if debugmode: log.append('Backhead Middle Position: %s. Frame: %f. Index: %i'% (left[7+skipleft,:], leftrange[7+skipleft][0], 7+skipleft))
#    print('Cabeça: ')
#    print(right)
#    print(left)

    print('Second file done')
    arq = os.path.join(realpath, backAnimName)
    print('Reading third BVH file')
    animation = bvhsdk.ReadFile(arq)
    print('Getting third surface data')
    right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, plot=debugmode, minrangewide = 30, minpeakwide=minpeakwide)
    threshold = 0.9
    while len(right) < 4 and len(left) < 4:
        right, left, rightrange, leftrange = FindHandsZeroSpeedGaps(animation, threshold, plot=debugmode, minrangewide = minrangewide, minpeakwide=minpeakwide)
        threshold = threshold - 0.1


    #Starting at TPose, ignore index 0
    if debugmode:
        log.append('\n\n')
        log.append('Third BVH File')
        log.append('Right Hand Position in Gaps:')
        log = log + list(right)
        log.append('Left Hand Position in Gaps:')
        log = log + list(left)
        log.append('Right Hand Gaps Center:')
        log = log + list(np.asarray(rightrange)[:,0])
        log.append('Left Hand Gaps Center:')
        log = log + list(np.asarray(leftrange)[:,0])
        log.append('Right Hand Gaps Range:')
        log = log + list(np.asarray(rightrange)[:,1])
        log.append('Left Hand Gaps Range:')
        log = log + list(np.asarray(leftrange)[:,1])
        log.append('Starting analysis')

    #Check if performer lowered his arm after the TPose
    if max(right[1,:]) < max(animation.root.getPosition(frame=0)) or max(left[1,:]) < max(animation.root.getPosition(frame=0)):
        right = np.delete(right, 1, 0)
        left = np.delete(left, 1, 0)
        rightrange = np.delete(rightrange, 1, 0)
        leftrange = np.delete(leftrange, 1, 0)
        if debugmode: log.append('Rest Pose after TPose detected. Index 1 deleted.')

    #Loin Right
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().hips, rightrange[1][0], right[1,:]))
    if debugmode: log.append('Loin Right Position: %s. Frame: %f.'% (right[1,:], rightrange[1][0]))
    #Loin Left
    localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().hips, leftrange[1][0], left[1,:]))
    if debugmode: log.append('Loin Right Position: %s. Frame: %f.'% (left[1,:], leftrange[1][0]))
    #Loin Up
    ground = np.asarray([0,1,0])
    leftYpos = np.dot(left[2,:], ground)
    rightYpos = np.dot(right[2,:], ground)
    if rightYpos > leftYpos:
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine1, rightrange[2][0], right[2,:]))
        if debugmode: log.append('Loin Up Position: %s. Frame: %f.'% (right[2,:], rightrange[2][0]))
    else:
        localTransforms.append(getLocalTransform(animation, animation.getskeletonmap().spine1, leftrange[2][0], left[2,:]))
        if debugmode: log.append('Loin Up Position: %s. Frame: %f.'% (left[2,:], leftrange[2][0]))

#    print('Costas: ')
#    print(right)
#    print(left)

    ###########################################################################

    #TODO: Uso a transformada local de um frame apenas

    ###########################################################################
    if savefile:
        filename = 'surface.txt'
        i = 0
        while checkName(filename):
            i = i + 1
            filename = 'surface' + str(i) + '.txt'
        with open(os.path.join(realpath, filename), "w") as text_file:
            for i in range(len(localTransforms)):
                print("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (localTransforms[i][0][0], localTransforms[i][0][1], localTransforms[i][0][2], localTransforms[i][0][3], localTransforms[i][1][0], localTransforms[i][1][1], localTransforms[i][1][2], localTransforms[i][1][3], localTransforms[i][2][0], localTransforms[i][2][1], localTransforms[i][2][2], localTransforms[i][2][3], localTransforms[i][3][0], localTransforms[i][3][1], localTransforms[i][3][2], localTransforms[i][3][3]), file=text_file)
                if i == 5:
                    print("%f\n%f\n%f\n%f" % (radius[0], radius[1], radius[2], radius[3]), file=text_file)
                elif i == 6:
                    print("%f\n%f\n%f\n%f" % (radius[4], radius[5], radius[6], radius[7]), file=text_file)
        print('Surface calibration file %s saved' % (os.path.join(realpath, filename)))
    if debugmode:
        filename = 'log_surface.txt'
        i = 0
        while checkName(filename):
            i = i + 1
            filename = 'log_surface' + str(i) + '.txt'
        with open(os.path.join(realpath, filename), "w") as text_file:
            for i in range(len(log)):
                print(log[i], file=text_file)


def GetMoCapSurfaceFromTXT(path=None, highpolymesh=True, minradius=True):
    if not path:
        realpath = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(realpath, 'surface.txt')
    try:
        with open(path, "r") as file:
            data = [np.asarray(line.replace('\n', '').split(','), dtype='float') if len(line.split(',')) > 1 else float(line.replace('\n', '')) for line in file]
    except FileNotFoundError as e:
        print('Invalid path provided or surface.txt not found in %s.\nError: %s.' % (path, str(e)))
        raise
    mocapSurface = Surface('MoCap', highpolymesh=highpolymesh)
    for point, i in zip(mocapSurface.points, range(len(mocapSurface.points))):
        if type(data[i]) == float:
            point.radius = data[i]
        else:
            point.calibrationLocalTransform = np.asarray([[data[i][j*4], data[i][j*4+1], data[i][j*4+2], data[i][j*4+3]] for j in range(4)])
    if minradius:
        limbnames = [['thightRight', 'thightLeft'], ['shinRight', 'shinLeft'], ['armRight', 'armLeft'], ['foreRight', 'foreLeft']]
        for limb in limbnames:
            right = mocapSurface.getPoint(limb[0])
            left = mocapSurface.getPoint(limb[1])
            if right.radius < left.radius:
                left.radius = right.radius
            else:
                right.radius = left.radius
    return mocapSurface


def GetAvatarSurfaceFromCSV(path, highpolymesh=True):
    try:
        with open(path, "r") as file:
            data = [np.asarray(line.replace('\n', '').split(','), dtype='float') if len(line.split(',')) > 1 else float(line.replace('\n', '')) for line in file]
    except FileNotFoundError as e:
        print('Invalid path provided or surface file not found in %s.\nError: %s.' % (path, str(e)))
        return None
    avatarSurface = Surface('Avatar', highpolymesh=highpolymesh)
    for point, i in zip(avatarSurface.points, range(len(avatarSurface.points))):
        if type(data[i]) == float:
            point.radius = data[i]
        else:
            point.baseposition = np.asarray(data[i], dtype='float')
    avatarSurface.avatarHipsbaseposition = np.asarray(data[i+1], dtype='float')
    return avatarSurface


def GetAvatarSurfaceLocalTransform(avatar, avatarSurface):
    """
    Pega a local baseado na animação com um frame da TPose
    """
    for point in avatarSurface.points:
        if point.pointtype == 'mesh':
            joint = skeletonmap.getmatchingjoint(point.jointlock, avatar)
            if not joint:
                print('Something went wrong in retarget.GetAvatarSurfaceLocalTransform()')
                print('Matching joint could not be found')
            else:
                globalTransform = mathutils.matrixTranslation(point.baseposition[0], point.baseposition[1], point.baseposition[2])
                parentInverse = mathutils.inverseMatrix(joint.getGlobalTransform(frame=0))
                point.calibrationLocalTransform = np.dot(parentInverse, globalTransform)
            joint = None


def AvatarSurfacePositionEstimation(avatar, avatarSurface):
    """
    Estimates the position of the surface points based on the local transform based on GetAvatarSurfaceLocalTransform()
    """
    print('DO NOT USE THIS FUNCTION. AvatarSurfacePositionEstimation is depreciated.')
    for point in avatarSurface.points:
        if point.pointtype == 'mesh':
            joint = skeletonmap.getmatchingjoint(point.jointlock, avatar)
            if not joint:
                print('Something went wrong in retarget.AvatarSurfacePositionEstimation()')
                print('Matching joint could not be found')
            else:
                for frame in range(avatar.frames):
                    globalPointTransform = np.dot(joint.getGlobalTransform(frame=frame), point.calibrationLocalTransform)
                    point.position.append(np.dot(globalPointTransform, [0, 0, 0, 1]))


# SurfacePoints('chestRight', 'mesh', 'Spine3')
# SurfacePoints('chestLeft', 'mesh', 'Spine3')
# SurfacePoints('abdomenRight', 'mesh', 'Spine')
# SurfacePoints('abdomenLeft', 'mesh', 'Spine')
# SurfacePoints('hipRight', 'mesh', 'Hips')
# SurfacePoints('hipLeft', 'mesh', 'Hips')
# SurfacePoints('thightRight', 'limb', 'RightUpLeg')
# SurfacePoints('thightLeft', 'limb', 'LeftUpLeg')
# SurfacePoints('shinRight', 'limb', 'RightLeg')
# SurfacePoints('shinLeft', 'limb', 'LeftLeg')
# SurfacePoints('abdomenUp', 'mesh', 'Spine2')
# SurfacePoints('armRight', 'limb', 'RightArm')
# SurfacePoints('foreRight', 'limb', 'RightForeArm')
# SurfacePoints('armLeft', 'limb', 'LeftArm')
# SurfacePoints('foreLeft', 'limb', 'LeftForeArm')
# SurfacePoints('headRight', 'mesh', 'Head')
# SurfacePoints('headLeft', 'mesh', 'Head')
# SurfacePoints('earRight', 'mesh', 'Head')
# SurfacePoints('earLeft', 'mesh', 'Head')
# SurfacePoints('chinRight', 'mesh', 'Head')
# SurfacePoints('chinLeft', 'mesh', 'Head')
# SurfacePoints('cheekRight', 'mesh', 'Head')
# SurfacePoints('cheekLeft', 'mesh', 'Head')
# SurfacePoints('mouth', 'mesh', 'Head')
# SurfacePoints('foreHead', 'mesh', 'Head')
# SurfacePoints('backHeadRight', 'mesh', 'Head')
# SurfacePoints('backHeadLeft', 'mesh', 'Head')
# SurfacePoints('backHead', 'mesh', 'Head')
# SurfacePoints('loinRight', 'mesh', 'Hips')
# SurfacePoints('loinLeft', 'mesh', 'Hips')
# SurfacePoints('loinUp', 'mesh', 'Spine1')
