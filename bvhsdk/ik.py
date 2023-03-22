# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:06:32 2018

@author: Rodolfo Luis Tonoli
"""

"""
Para usar junto com o Samuel R. Buss
https://stackoverflow.com/questions/3993403/inverse-kinematics-calculating-the-jacobian
"""

import numpy as np
from . import mathutils



class SimpleJacobian:
    """
    Jacobian transpose method with only one end effector
    1 End effector
    N Joints
    3 DOF joints
    No orientation
    """

    def __init__(self, animation, end_effector, depth=-1):
        assert animation is not None, "'NoneType' Animation object"
        assert end_effector is not None, "'NoneType' End Effector Joint object"
        self.animation = animation
        self.end_effector = end_effector
        self.jacobian = []
        self.target = []
        self.dtheta = []
        self.currentframe = 0
        self.depth = depth


    def computeJacobian(self, referenceframe):
        """
        Create the jacobian VECTOR (one end effector)
        """
        jacobian = []
        for joint in self.end_effector.pathFromDepthToJoint(self.depth):
            transform = joint.getGlobalTransform(frame=referenceframe)
            vjx = transform[:-1,0]
            vjy = transform[:-1,1]
            vjz = transform[:-1,2]
            vjx = vjx/np.linalg.norm(vjx)
            vjy = vjy/np.linalg.norm(vjy)
            vjz = vjz/np.linalg.norm(vjz)
            position = transform[:-1,-1]
            j1 = np.cross(vjx, self.target-position)
            j2 = np.cross(vjy, self.target-position)
            j3 = np.cross(vjz, self.target-position)
            jacobian.append(j1)
            jacobian.append(j2)
            jacobian.append(j3)
        self.jacobian = np.asarray(jacobian).T


    def updateValues(self, rotnextframe = False):
        """
        Update the rotation values of the joints in the path from root to end effector.
        Construct a matrix with theta values and rotate (old) local matrix with it;
        Extract euler angles from this new rotation (local) matrix;
        Repeat for the other joints down the hierarchy.
        """
        i = 0
        for joint in self.end_effector.pathFromDepthToJoint(self.depth):
            drotation = mathutils.matrixR([self.dtheta[i*3],self.dtheta[i*3+1],self.dtheta[i*3+2]], joint.order)
            local = mathutils.shape4ToShape3(joint.getLocalTransform(self.currentframe))
            new_local = np.dot(local,drotation)
            new_theta, warning = mathutils.eulerFromMatrix(new_local, joint.order)
            if warning:
                print('Warning raised from mathutils.eulerFromMatrix() at jacobian.updateValues()')
            #joint.rotation[self.currentframe] = np.asarray(new_theta)
            joint.setLocalRotation(self.currentframe, np.asarray(new_theta))
            i = i + 1

            if rotnextframe:
                try:
                    #May be the last frame
                    next_frame_local = mathutils.shape4ToShape3(joint.getLocalTransform(self.currentframe + 1))
                    new_next_frame_local = np.dot(next_frame_local,drotation)
                    new_next_frame_theta, warning = mathutils.eulerFromMatrix(new_next_frame_local, joint.order)
                    if warning:
                        print('Warning raised from mathutils.eulerFromMatrix() at jacobian.updateValues()')
                    #joint.rotation[self.currentframe + 1] = np.asarray(new_next_frame_theta)
                    joint.setLocalRotation(self.currentframe + 1,np.asarray(new_next_frame_theta))
                except:
                    pass


    def jacobianTranspose(self, frame, target, rotnextframe = False, epsilon = 1, maxiter = 10):
        """
        Perform the Inverse Kinematics Jacobian Transpose.

        Calculate the Jacobian vector (only one end effector), calculate the
        rotations step, update the angles values and repeat it until the distance
        between the target and end effector is less than epsilon

        #Ignore
        :type lastframeref: bool
        :param lastframeref: if True, uses the previous frame as initial pose
        """
        self.target = target
        self.currentframe = frame
        e_norm = np.linalg.norm(self.target-self.end_effector.getPosition(frame=self.currentframe))
#        print('Frame: %i' % self.currentframe)
        count = 0
        log = ''
        while (e_norm > epsilon) and (count < maxiter):
            self.computeJacobian(self.currentframe)
            self.__transpose()
            self.updateValues(rotnextframe)
            e_norm = np.linalg.norm(self.target-self.end_effector.getPosition(frame=self.currentframe))
            count=count+1

        if e_norm <= epsilon:
            log = log + 'Target reached. '
        if count == maxiter:
            log = log + 'Max iteration reached.'
        return log


    def __transpose(self):
        """
        Transpose the jacobian calculated with computeJacobian()

        From Samuel R Buss code Jacobian.cpp -> Jacobian::CalcDeltaThetasTranspose()
        """
        if len(self.jacobian) == 0:
            print('First compute the jacobian')
        else:
            dtheta = []
            maxAngleChange = 30*np.pi/180

            e = self.target-self.end_effector.getPosition(frame=self.currentframe) #Distance
#            target = self.target
#            eepos = self.end_effector.getPosition(frame=self.currentframe)
#            jacobian = np.asarray(self.jacobian)
            dtheta2 = np.dot(self.jacobian.T, e)
            j_jt_e = np.dot(np.dot(self.jacobian, self.jacobian.T), e)
            alpha = np.dot(e,j_jt_e)/np.dot(j_jt_e,j_jt_e)
            beta = maxAngleChange/np.linalg.norm(dtheta2)
            dtheta2 = dtheta2*np.min([alpha,beta])*180/np.pi
            for i in range(0,self.jacobian.shape[1]):
                #i = current rotation axis being used
                j_jt_e = np.dot(np.dot(self.jacobian[:,i], self.jacobian[:,i].T), e)
                alpha = np.dot(e,j_jt_e)/np.dot(j_jt_e,j_jt_e)
                beta = maxAngleChange
                dtheta.append(alpha*np.dot(self.jacobian[:,i].T, e)*180/np.pi)
#            self.dtheta = dtheta[:]
            self.dtheta=dtheta2



#    def pseudoInverse(self, frame=0):
#        if len(self.jacobian) == 0:
#            print('First compute the jacobian')
#        self.jacobian = []

#        TODO: estou usando
#        https://stackoverflow.com/questions/3993403/inverse-kinematics-calculating-the-jacobian
#        https://inst.eecs.berkeley.edu/~cs294-13/fa09/lectures/294-lecture19.pdf
#        http://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
#    https://www.proun-game.com/Oogst3D/ARTICLES/InverseKinematics.pdf
#        IK.pdf
#        IK2.pdf dropbox





#        TODO: OLHAR
#        https://stackoverflow.com/questions/30147263/7-dof-inverse-kinematic-with-jacobian-and-pseudo-inverse?rq=1
#        https://cs.uwaterloo.ca/research/tr/2000/19/CS-2000-19.pdf
