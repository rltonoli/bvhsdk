# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:29:18 2018

@author: Rodolfo Luis Tonoli
"""

import numpy as np
import time



def xaxis():
    """
    :return: Return a numpy array representing the x axis
    :rtype: numpy.ndarray
    """
    return np.asarray([1,0,0])

def yaxis():
    """
    :return: Return a numpy array representing the y axis
    :rtype: numpy.ndarray
    """
    return np.asarray([0,1,0])

def zaxis():
    """
    :return: Return a numpy array representing the z axis
    :rtype: numpy.ndarray
    """
    return np.asarray([0,0,1])

def matrixIdentity(shape=3):
    """
    Return identity matrix of shape 3x3 or 4x4
    
    :param int shape: shape of the matrix. 3 for 3x3 and 4 for 4x4
    
    :return: identity matrix
    :rtype: numpy.ndarray
    """
    if shape==3:
        matrix = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
                ])
    elif shape==4:
        matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])
    else:
        print("Incompatible shape, please choose 3 or 4.")
        return None
    return matrix


def matrixTranslation(tx,
                      ty,
                      tz):
    """
    Construct a transformation matrix for translation of tx, ty and tz

    :param float tx: translation in x axis
    :param float ty: translation in y axis
    :param float tz: translation in z axis

    :return: translation matrix
    :rtype: numpy.ndarray
    """
    matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
            ])
    return matrix

def matrixRotation(angle,
                   x=0,
                   y=0,
                   z=0,
                   shape=4):
    """
    Construct a transformation matrix for rotation of angle degrees around the axis defined by x, y and z.
    If shape is 4 it will append create a 4x4 transformation matrix with translation equals to zero.

    :param float angle: angle in degrees
    :param float x: x axis
    :param float y: y axis
    :param float z: z axis
    :param int shape: shape of the matrix. 3 for 3x3 and 4 for 4x4

    :return: rotation matrix
    :rtype: numpy.ndarray
    """

    if x==0 and y==0 and z==0:
        print("No axis found. Values x, y and z can't be all zero.")
        return None

    if shape!=3 and shape!=4:
        print("Incompatible shape. Please choose 3 or 4")
        return None

    length = np.sqrt(x*x+y*y+z*z)
    if length>0:
        x=x/length
        y=y/length
        z=z/length

    cosi = np.cos(angle*np.pi/180)
    sine = np.sin(angle*np.pi/180)

    matrix = np.zeros([4,4])
    matrix[0,0] = cosi + x*x*(1-cosi)
    matrix[0,1] = x*y*(1-cosi) - z*sine
    matrix[0,2] = x*z*(1-cosi) + y*sine
    matrix[0,3] = 0

    matrix[1,0] = x*y*(1-cosi) + z*sine
    matrix[1,1] = cosi + y*y*(1-cosi)
    matrix[1,2] = y*z*(1-cosi) - x*sine
    matrix[1,3] = 0

    matrix[2,0] = x*z*(1-cosi) - y*sine
    matrix[2,1] = y*z*(1-cosi) + x*sine
    matrix[2,2] = cosi + z*z*(1-cosi)
    matrix[2,3] = 0

    matrix[3,0] = 0
    matrix[3,1] = 0
    matrix[3,2] = 0
    matrix[3,3] = 1

    if shape==3:
        matrix = shape4ToShape3(matrix)

    return matrix



def matrixMultiply(m0,
                   m1):
    """
    Perform matrix multiplication between m0 and m1. m0 and m1 can be 3x3 or 4x4 matrices.
    For performance reasons, use numpy.dot() instead.

    :param numpy.ndarray m0: matrix 0
    :param numpy.ndarray m1: matrix 1

    :return: matrix resulting from the multiplication
    :rtype: numpy.ndarray
    """
    start=time.time()

    if type(m0)!=type(np.ndarray([])):
        try:
            m0 = np.asarray(m0)
        except:
            print("First argument could not be converted to a numpy array.")
            return None

    if type(m1)!=type(np.ndarray([])):
        try:
            m1 = np.asarray(m1)
        except:
            print("Second argument could not be converted to a numpy array.")
            return None

    if m0.shape[1]!=m1.shape[0]:
        print("The matrices cannot be multipled. Incompatible shapes.")
        return None

    if m0.shape==m1.shape:
        if m0.shape[1]==3 and m1.shape[0]==3:
            matrix = np.zeros([3,3])
            matrix[0,0] = m0[0,0]*m1[0,0] + m0[0,1]*m1[1,0] + m0[0,2]*m1[2,0]
            matrix[0,1] = m0[0,0]*m1[0,1] + m0[0,1]*m1[1,1] + m0[0,2]*m1[2,1]
            matrix[0,2] = m0[0,0]*m1[0,2] + m0[0,1]*m1[1,2] + m0[0,2]*m1[2,2]

            matrix[1,0] = m0[1,0]*m1[0,0] + m0[1,1]*m1[1,0] + m0[1,2]*m1[2,0]
            matrix[1,1] = m0[1,0]*m1[0,1] + m0[1,1]*m1[1,1] + m0[1,2]*m1[2,1]
            matrix[1,2] = m0[1,0]*m1[0,2] + m0[1,1]*m1[1,2] + m0[1,2]*m1[2,2]

            matrix[2,0] = m0[2,0]*m1[0,0] + m0[2,1]*m1[1,0] + m0[2,2]*m1[2,0]
            matrix[2,1] = m0[2,0]*m1[0,1] + m0[2,1]*m1[1,1] + m0[2,2]*m1[2,1]
            matrix[2,2] = m0[2,0]*m1[0,2] + m0[2,1]*m1[1,2] + m0[2,2]*m1[2,2]


        elif m0.shape[1]==4 and m1.shape[0]==4:
            matrix = np.zeros([4,4])
            matrix[0,0] = m0[0,0]*m1[0,0] + m0[0,1]*m1[1,0] + m0[0,2]*m1[2,0] + m0[0,3]*m1[3,0]
            matrix[0,1] = m0[0,0]*m1[0,1] + m0[0,1]*m1[1,1] + m0[0,2]*m1[2,1] + m0[0,3]*m1[3,1]
            matrix[0,2] = m0[0,0]*m1[0,2] + m0[0,1]*m1[1,2] + m0[0,2]*m1[2,2] + m0[0,3]*m1[3,2]
            matrix[0,3] = m0[0,0]*m1[0,3] + m0[0,1]*m1[1,3] + m0[0,2]*m1[2,3] + m0[0,3]*m1[3,3]

            matrix[1,0] = m0[1,0]*m1[0,0] + m0[1,1]*m1[1,0] + m0[1,2]*m1[2,0] + m0[1,3]*m1[3,0]
            matrix[1,1] = m0[1,0]*m1[0,1] + m0[1,1]*m1[1,1] + m0[1,2]*m1[2,1] + m0[1,3]*m1[3,1]
            matrix[1,2] = m0[1,0]*m1[0,2] + m0[1,1]*m1[1,2] + m0[1,2]*m1[2,2] + m0[1,3]*m1[3,2]
            matrix[1,3] = m0[1,0]*m1[0,3] + m0[1,1]*m1[1,3] + m0[1,2]*m1[2,3] + m0[1,3]*m1[3,3]

            matrix[2,0] = m0[2,0]*m1[0,0] + m0[2,1]*m1[1,0] + m0[2,2]*m1[2,0] + m0[2,3]*m1[3,0]
            matrix[2,1] = m0[2,0]*m1[0,1] + m0[2,1]*m1[1,1] + m0[2,2]*m1[2,1] + m0[2,3]*m1[3,1]
            matrix[2,2] = m0[2,0]*m1[0,2] + m0[2,1]*m1[1,2] + m0[2,2]*m1[2,2] + m0[2,3]*m1[3,2]
            matrix[2,3] = m0[2,0]*m1[0,3] + m0[2,1]*m1[1,3] + m0[2,2]*m1[2,3] + m0[2,3]*m1[3,3]

            matrix[3,0] = m0[3,0]*m1[0,0] + m0[3,1]*m1[1,0] + m0[3,2]*m1[2,0] + m0[3,3]*m1[3,0]
            matrix[3,1] = m0[3,0]*m1[0,1] + m0[3,1]*m1[1,1] + m0[3,2]*m1[2,1] + m0[3,3]*m1[3,1]
            matrix[3,2] = m0[3,0]*m1[0,2] + m0[3,1]*m1[1,2] + m0[3,2]*m1[2,2] + m0[3,3]*m1[3,2]
            matrix[3,3] = m0[3,0]*m1[0,3] + m0[3,1]*m1[1,3] + m0[3,2]*m1[2,3] + m0[3,3]*m1[3,3]

    else:
        if m0.shape==(3,3) and m1.shape[0]==3:
            matrix = np.zeros([3,1])
            matrix[0] = m0[0,0]*m1[0] + m0[0,1]*m1[1] + m0[0,2]*m1[2]
            matrix[1] = m0[1,0]*m1[0] + m0[1,1]*m1[1] + m0[1,2]*m1[2]
            matrix[2] = m0[2,0]*m1[0] + m0[2,1]*m1[1] + m0[2,2]*m1[2]

        elif m0.shape==(4,4) and m1.shape[0]==4:
            matrix = np.zeros([4,1])
            matrix[0] = m0[0,0]*m1[0] + m0[0,1]*m1[1] + m0[0,2]*m1[2] + m0[0,3]*m1[3]
            matrix[1] = m0[1,0]*m1[0] + m0[1,1]*m1[1] + m0[1,2]*m1[2] + m0[1,3]*m1[3]
            matrix[2] = m0[2,0]*m1[0] + m0[2,1]*m1[1] + m0[2,2]*m1[2] + m0[2,3]*m1[3]
            matrix[3] = m0[3,0]*m1[0] + m0[3,1]*m1[1] + m0[3,2]*m1[2] + m0[3,3]*m1[3]

        else:
            print("The matrices cannot be multipled. Incompatible shapes.")
            return None


    matrixMultiply.time += time.time()-start
    matrixMultiply.count += 1
    return matrix


def inverseMatrix(m0):
    """
    Tries to invert matrix m0 using numpy.linalg.inv(). If it is not possible, returns None.

    :param numpy.ndarray m0: matrix to be inverted

    :return: inverted matrix
    :rtype: numpy.ndarray
    """
    try:
        matrix = np.linalg.inv(m0)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        print('Matrix not invertible.')
        return None
    return matrix


def projectedBarycentricCoord(p,
                              p1,
                              p2,
                              p3):
    """
    Compute the baricentric coordinates of a point p projected onto a triangle defined by p1, p2 and p3.
    Algorithm from "Computing the barycentric coordinates of a projected point." by Heidrich, Wolfgang, Journal of Graphics Tools 10, no. 3 (2005): 9-12.
    
    :param numpy.ndarray p: point to be projected
    :param numpy.ndarray p1: first point of the triangle
    :param numpy.ndarray p2: second point of the triangle
    :param numpy.ndarray p3: third point of the triangle

    :return: normal vector, baricentric coordinates, displacement vector, baricentric coordinates in cartesian space and a boolean indicating if the point is inside the triangle
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, bool
    """
    start = time.time()
    
    if len(p)>3: p=p[:3]
    if len(p1)>3: p1=p1[:3]
    if len(p2)>3: p2=p2[:3]
    if len(p3)>3: p3=p3[:3]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    inside = False

    u = p2-p1
    v = p3-p1
    b = np.zeros(3)
    #Algorithm:
    # n = unitVector(np.cross( u, v ))
    n = np.cross( u, v )
    oneOver4ASquared = 1.0 / np.dot( n, n )
    w = p - p1
    b[2]= np.dot( np.cross( u, w ), n ) * oneOver4ASquared
    b[1]= np.dot( np.cross( w, v ), n ) * oneOver4ASquared
    b[0]= 1.0 - b[1] - b[2]
    if b[0]>=0 and b[0]<=1:
        if b[1]>=0 and b[1]<=1:
            if b[2]>=0 and b[2]<=1:
                inside = True
    #b = b/np.linalg.norm(b)
    b_cartesian = b[0]*p1 + b[1]*p2 + b[2]*p3
    dispvector = p-b_cartesian
    n = n/np.linalg.norm(n)

    projectedBarycentricCoord.count += 1
    projectedBarycentricCoord.time += time.time()-start
    return n, b, dispvector, b_cartesian, inside

def clampedBarycentric(p,
                       p1,
                       p2,
                       p3):
    """
    Compute the baricentric coordinates of a point p projected onto a triangle defined by p1, p2 and p3.
    Different from mathutils.projectedBarycentricCoord(), if p is outside the triangle, the baricentric coordinates are clamped to the triangle edges.
    Algorithm from "Computing the barycentric coordinates of a projected point." by Heidrich, Wolfgang, Journal of Graphics Tools 10, no. 3 (2005): 9-12.
    
    :param numpy.ndarray p: point to be projected
    :param numpy.ndarray p1: first point of the triangle
    :param numpy.ndarray p2: second point of the triangle
    :param numpy.ndarray p3: third point of the triangle

    :return: normal vector, baricentric coordinates, displacement vector, baricentric coordinates in cartesian space and a boolean indicating if the point is inside the triangle
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, bool
    """
    start = time.time()
    if len(p)>3: p=p[:3]
    if len(p1)>3: p1=p1[:3]
    if len(p2)>3: p2=p2[:3]
    if len(p3)>3: p3=p3[:3]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    inside = False

    u = p2-p1
    v = p3-p1
    b = np.zeros(3)
    #Algorithm:
    # n = unitVector(np.cross( u, v ))
    n = np.cross( u, v )
    oneOver4ASquared = 1.0 / np.dot( n, n )
    w = p - p1
    b[2]= np.dot( np.cross( u, w ), n ) * oneOver4ASquared
    b[1]= np.dot( np.cross( w, v ), n ) * oneOver4ASquared
    b[0]= 1.0 - b[1] - b[2]
    if b[0]>=0 and b[0]<=1:
        if b[1]>=0 and b[1]<=1:
            if b[2]>=0 and b[2]<=1:
                inside = True
    b = np.clip(b,0,np.inf)
    b = b/np.sum(b)
    b_cartesian = b[0]*p1 + b[1]*p2 + b[2]*p3
    dispvector = p-b_cartesian
    n = n/np.linalg.norm(n)

    clampedBarycentric.count += 1
    clampedBarycentric.time += time.time()-start
    return n, b, dispvector, b_cartesian, inside

def barycentric2cartesian(bary,
                          v1,
                          v2,
                          v3):
    """
    Convert baricentric coordinates to cartesian coordinates

    :param numpy.ndarray bary: baricentric coordinates
    :param numpy.ndarray v1: first point of the triangle
    :param numpy.ndarray v2: second point of the triangle
    :param numpy.ndarray v3: third point of the triangle

    :return: cartesian coordinates, normal vector
    :rtype: numpy.ndarray, numpy.ndarray
    """
    if len(bary)>3: bary=bary[:3]
    if len(v1)>3: v1=v1[:3]
    if len(v2)>3: v2=v2[:3]
    if len(v3)>3: v3=v3[:3]
    u = v2-v1
    v = v3-v1
    n = np.cross( u, v )
    cart = np.zeros(3)
    cart = bary[0]*v1 + bary[1]*v2 + bary[2]*v3
    return cart, n

def getCentroid(p1,
                p2,
                p3):
    """
    Compute the centroid of a triangle defined by p1, p2 and p3.

    :param numpy.ndarray p1: first point of the triangle
    :param numpy.ndarray p2: second point of the triangle
    :param numpy.ndarray p3: third point of the triangle

    :return: centroid, normal vector
    :rtype: numpy.ndarray, numpy.ndarray
    """
    if len(p1)>3: p1=p1[:3]
    if len(p2)>3: p2=p2[:3]
    if len(p3)>3: p3=p3[:3]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    u = p2-p1
    v = p3-p1
    n = unitVector(np.cross( u, v ))
    return np.mean([p1,p2,p3], axis=0), n

def distFromCentroid(p,
                     p1,
                     p2,
                     p3):
    """
    Compute the distance from a point p to the centroid of a triangle defined by p1, p2 and p3.

    :param numpy.ndarray p: point of interest
    :param numpy.ndarray p1: first point of the triangle
    :param numpy.ndarray p2: second point of the triangle
    :param numpy.ndarray p3: third point of the triangle
    
    :return: centroid, distance vector, normal vector
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    if len(p)>3: p=p[:3]
    centroid, n = getCentroid(p1,p2,p3)
    distance = p-centroid
    return centroid, distance ,n



def matrixSkew(vec):
    """
    Construct a skew-symmetric matrix from a vector. If the vector is 3D, the matrix will be 3x3. If the vector is 4D, the matrix will be 4x4.

    :param numpy.ndarray vec: vector to be converted to skew-symmetric matrix

    :return: skew-symmetric matrix
    :rtype: numpy.ndarray
    """
    if vec.shape[0] == 4:
        matrix = np.array([
                [0,       -vec[2], vec[1],  0],
                [vec[2],  0,       -vec[0], 0],
                [-vec[1], vec[0],  0,       0],
                [0,       0,       0,       1]
                ])
    elif vec.shape[0] == 3:
        matrix = np.array([
            [0,       -vec[2], vec[1]],
            [vec[2],  0,       -vec[0]],
            [-vec[1], vec[0],  0]
            ])
    else:
        print('Wrong vector shape, expected 3.')
        return None
    return matrix

def alignVectors(a,
                 b,
                 shape=3):
    """
    Returns a rotation matrix to align vector a onto vector b.
    This matrix is not constructed as RxRyRz, but from an axis-angle representation.
    
    Based on the algorithm available at https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    :param numpy.ndarray a: vector to be aligned
    :param numpy.ndarray b: vector to be aligned to

    :return: rotation matrix
    :rtype: numpy.ndarray
    """

    start=time.time()
    if (shape != 3) and (shape != 4):
        print('Shape %i not supported to represent a rotation matrix at mathutils.alignVectores(). Please choose 3 or 4.' % shape)
        return matrixIdentity(3)
    a_norm = np.asarray(a / np.linalg.norm(a))
    b_norm = np.asarray(b / np.linalg.norm(b))
    aux_vector = np.cross(a_norm, b_norm)
#    sin = np.linalg.norm(aux_vector)
    cos = np.dot(a_norm, b_norm)
    #This approach fails when the vectors point into exactly the same or opposite directions
    if cos == -1:
        #If they point into the same direction, return identity matrix
        if (a_norm == b_norm).all():
            return matrixIdentity(shape)
        #If they point into the
        elif (a_norm == -b_norm).all():
            #Hardcoded case when they point into opposite directions of y
            dot_a = np.dot(a_norm,[0,1,0])
            dot_b = np.dot(b_norm,[0,1,0])
            if (dot_a == 1 or dot_a == -1) and (dot_b == 1 or dot_b == -1):
                return matrixRotation(180,0,0,1, shape)

        print('Invalid vectors in mathutils.alignVectors(a,b). Vectors:')
        print(a)
        print(b)
        print('Identity matrix returned')
        return matrixIdentity(shape)
    rotationMatrix = matrixIdentity(3)
    skew = matrixSkew(aux_vector)
    matrix = rotationMatrix + skew + np.dot(skew,skew)*(1/(1+cos))
    if shape==4:
        matrix = shape3ToShape4(matrix)

    alignVectors.time += time.time()-start
    alignVectors.count += 1
    return matrix

def angleBetween(a,
                 b):
    """
    Returns the euler angle between the vectors (from a to b) and the rotation axis
    https://stackoverflow.com/questions/15101103/euler-angles-between-two-3d-vectors
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/index.htm
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions

    :param numpy.ndarray a: vector to be aligned
    :param numpy.ndarray b: vector to be aligned to

    :return: angle between the vectors, axis of rotation
    :rtype: float, numpy.ndarray
    """
    start = time.time()
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    axis = np.cross(a_norm, b_norm)
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(a_norm, b_norm))
    angleBetween.time += time.time() - start
    angleBetween.count += 1
    return angle, axis_norm
    #axisAngleToEuler(aux_vector_norm,angle)

#def axisAngleToEuler(axis,angle):

def multiInterp(x,
                xp,
                arr):
    """
    Performs a linear interpolation like numpy.interp() but for multidimensional arrays. Interpolation is performed independently for each dimension.
    
    :param numpy.ndarray x: The x-coordinates at which to evaluate the interpolated values (same for all dimensions)
    :param numpy.ndarray xp: The x-coordinates of the data points (same for all dimensions)
    :param numpy.ndarray arr: Multi-dimensional array of data with the same length as xp along the interpolation axis.

    :return: interpolated array
    :rtype: numpy.ndarray
    """
    return np.array([np.interp(x, xp, arr[:,i]) for i in range(arr.shape[1])]).T

def isnear(x,y, epsilon = 1e-4):
    if abs(x-y) < epsilon:
        return True
    else:
        return False

def isequal(a1, a2, epsilon=1e-4):
    """
    Check if array v1 and array v2 are equal element-wise

    :type a1: numpy.ndarray
    :param a1: First array

    :type a2: numpy.ndarray
    :param a2: Second array

    :type epsilon: numpy.ndarray
    :param epsilon: Error margin
    """
    a1=np.asarray(a1)
    a2=np.asarray(a2)
    if (a1.ndim != a2.ndim):
        raise ValueError('v1 and v2 must have the same dimension.')
    if a1.ndim > 3 or a1.ndim <= 0:
        raise ValueError('Vectors dimension must be 1, 2 or 3.')
    else:
        for i in range(a1.ndim):
            if a1.shape[i] != a2.shape[i]:
                raise ValueError('Vectors must have the same length.')

    equal = [isnear(a1.flat[i],a2.flat[i], epsilon) for i in range(a1.size)]
    if False in equal:
        return False
    else:
        return True

def unitVector(vec):
    return vec/np.linalg.norm(vec)

def isRotationMatrix(matrix):
    """
    Check if matrix is a valid rotation matrix
    """
    if type(matrix)!=np.ndarray:
        try:
            matrix = np.asarray(matrix)
        except:
            print('Could not convert matrix to numpy array')
            return None
    if matrix.shape[0] == 4:
        matrix = shape4ToShape3(matrix)
    matrix_T = np.transpose(matrix)
    shouldBeIdentity = np.dot(matrix_T, matrix)
    I = matrixIdentity(shape=3)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def isNear(value, constant, e=1e-6):
    """
    Test if value is near the constant inside the range giving by e
    constant - e <= value <= constant + e
    """
    if value >= constant-e and value <= constant+e:
        return True
    else:
        return False

def eulerFromMatrix(matrix, order='ZXY'):
    """
    Return the euler angles
    from Computing Euler angles from a rotation matrix by Gregory G. Slabaugh
    available online at: http://www.close-range.com/docs/Computing_Euler_angles_from_a_rotation_matrix.pdf
    Other references:
    https://en.wikipedia.org/wiki/Euler_angles#Extrinsic_rotations
    https://gist.github.com/crmccreary/1593090
    https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
    NOTE: This function will work only with matrices created using RxRyRz

    order = 'XYZ'
    ((RxRy)Rz) =
        cos(y)          -sen(z)cos(y)                       sen(y)
        sen(x)sen(y)    -sen(x)sen(y)sen(z)+cos(x)cos(z)    -sen(x)cos(y)
        -sen(y)cos(x)   sen(y)cos(x)sen(z)+sen(x)cos(z)     cos(x)cos(y)

    order = 'ZXY'
    ((RzRx)Ry) =
        cos(y)cos(z)-sen(x)sen(y)sen(z)     -cos(x)sen(z)   cos(z)sen(y)+sen(x)sen(z)cos(y)
        cos(y)sen(z)+sen(x)sen(y)cos(z)     cos(x)cos(z)    sen(y)sen(z)-sen(x)cos(y)cos(z)
        -cos(x)sen(y)                       sen(x)          cos(x)cos(y)

        if cosX = 0 and sen(x) = -1:
            x = -pi/2
            matrix[0,0] = cy*cz+sy*sz = cos(z-y)
            matrix[1,0] = cy*sz-sy*cz = sen(z-y)
            matrix[0,2] = sy*cz-cy*sz = -sen(z-y) = -matrix[1,0]
            matrix[1,2] = sy*sz+cy*cz = cos(z-y) = matrix[0,0]
            z - y = atan2(-matrix[1,0], matrix[0,0])
            z = y + atan2(-matrix[1,0], matrix[0,0])
        if cosX = 0 and sen(x) = 1:
            x = pi/2
            matrix[0,0] = cy*cz-sy*sz = cos(y+z)
            matrix[1,0] = cy*sz+sy*cz = sen(y+z)
            matrix[0,2] = sy*cz+sz*cy = sen(y+z) = matrix[1,0]
            matrix[1,2] = sy*sz-cy*cz = -cos(y+z) = -matrix[0,0]
            z + y = atan2(matrix[1,0], -matrix[0,0])
            z = -y + atan2(matrix[1,0], -matrix[0,0])

    order = 'ZYX'
    ((RzRx)Ry) =
        cos(y)cos(z)    sin(x)sin(y)cos(z)-cos(x)sen(z)   cos(x)sin(y)cos(z)+sen(x)sen(z)
        cos(y)sin(z)    sin(x)sin(y)sin(z)+cos(x)cos(z)   cos(x)sin(y)sin(z)-sen(x)cos(z)
        -sin(y)         sin(x)cos(y)                      cos(x)cos(y)

    :type matrix: numpy.ndarray
    :param matrix: 3x3 rotation matrix or 4x4 transform matrix

    :type order: string
    :param order: order of rotation (working only for ZXY)
    """
    start= time.time()
    warning = None
    if type(matrix)!=np.ndarray:
        try:
            matrix = np.asarray(matrix)
        except:
            print('Could not convert matrix to numpy array')
            return None

    if matrix.shape[0] == 4:
        matrix = shape4ToShape3(matrix)

#    if order == 'ZXY':
#        x = np.arcsin(matrix[2,1])
#        y = np.arcsin(-matrix[2,0]/np.cos(x))
#        z = np.arcsin(-matrix[0,1]/np.cos(x))
#        if np.cos(x) < 0.01:
#            warning = True
    if order == 'ZXY':
        if not isNear(matrix[2,1],1) and not isNear(matrix[2,1], -1):
            x1 = np.arcsin(matrix[2,1])
            #sin(pi-theta) = sin(theta)
            # x2 = np.pi - x1
            y1 = np.arctan2(-matrix[2,0]/np.cos(x1),matrix[2,2]/np.cos(x1))
            # y2 = np.arctan2(-matrix[2,0]/np.cos(x2),matrix[2,2]/np.cos(x2))
            z1 = np.arctan2(-matrix[0,1]/np.cos(x1),matrix[1,1]/np.cos(x1))
            # z2 = np.arctan2(-matrix[0,1]/np.cos(x2),matrix[1,1]/np.cos(x2))
        else:
            warning = True
            if isNear(matrix[2,1],-1):
                x1 = -np.pi/2
                y1 = 0
                z1 = np.arctan2(-matrix[1,0], matrix[0,0]) #(+y)
            elif isNear(matrix[2,1],1):
                x1 = np.pi/2
                y1 = 0
                z1 = np.arctan2(matrix[1,0], -matrix[0,0]) #(-y)
            else:
                x1,y1,z1=0,0,0

    elif order == 'ZYX':
        if not isNear(matrix[2,0],1) and not isNear(matrix[2,0], -1):
            y1 = -np.arcsin(matrix[2,0])
            #sin(pi-theta) = sin(theta)
            #y2 = np.pi - y1
            x1 = np.arctan2(matrix[2,1]/np.cos(y1),matrix[2,2]/np.cos(y1))
            #x2 = np.arctan2(matrix[2,1]/np.cos(y2),matrix[2,2]/np.cos(y2))
            z1 = np.arctan2(matrix[1,0]/np.cos(y1),matrix[0,0]/np.cos(y1))
            #z2 = np.arctan2(matrix[1,0]/np.cos(y2),matrix[0,0]/np.cos(y2))
        else:
            warning = True
            z1 = 0
            if isNear(matrix[2,0],-1):
                y1 = np.pi/2
                x1 = np.arctan2(matrix[0,1], matrix[0,2])
            elif isNear(matrix[2,0],1):
                y1 = -np.pi/2
                x1 = np.arctan2(-matrix[0,1], -matrix[0,2])

    #TODO: Corrigir
    elif order == 'XYZ':
        y1 = np.arccos(matrix[0,0])
        x1 = np.arccos(matrix[2,2]/matrix[0,0])
        z1 = np.arcsin(-matrix[1,0]/matrix[0,0])
        x2,y2,z2 =0,0,0
        if matrix[0,0] < 0.01:
            warning = True
        raise ValueError('Order %s not implemented yet.' % order)

#    return np.asarray([x1,y1,z1])*180/np.pi, np.asarray([x2,y2,z2])*180/np.pi, warning

    eulerFromMatrix.count+=1
    eulerFromMatrix.time+=time.time()-start
    return np.asarray([x1,y1,z1])*180/np.pi, warning


def eulerFromVector(vector, twistaxis = 'y'):

    #TODO: FAZER OS OUTROS CASOS
    start=time.time()
    vector = unitVector(np.asarray(vector))
    xunit = np.asarray([1,0,0])
    yunit = np.asarray([0,1,0])
    zunit = np.asarray([0,0,1])
    x,y,z = vector[0],vector[1],vector[2]
    if twistaxis=='y':
        dotx = np.dot(vector, unitVector([x,y,0]))
        if dotx==0:
            print('1')
            thetaz = np.arccos(np.dot(vector, [0,1,0]))
            return 0,-thetaz
        else:
            thetax = np.arccos(dotx)

        dotz = np.dot([x,y,0], [0,1,0])
#        if dotz==0:
#            print('2')
#            thetax = np.arccos(np.dot(vector, [0,1,0]))
#            return thetax,0
#        else:
        thetaz = np.arccos(dotz)
    #elif twistaxis....
    eulerFromVector.count+=1
    eulerFromVector.time+=time.time()-start
    return -thetax,-thetaz


def matrixR(R, order='ZXY', shape=3):
    """
    Creates rotation matrix

     Parameters
    ----------
    R : numpy array
        Size 3 vector containing the rotation over the x-, y- and z-axis
    order : string, optional
        Order to create the rotation matrix. Only 'ZXY' and 'XYZ' implemented. The default is 'ZXY'.
    shape : TYPE, optional
        Matrix shape, 3x3 or 4x4. The default is 3.

    Returns
    -------
    matrix : numpy array
        Rotation matrix.
    """
    if type(R)!=np.ndarray:
        try:
            R = np.asarray(R)
        except:
            print('Could not convert rotation vector to numpy array at mathutils.matrixR()')
            return None
    if R.shape[0] != 3:
        print('Wrong rotation vector shape. Shape = %f and not 3' % R.shape[0])
        return None

    rotx = matrixRotation(R[0],1,0,0)
    roty = matrixRotation(R[1],0,1,0)
    rotz = matrixRotation(R[2],0,0,1)
    if order == "ZXY":
        matrix = np.dot(rotz, rotx)
        matrix = np.dot(matrix, roty)
    elif order == "XYZ":
        matrix = np.dot(rotx, roty)
        matrix = np.dot(matrix, rotz)
    elif order == "ZYX":
        matrix = np.dot(rotz, roty)
        matrix = np.dot(matrix, rotx)
    else:
        print('mathutils.matrixR does not accept rotation order %s' %order)
    if shape == 3:
        matrix = shape4ToShape3(matrix)
    return matrix


def transformMatrix(R, T, order='ZXY'):
    """
    Creates transform matrix.

    R: Size 3 vector containing the rotation over the x-, y- and z-axis
    T: Size 3 vector containing x, y and z translation
    """
    if type(R)!=np.ndarray or type(T)!=np.ndarray:
        try:
            R = np.asarray(R)
        except:
            print('Could not convert rotation vector to numpy array')
            return None
        try:
            T = np.asarray(T)
        except:
            print('Could not convert translation vector to numpy array')
            return None
    if R.shape[0] != 3:
        print('Wrong rotation vector shape. Shape = %f and not 3' % R.shape[0])
        return None
    if T.shape[0] != 3:
        print('Wrong rotation vector shape. Shape = %f and not 3' % R.shape[0])
        return None
    matrix = matrixTranslation(T[0], T[1], T[2])
    matrix = np.dot(matrix, matrixR(R, order, shape=4))
    return matrix

def shape4ToShape3(matrix):
    """
    Get the rotation matrix from the transform matrix

    matrix: 4x4 transform matrix
    """
    if type(matrix)!=np.ndarray:
        try:
            matrix = np.asarray(matrix)
        except:
            print('Could not convert matrix to numpy array')
            return None
    if matrix.shape[0] != 4 and matrix.shape[1] != 4:
        if matrix.shape[0] == 3 and matrix.shape[1] == 3:
            print('3x3 matrix, expected 4x4')
            return matrix
        else:
            print('Wrong shape matrix, expected 4x4')
            return None
    new_matrix = np.array([
                            [matrix[0,0],matrix[0,1],matrix[0,2]],
                            [matrix[1,0],matrix[1,1],matrix[1,2]],
                            [matrix[2,0],matrix[2,1],matrix[2,2]]
                        ])
    return new_matrix


def shape3ToShape4(matrix, T=None):
    """
    Expand 3x3 matrix to 4x4

    matrix: 3x3 matrix
    T: translation vector
    """
    if type(matrix) != np.ndarray:
        try:
            matrix = np.asarray(matrix)
        except:
            print('Could not convert matrix to numpy array')
            return None
    if matrix.shape[0] != 4 and matrix.shape[1] != 4:
        if matrix.shape[0] != 3 and matrix.shape[1] != 3:
            print('Unknown matrix shape')
            return matrix
    else:
        return matrix
    if not T:
        T = [0, 0, 0]
    else:
        if T.shape[0] != 3:
            print('Wrong translation vector shape. Shape = %f and not 3' % T.shape[0])
            return None
    new_matrix = np.array([
                            [matrix[0, 0], matrix[0, 1], matrix[0, 2], T[0]],
                            [matrix[1, 0], matrix[1, 1], matrix[1, 2], T[1]],
                            [matrix[2, 0], matrix[2, 1], matrix[2, 2], T[2]],
                            [0, 0, 0, 1]
                        ])
    return new_matrix


def matrixError(m1, m2):
    """
    Calcula a diferença entre as duas matrizes.

    :type m1: numpy.ndarray
    :param m1: Matrix to subtract

    :type m2: numpy.ndarray
    :param m2: Matrix to be subtracted
    """
    return m2-m1


def projection(a, b):
    """
    Compute the projection of a onto b.
    Exemple: https://math.boisestate.edu/~jaimos/notes/275/vector-proj.html

    :type a: numpy.ndarray
    :param a: Vector to be projected onto the other vector

    :type b: numpy.ndarray
    :param b: Projection target vector
    """
    return (np.dot(a,b)/np.dot(b,b))*b



def capsuleCollision(point, p0, p1, capradius):
    """
    Creates a capsule (extruded sphere): a cylinder with radius capradius and two half spheres (top and bottom) with radius capradius
    Returns the normalized intersection LOCAL cylindric coordinates and the intersection GLOBAL euclidean coordinates

    Surface: x^2+y^2+(1/4)*(|z-caplength|+|z+caplength| - 2*caplength)^2 - capradius^2 = 0

    Example:
    caplength = 10
    capradius = 2
    if z=5:
        x^2+y^2+(1/4)*(|5-10|+|5+10| - 2*10)^2 - 2^2 = 0
        x^2+y^2+(1/4)*(20 - 20)^2 - 2^2 = 0
        x^2+y^2 - 4 = 0
        if x^2+y^2 = 4:
            (On the surface of the cylinder)
        elif x^2+y^2 > 4:
            (Outside cylinder)
        elif x^2+y^2 < 4:
            (Inside cylinder)
    if z=15:
        x^2+y^2+(1/4)*(|15-10|+|15+10| - 2*10)^2 - 2^2 = 0
        x^2+y^2+(1/4)*(30 - 20)^2 - 2^2 = 0
        x^2+y^2+ 25 - 2^2 = 0
        x^2+y^2+ 21 > 0
        (Outside cylinder)
    if z=11:
        x^2+y^2+(1/4)*(|11-10|+|11+10| - 2*10)^2 - 2^2 = 0
        x^2+y^2+(1/4)*(22 - 20)^2 - 2^2 = 0
        x^2+y^2+ 1 - 4 = 0
        x^2+y^2 - 3 = 0
        (Analogous to z=5)
    Finding surface starting from [0,0,0] and moving along direction=[a,b,c] with step=t
    (a*t)^2+(b*t)^2+(1/4)*(|(z*t)-caplength|+|(z*t)+caplength| - 2*caplength)^2 - capradius^2 = 0

    :type point: numpy.ndarray
    :param point: Point outside the capsule to give the direction of the collinsion vector (from the origin to the point)

    :type p0: numpy.ndarray
    :param p0: Cylinder "bottom" point, first point

    :type p1: numpy.ndarray
    :param p1: Cylinder "top" point, first point

    :type capradius: int
    :param capradius: Cylinder (and half spheres) radius
    """
    start=time.time()
    def printDebug():
        print('Point: %.2f,%.2f,%.2f' %(point[0],point[1],point[2]))
        print('P0: %.2f,%.2f,%.2f' %(p0[0],p0[1],p0[2]))
        print('P1: %.2f,%.2f,%.2f' %(p1[0],p1[1],p1[2]))
        print('Radius: %.2f' % capradius)

    if type(p0)!=np.ndarray: p0=np.asarray(p0)
    if type(p1)!=np.ndarray: p1=np.asarray(p1)
    capaxis = p1-p0
    caplength = np.linalg.norm(capaxis)/2 - capradius
    center = (p0+p1)/2
    if isNear(caplength,0) or caplength<0:
        print('Capsule length cannot be zero')
        printDebug()
        return None
    #Rotation to align to z axis
    mRotate = alignVectors(capaxis, [0,0,1], shape=4)
    #Translating center of the vector to the origin
    mTranslate = matrixTranslation(-center[0],-center[1],-center[2])
    mTranslate_inv = matrixTranslation(center[0],center[1],center[2])
    #Compose transform
    mTransform = np.dot(mRotate, mTranslate)
    mInv = np.dot(mTranslate_inv, mRotate.T)
    #Transform vector and point
    rotatedPoint = np.dot(mTransform, np.append(point,1))[:-1]
    #Get unit vector pointing to point (direction)
    direction = unitVector(rotatedPoint)
    #Get max value of the step (assuming that the point is outside cylinder.
    #That is, the line passing through the origin and the point will always cross the capsule surface)
    #https://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/lineplane/lineplane.html
    tmax = np.linalg.norm(rotatedPoint)
    tmin = 0
    result = -1
    #Find intersection of the line and surface
    #https://math.stackexchange.com/questions/380576/find-the-point-of-intersection-of-the-line-and-surface
    #https://math.stackexchange.com/questions/1609139/equation-of-a-spherocylinder-capsule
    while not isNear(result, 0, e=0.1):
        t = (tmax+tmin)/2
        if isNear(t, tmax):
            tmax = tmax*2
            print('Warning at mathutils.capsuleCollision(): the joint seems to be inside limb capsule. Please check your calibration.')
            printDebug()
        result = (direction[0]*t)**2 + (direction[1]*t)**2 + (1/4)*(np.abs(direction[2]*t-caplength) + np.abs(direction[2]*t+caplength) - 2*caplength)**2 - capradius**2
        if result > 0:
            tmax = t
        elif result < 0:
            tmin = t
    #Calculate intersection
    intersection = direction*t
    #Get surface normal at intersection
    if intersection[2] >= caplength:
        #sphere surface normal (the unit vector point from the center of the sphere to the intersection point )
        normal = intersection - np.asarray([0,0,caplength])
        theta = np.arccos(np.clip(normal[2]/capradius,-1,1))
        sphere = 1
        if theta is None: printDebug()
        if theta is np.nan: printDebug()

    elif  intersection[2] <= -caplength:
        normal = intersection - np.asarray([0,0,-caplength])
        theta = np.arccos(np.clip(normal[2]/capradius,-1,1))
        sphere = -1
        if theta is None: printDebug()
        if theta is np.nan: printDebug()
    else:
        #cylinder surface normal
        normal = np.asarray([intersection[0],intersection[1],0])
        theta = None
        sphere = 0
    normal = unitVector(normal)
    normal = np.dot(mRotate.T,np.append(normal,1))[:-1]
    if np.any(normal==np.nan) or np.any(normal==None): printDebug()
    #Convert to cylindric coordinates
    #Note: The cases that the intersection occurs on the half spheres, the radius is NOT equal to capradius!
    psi = np.arctan2(intersection[1],intersection[0])
    #cylindcoord = np.asarray([rho/capradius, psi, intersection[2]])
    #This is clearly not the best way to perform the normalization but for similar
    #radius and length sizes it should be enough
    z = intersection[2]/ (caplength)
    capsulecoord = np.asarray([theta, psi, z, sphere])
    intersection = np.dot(mInv,np.append(intersection,1))[:-1]
    capsuleCollision.count+=1
    capsuleCollision.time+=time.time()-start
    return capsulecoord, intersection, normal



def capsuleCartesian(capsulecoord, p0, p1, capradius):
    """
    Transforms the normalized cylindrical coordinates from capsuleCollision to denormalized cartesian coordinates

    :type capsulecoord: numpy.ndarray
    :param point: Normalized capsule coordinates from capsuleCollision()

    :type p0: numpy.ndarray
    :param p0: Cylinder "bottom" point, first point

    :type p1: numpy.ndarray
    :param p1: Cylinder "top" point, first point

    :type capradius: int
    :param capradius: Cylinder (and half spheres) radius
    """
    start=time.time()
    if type(p0)!=np.ndarray: p0=np.asarray(p0)
    if type(p1)!=np.ndarray: p1=np.asarray(p1)
    capaxis = p1-p0
    caplength = np.linalg.norm(capaxis)/2 - capradius
    center = (p0+p1)/2
    if isNear(caplength,0) or caplength<0:
        print('Capsule length cannot be zero. p0: (%.2f,%.2f,%.2f), p1: (%.2f,%.2f,%.2f), radius: %.2f' %(p0[0],p0[1],p0[2],p1[0],p1[1],p1[2], capradius))
        return None
    #Note: we actually want to transform the normalized point to the capsule place
    #So we will store the "inverse" of rotation and translation
    #Rotation to align to z axis
    mRotate = alignVectors([0,0,1], capaxis, shape=4)
    #Translating center of the vector to the origin
    mTranslate = matrixTranslation(center[0],center[1],center[2])
    #Compose inverse transform because
    mTransform = np.dot(mTranslate, mRotate)
    r = capradius
    theta = capsulecoord[0]
    psi = capsulecoord[1]
    z = capsulecoord[2]* (caplength)
    sphere = capsulecoord[3]
    if sphere != 0:
        xyz = np.asarray([r*np.sin(theta)*np.cos(psi), r*np.sin(theta)*np.sin(psi), z])
    else:
        xyz = np.asarray([r*np.cos(psi), r*np.sin(psi), z])

    #Get surface normal at intersection
    if xyz[2] >= caplength:
        #sphere surface normal (the unit vector point from the center of the sphere to the intersection point )
        normal = xyz - np.asarray([0,0,caplength])
    elif xyz[2] <= -caplength:
        normal = xyz - np.asarray([0,0,-caplength])
    else:
        #cylinder surface normal
        normal = np.asarray([xyz[0],xyz[1],0])
    normal = unitVector(normal)
    normal = np.dot(mRotate, np.append(normal,1))[:-1]
    intersection = np.dot(mTransform, np.append(xyz,1))[:-1]

    capsuleCartesian.count+=1
    capsuleCartesian.time+=time.time()-start
    return intersection, normal


def cosBetween(a, b, absolute = True):
    """
    Return the cosine of the angle between vectors a and b, the absolute value is returned is absolute = True.

    :type a: numpy.ndarray
    :param a: Vector one

    :type b: numpy.ndarray
    :param b: Vector two

    :type absolute: int
    :param absolute: Return absolute value. True for yes, False for no.
    """
    if absolute:
        return np.abs(np.dot(unitVector(a),unitVector(b)))
    else:
        return np.dot(unitVector(a),unitVector(b))


def printLog():
    print('capsuleCartesian: %i %f' % (capsuleCartesian.count, capsuleCartesian.time))
    print('capsuleCollision: %i %f' % (capsuleCollision.count, capsuleCollision.time))
    print('eulerFromVector: %i %f' % (eulerFromVector.count, eulerFromVector.time))
    print('eulerFromMatrix: %i %f' % (eulerFromMatrix.count, eulerFromMatrix.time))
    print('angleBetween: %i %f' % (angleBetween.count, angleBetween.time))
    print('alignVectors: %i %f' % (alignVectors.count, alignVectors.time))
    print('projectedBarycentricCoord: %i %f' % (projectedBarycentricCoord.count, projectedBarycentricCoord.time))
    print('clampedBarycentric: %i %f' % (clampedBarycentric.count, clampedBarycentric.time))
    print('matrixMultiply: %i %f' % (matrixMultiply.count, matrixMultiply.time))

capsuleCartesian.count=0
capsuleCartesian.time = 0
capsuleCollision.count=0
capsuleCollision.time=0
eulerFromVector.count=0
eulerFromVector.time=0
eulerFromMatrix.count=0
eulerFromMatrix.time=0
angleBetween.count=0
angleBetween.time=0
alignVectors.time=0
alignVectors.count=0
projectedBarycentricCoord.count=0
projectedBarycentricCoord.time=0
matrixMultiply.time=0
matrixMultiply.count=0
clampedBarycentric.time=0
clampedBarycentric.count=0

#cilcoord, inter = capsuleCollision([0,10,0],[0,1,0],[0,-1,0],5)
#print(inter)
#print(cilcoord)
#print(capsuleCartesian(cilcoord, [0,1,0],[0,-1,0],5))


#x=matrixIdentity(4)
#y=matrixIdentity(4)
#z=matrixMultiply(x,y)
#rot = matrixRotation(45,0,0,1,4)
#vector = [1,0,0,0]
#result = matrixMultiply(rot,vector)

#thetax,thetaz = eulerFromVector([0,0,1])
#print(thetax*180/np.pi)
#print(thetaz*180/np.pi)
