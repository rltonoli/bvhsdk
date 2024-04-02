import numpy as np
import matplotlib.pyplot as plt
from bvhsdk import mathutils

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 20.0

def AnimPlot(data):

    def update(frame):
        xdata, ydata, zdata = [], [], []
        #print("-------------------Frame: %f"%(frame))
        #xdata.append(data[frame,0,:])
        #ydata.append(data[frame,1,:])
        #zdata.append(data[frame,2,:])

        xdata = data[:,0,frame]
        ydata = data[:,1,frame]
        zdata = data[:,2,frame]
        ln.set_data(xdata, ydata)
        ln.set_3d_properties(zdata)
        return ln,

    def animate():
        #ax.set_xlim(np.min(data[:,0,:]),np.max(data[:,0,:]))
        #ax.set_xlim(np.min(data[:,2,:]),np.max(data[:,2,:]))
        #ax.set_zlim(np.min(data[:,1,:]),np.max(data[:,1,:]))
        mini,maxi = np.min(data), np.max(data)
        ax.set_xlim(mini,maxi)
        ax.set_ylim(mini,maxi)
        ax.set_zlim(mini,maxi)
        ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), interval=10,
                             blit=True)

        plt.show()


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    xdata, ydata, zdata = [], [], []
    ln, = plt.plot([], [], 'ro', animated=True)
    animate()

def AnimPlotBones(animation, color='black', frameDelay = 0, viewPlane = 0,floorPlane = True, skiproot = 0, dist=7):

    def update(frame, lines, precomp_positions, parents):
        for line, i in zip(lines, range(len(parents)-len(lines), len(parents))): # This range is to account for skiproot
            x = [precomp_positions[frame, parents[i], 0], precomp_positions[frame, i, 0]]
            y = [precomp_positions[frame, parents[i], 1], precomp_positions[frame, i, 1]]
            z = [precomp_positions[frame, parents[i], 2], precomp_positions[frame, i, 2]]
            line.set_data(x,y)
            line.set_3d_properties(z)
        return lines

    print('Precomputing joint positions...')
    precomp_positions = [joint.getPosition(frame) for frame in range(animation.frames) for joint in animation.getlistofjoints()]
    precomp_positions = np.reshape(np.asarray(precomp_positions), newshape = (animation.frames, len(animation.getlistofjoints()), 3))
    # precomp_positions shape: (frames, joints, xyz)
    
    mindata = np.min(precomp_positions)
    maxdata = np.max(precomp_positions)
    
    frameDelay = int(animation.frametime * 1000) if frameDelay == 0 else frameDelay
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    
    parents = animation.arrayParent()
    
    lines = []
    for i in range(skiproot, len(parents)):
        x = [precomp_positions[0, parents[i], 0], precomp_positions[0 , i, 0]]
        y = [precomp_positions[0, parents[i], 1], precomp_positions[0 , i, 1]]
        z = [precomp_positions[0, parents[i], 2], precomp_positions[0 , i, 2]]
        lines.append(ax.plot(x, y, z, marker='o', linestyle='-', markersize=2, c=color)[0])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(mindata,maxdata)
    ax.set_ylim(mindata,maxdata)
    ax.set_zlim(mindata,maxdata)    
    
    ani = FuncAnimation(fig, update, frames=np.arange(animation.frames), fargs=([lines, precomp_positions, parents]),
                        interval=frameDelay, blit=True)
    
    # Draw floor plane
    if floorPlane:
        
        sx, sz = np.meshgrid(range(-int(maxdata), int(maxdata)), range(-int(maxdata), int(maxdata)))
        sy = np.zeros(shape=sx.shape)
        ax.plot_surface(sx, sy, sz, alpha=0.2)
    
    # Set initial view direction
    if viewPlane:
        # (plane, (elev, azim, roll))
        views = [('XY',   (90, -90, 0)),
                 ('XZ',    (0, -90, 0)),
                 ('YZ',    (0,   0, 0)),
                 ('-XY', (-90,  90, 0)),
                 ('-XZ',   (0,  90, 0)),
                 ('-YZ',   (0, 180, 0))]
        angles = views[viewPlane - 1][1]
        ax.view_init(elev=angles[0], azim=angles[1])
    else:
        ax.view_init(elev=100, azim=-90)
        
    if dist:
        ax.dist=dist
        
    ax.grid(False)
    ax.axis('off')
    
    plt.show()
    return ani



def AnimPlotBones2D(data, plotax='xy'):

    def update(frame, lines, data, ax1, ax2):
        for line, dataBone in zip(lines, data):
            x = np.asarray([dataBone[ax1[0],frame], dataBone[ax1[1],frame]])
            y = np.asarray([dataBone[ax2[0],frame], dataBone[ax2[1],frame]])
            line.set_data(x,y)
        return lines

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    choices = {'xy': ([0,4], [1,5]),
               'xz': ([0,4], [2,6]),
               'yx': ([1,5], [0,4]),
               'yz': ([1,5], [2,6]),
               'zx': ([2,6], [0,4]),
               'zy': ([2,6], [1,5])}


    ax1, ax2 = choices.get(plotax, ([0,4],[1,5]))


    lines = []
    for i in range(len(data)):
        lines.append(ax.plot([data[i,ax1[0],0], data[i,ax1[1],0]], [data[i,ax2[0],0], data[i,ax2[1],0]],'-o')[0])


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    mini,maxi = np.min(data), np.max(data)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(lines, data, ax1, ax2),interval=100,
                             blit=True)

    plt.show()


def JointVelocity(data, box=0):
    x,y,z = data.shape
    lines = np.zeros([x,1,z-1])
    for jointCount in range(x):
        for frameCount in range(z-1):
            deltaXsquare = np.square(data[jointCount,0,frameCount+1]-data[jointCount,0,frameCount])
            deltaYsquare = np.square(data[jointCount,1,frameCount+1]-data[jointCount,1,frameCount])
            deltaZsquare = np.square(data[jointCount,2,frameCount+1]-data[jointCount,2,frameCount])
            lines[jointCount, 0, frameCount] = np.sqrt(deltaXsquare + deltaYsquare + deltaZsquare)

    fig = plt.figure(figsize=(12,8))

    for joint in range(x):
        plt.plot(lines[joint, 0, :])
    plt.show()


def PosePlot(data, label=[]):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    xdata = data[:,0]
    #ydata = -data[:,2]
    #zdata = data[:,1]
    ydata = data[:,1]
    zdata = data[:,2]
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xdata,ydata,zdata)
    if label:
        for i in range(len(label)): #plot each point + it's index as text above
            ax.text(data[i,0],-data[i,2],data[i,1],  '%s' % (label[i]), size=8, zorder=1, color='k')

def PosePlotBones(joints,bones):
    """
    Plot a pose (just one frame) with joints and bones
    """
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(joints), np.max(joints)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2])
    for i in range(len(bones)):
        ax.plot([bones[i,0], bones[i,3]], [bones[i,1], bones[i,4]], [bones[i,2], bones[i,5]], color='black')

def PosePlotBonesSurface(joints,bones,surface):
    """
    Plot a pose (just one frame) with joints, bones and surface data
    """
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(joints), np.max(joints)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2])
    ax.scatter(surface[:,0], surface[:,1], surface[:,2], s=40,c='r',marker = '^')
    for i in range(len(bones)):
        ax.plot([bones[i,0], bones[i,3]], [bones[i,1], bones[i,4]], [bones[i,2], bones[i,5]], color='black')
    plt.show()


def AnimationSurface(animation, data, listofpoints):
    """
    Plot animation with surface information
    """
    def update(frame, lines, data, scatters, listofpoints):
        print(frame)
        for line, dataBone in zip(lines, data):
            x = np.asarray([dataBone[0,frame], dataBone[3,frame]])
            y = np.asarray([dataBone[1,frame], dataBone[4,frame]])
            z = np.asarray([dataBone[2,frame], dataBone[5,frame]])
            line.set_data(x,y)
            line.set_3d_properties(z)
        for scat, point in zip(scatters,listofpoints):
            x,y,z = point.getPosition(animation, frame)
            scat.set_data([x],[y])
            scat.set_3d_properties([z])
            #scat.set_data([point.position[frame][0]],[point.position[frame][1]])
            #scat.set_3d_properties([point.position[frame][2]])

        return lines+scatters

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    for i in range(len(data)):
        lines.append(ax.plot([data[i,0,0], data[i,3,0]], [data[i,1,0], data[i,4,0]], [data[i,2,0], data[i,5,0]],'-o', color='black', markersize=1)[0])

    scatters = []
    for i in range(len(listofpoints)):
        x,y,z = listofpoints[i].getPosition(animation, 0)
        scatters.append(ax.plot([x],[y],[z],'o', color='red', markersize=1)[0])


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(data), np.max(data)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    print('hi')
    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(lines, data, scatters, listofpoints),interval=1,
                             blit=True)
    print('hi')
    plt.show()


def AnimationSurfaceAndVectors(data, listofpoints, dispvectors):
    """
    Plot animation with surface information and displacement vectors
    """
    def update(frame, bones, data, scatters, listofpoints, vectors, dispvectors):
        print(frame)
        for line, dataBone in zip(bones, data):
            x = np.asarray([dataBone[0,frame], dataBone[4,frame]])
            y = np.asarray([dataBone[1,frame], dataBone[5,frame]])
            z = np.asarray([dataBone[2,frame], dataBone[6,frame]])
            line.set_data(x,y)
            line.set_3d_properties(z)
        for scat, point in zip(scatters,listofpoints):
            scat.set_data([point.position[frame,0]],[point.position[frame,1]])
            scat.set_3d_properties([point.position[frame,2]])
        for vec, dispvec in zip(vectors, dispvectors[frame]):
            x = np.asarray([dispvec[0][0], dispvec[1][0]])
            y = np.asarray([dispvec[0][1], dispvec[1][1]])
            z = np.asarray([dispvec[0][2], dispvec[1][2]])
            vec.set_data(x,y)
            vec.set_3d_properties(z)

        return bones+scatters+vectors

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    bones = []
    for i in range(len(data)):
        bones.append(ax.plot([data[i,0,0], data[i,4,0]], [data[i,1,0], data[i,5,0]], [data[i,2,0], data[i,6,0]],'-o', color='black')[0])

    scatters = []
    for i in range(len(listofpoints)):
        scatters.append(ax.plot([listofpoints[i].position[0,0]],[listofpoints[i].position[0,1]],[listofpoints[i].position[0,2]],'o', color='red', markersize=1)[0])

    vectors = []
    for i in range(len(dispvectors[0])):
        #[frame][vectorfromtriangle][p1orp2][xyz]
        vectors.append(ax.plot([dispvectors[0][i][0][0], dispvectors[0][i][1][0]], [dispvectors[0][i][0][1], dispvectors[0][i][1][1]], [dispvectors[0][i][0][2], dispvectors[0][i][1][2]],'-', color='red')[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(data), np.max(data)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(bones, data, scatters, listofpoints, vectors, dispvectors),interval=1,
                             blit=True)

    plt.show()

def AnimationSurface2(data, listofpoints):
    """
    OLD (bones with 6 float array)
    Plot animation with surface information
    """
    def update(frame, lines, data, scatters, listofpoints):
        print(frame)
        for line, dataBone in zip(lines, data):
            x = np.asarray([dataBone[0,frame], dataBone[3,frame]])
            y = np.asarray([dataBone[1,frame], dataBone[4,frame]])
            z = np.asarray([dataBone[2,frame], dataBone[5,frame]])
            line.set_data(x,y)
            line.set_3d_properties(z)
        for scat, point in zip(scatters,listofpoints):
            scat.set_data([point.position[frame,0]],[point.position[frame,1]])
            scat.set_3d_properties([point.position[frame,2]])

        return lines+scatters

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    for i in range(len(data)):
        lines.append(ax.plot([data[i,0,0], data[i,3,0]], [data[i,1,0], data[i,4,0]], [data[i,2,0], data[i,5,0]],'-o', color='black')[0])

    scatters = []
    for i in range(len(listofpoints)):
        scatters.append(ax.plot([listofpoints[i].position[0,0]],[listofpoints[i].position[0,1]],[listofpoints[i].position[0,2]],'o', color='red', markersize=1)[0])


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(data), np.max(data)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(lines, data, scatters, listofpoints),interval=1,
                             blit=True)

    plt.show()


def plotAxesFromMatrix(matrix):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
#    ax.plot([0,0,0],[matrix[0,0], matrix[1,0], matrix[2,0]], color='red')
#    ax.plot([0,0,0],[matrix[0,1], matrix[1,1], matrix[2,1]], color='green')
#    ax.plot([0,0,0],[matrix[0,2], matrix[1,2], matrix[2,2]], color='blue')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.plot([0,matrix[0,0]],[0,matrix[1,0]],[0, matrix[2,0]], color='red')
    ax.plot([0,matrix[0,1]],[0,matrix[1,1]],[0, matrix[2,1]], color='green')
    ax.plot([0,matrix[0,2]],[0,matrix[1,2]],[0, matrix[2,2]], color='blue')
    plt.show()

def plotAxesVectors(x,y,z=False):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
#    ax.plot([0,0,0],[matrix[0,0], matrix[1,0], matrix[2,0]], color='red')
#    ax.plot([0,0,0],[matrix[0,1], matrix[1,1], matrix[2,1]], color='green')
#    ax.plot([0,0,0],[matrix[0,2], matrix[1,2], matrix[2,2]], color='blue')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.plot([0,x[0]],[0,x[1]],[0, x[2]], color='red')
    ax.plot([0,y[0]],[0,y[1]],[0, y[2]], color='green')
    if z:
        ax.plot([0,z[0]],[0,z[1]],[0, z[2]], color='blue')
    plt.show()

def plotVector(vec):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.plot([0,vec[0]],[0,vec[1]],[0, vec[2]], color='red')
    plt.show()

def AnimationSurfaceAndVectors2(data, listofpoints, dispvectors):
    """
    OLD (bones with 6 float array)
    Plot animation with surface information and displacement vectors
    """
    def update(frame, bones, data, scatters, listofpoints, vectors, dispvectors):
        print(frame)
        for line, dataBone in zip(bones, data):
            x = np.asarray([dataBone[0,frame], dataBone[3,frame]])
            y = np.asarray([dataBone[1,frame], dataBone[4,frame]])
            z = np.asarray([dataBone[2,frame], dataBone[5,frame]])
            line.set_data(x,y)
            line.set_3d_properties(z)
        for scat, point in zip(scatters,listofpoints):
            scat.set_data([point.position[frame,0]],[point.position[frame,1]])
            scat.set_3d_properties([point.position[frame,2]])
        for vec, dispvec in zip(vectors, dispvectors[frame]):
            x = np.asarray([dispvec[0][0], dispvec[1][0]])
            y = np.asarray([dispvec[0][1], dispvec[1][1]])
            z = np.asarray([dispvec[0][2], dispvec[1][2]])
            vec.set_data(x,y)
            vec.set_3d_properties(z)

        return bones+scatters+vectors

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    bones = []
    for i in range(len(data)):
        bones.append(ax.plot([data[i,0,0], data[i,3,0]], [data[i,1,0], data[i,4,0]], [data[i,2,0], data[i,5,0]],'-o', color='black')[0])

    scatters = []
    for i in range(len(listofpoints)):
        scatters.append(ax.plot([listofpoints[i].position[0,0]],[listofpoints[i].position[0,1]],[listofpoints[i].position[0,2]],'o', color='red', markersize=1)[0])

    vectors = []
    for i in range(len(dispvectors[0])):
        #[frame][vectorfromtriangle][p1orp2][xyz]
        vectors.append(ax.plot([dispvectors[0][i][0][0], dispvectors[0][i][1][0]], [dispvectors[0][i][0][1], dispvectors[0][i][1][1]], [dispvectors[0][i][0][2], dispvectors[0][i][1][2]],'-', color='red')[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mini,maxi = np.min(data), np.max(data)
    ax.set_xlim(mini,maxi)
    ax.set_ylim(mini,maxi)
    ax.set_zlim(mini,maxi)
    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(bones, data, scatters, listofpoints, vectors, dispvectors),interval=1,
                             blit=True)

    plt.show()





def DebugEgoCoord(animation, frame, proj = '3d'):

    def getVectors(animation, frame):
        """
        Get vectors to calculate the kinematic path

        :type animation: pyanimation.Animation
        :param animation: Animation (skeleton) to get the distance between mapped joints
        """
        lhand = animation.getskeletonmap().lhand
        rhand = animation.getskeletonmap().rhand
        lforearm = animation.getskeletonmap().lforearm
        rforearm = animation.getskeletonmap().rforearm
        larm = animation.getskeletonmap().larm
        rarm = animation.getskeletonmap().rarm
        upspine = animation.getskeletonmap().spine3
        #não estou procurando "head" pq o 'head' da Talita não corresponde ao 'head' do shogun
        neck = animation.getskeletonmap().neck
        hips = animation.getskeletonmap().hips

        lvec_fore = lhand.getPosition(frame) - lforearm.getPosition(frame)
        rvec_fore = rhand.getPosition(frame) - rforearm.getPosition(frame)

        lvec_arm = lforearm.getPosition(frame) - larm.getPosition(frame)
        rvec_arm = rforearm.getPosition(frame) - rarm.getPosition(frame)

        lvec_clavicle = larm.getPosition(frame) - upspine.getPosition(frame)
        rvec_clavicle = rarm.getPosition(frame) - upspine.getPosition(frame)

        vec_neck = upspine.getPosition(frame) - neck.getPosition(frame)

        vec_hips = upspine.getPosition(frame) - hips.getPosition(frame)

        return lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_hips


    lvec_fore, rvec_fore, lvec_arm, rvec_arm, lvec_clavicle, rvec_clavicle, vec_neck, vec_hips = getVectors(animation, frame)
    hips = [0,0,0]
    Upspine = vec_hips - hips
    Neck = - vec_neck + Upspine
    RArm = rvec_clavicle + Upspine
    LArm = lvec_clavicle + Upspine
    RFore = rvec_arm + RArm
    LFore = lvec_arm + LArm
    RHand = rvec_fore + RFore
    LHand = lvec_fore + LFore
    print(rvec_fore)
    print(rvec_arm)
    print(rvec_clavicle)
    print(vec_neck)
    print(vec_hips)
    if proj=='3d':
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.plot([hips[0],Upspine[0]],[hips[1],Upspine[1]],[hips[2],Upspine[2]], '-o',color='black')
        ax.plot([Upspine[0],Neck[0]],[Upspine[1],Neck[1]],[Upspine[2],Neck[2]], '-o',color='gray')
        ax.plot([Upspine[0],RArm[0]],[Upspine[1],RArm[1]],[Upspine[2],RArm[2]], '-o',color='yellow')
        ax.plot([Upspine[0],LArm[0]],[Upspine[1],LArm[1]],[Upspine[2],LArm[2]], '-o',color='yellow')
        ax.plot([RArm[0],RFore[0]],[RArm[1],RFore[1]],[RArm[2],RFore[2]], '-o',color='red')
        ax.plot([LArm[0],LFore[0]],[LArm[1],LFore[1]],[LArm[2],LFore[2]], '-o',color='red')
        ax.plot([RFore[0],RHand[0]],[RFore[1],RHand[1]],[RFore[2],RHand[2]], '-o',color='blue')
        ax.plot([LFore[0],LHand[0]],[LFore[1],LHand[1]],[LFore[2],LHand[2]], '-o',color='blue')
        plt.show()
    else:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.plot([hips[0],Upspine[0]],[hips[1],Upspine[1]], '-o',color='black')
        ax.plot([Upspine[0],Neck[0]],[Upspine[1],Neck[1]], '-o',color='gray')
        ax.plot([Upspine[0],RArm[0]],[Upspine[1],RArm[1]], '-o',color='yellow')
        ax.plot([Upspine[0],LArm[0]],[Upspine[1],LArm[1]], '-o',color='yellow')
        ax.plot([RArm[0],RFore[0]],[RArm[1],RFore[1]], '-o',color='red')
        ax.plot([LArm[0],LFore[0]],[LArm[1],LFore[1]], '-o',color='red')
        ax.plot([RFore[0],RHand[0]],[RFore[1],RHand[1]], '-o',color='blue')
        ax.plot([LFore[0],LHand[0]],[LFore[1],LHand[1]], '-o',color='blue')
        plt.show()



def PlotBVH(animation, 
            frameDelay = 0, 
            precompute = True, 
            viewPlane = 0,
            floorPlane = True,
            ):
    """
    Plot BVH animation joints as point cloud. Currently assumes Y-up character.
    Works better if precompute is True.
    Calculate the position inside this funtion.
    Ipython Jupyter Notebook-friendly function.

    Make sure to include the following line in your notebook:
    %matplotlib notebook
    import matplotlib
    matplotlib.rc('animation', html='html5')

    These lines may also be necessary:
    from ipywidgets import interact, interactive, widgets
    from IPython.display import display
    
    :param anim.Animation animation: Animation object to be draw
    :param int frameDelay: Interval or delay between frames of matplotlib's FuncAnimation function in miliseconds. If 0, use animation's frametime to match intended fps from BVH file.
    :param int viewPlane: Primary view plane option (choose between 1 and )
    :param bool precompute: If True, the function will precompute the positions of the joints. If False, the function will calculate the position of the joints at each frame.
    """
    def update(frame, scatters, precomp_positions):
        for scat, joint, i in zip(scatters, animation.getlistofjoints(), range(len(animation.getlistofjoints()))):
            if precompute:
                position = precomp_positions[frame, i]
            else:
                position = joint.getPosition(frame)
            scat.set_data([position[0]],[position[1]])
            scat.set_3d_properties([position[2]])

        return scatters

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
        
    precomp_positions = None
    if precompute:
        print('Precomputing positions...')
        precomp_positions = [joint.getPosition(frame) for frame in range(animation.frames) for joint in animation.getlistofjoints()]
        precomp_positions = np.reshape(np.asarray(precomp_positions), newshape = (animation.frames, len(animation.getlistofjoints()), 3))
        
    frameDelay = int(animation.frametime * 1000) if frameDelay == 0 else frameDelay

    scatters = []
    maxdata = -np.inf
    mindata = np.inf
    for i, joint in enumerate(animation.getlistofjoints()):
        if precompute:
            position = precomp_positions[0, i]
        else:
            position = joint.getPosition(frame = 0)
        scatters.append(ax.plot([position[0]],[position[1]],[position[2]],'o', color='red', markersize=1)[0])
        if np.min(position)<mindata:
            mindata = np.min(position)
        if np.max(position)>maxdata:
            maxdata = np.max(position)      
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(mindata,maxdata)
    ax.set_ylim(mindata,maxdata)
    ax.set_zlim(mindata,maxdata)
    
    # Draw floor plane
    if floorPlane:
        
        sx, sz = np.meshgrid(range(-int(maxdata), int(maxdata)), range(-int(maxdata), int(maxdata)))
        sy = np.zeros(shape=sx.shape)
        ax.plot_surface(sx, sy, sz, alpha=0.2)
    
    # Set initial view direction
    if viewPlane:
        # (plane, (elev, azim, roll))
        views = [('XY',   (90, -90, 0)),
                 ('XZ',    (0, -90, 0)),
                 ('YZ',    (0,   0, 0)),
                 ('-XY', (-90,  90, 0)),
                 ('-XZ',   (0,  90, 0)),
                 ('-YZ',   (0, 180, 0))]
        angles = views[viewPlane - 1][1]
        ax.view_init(elev=angles[0], azim=angles[1])
    else:
        ax.view_init(elev=100, azim=-90)
        
    ax.grid(False)
    ax.axis('off')
    
    ani = FuncAnimation(fig, update, frames=np.arange(animation.frames), fargs=([scatters, precomp_positions]) ,interval=frameDelay, blit=True)
    plt.show()
    return ani


def PlotBVHSurface(animation, surface):
    def update(frame, surf, lines):
        for triangle_plot, i in zip(surf, range(len(surface.headmesh)+len(surface.bodymesh))):
            if i< len(surface.headmesh):
                vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in surface.headmesh[i]]
                vertices.append([surface.headmesh[i][0].getPosition(animation,frame)[0],surface.headmesh[i][0].getPosition(animation,frame)[1],surface.headmesh[i][0].getPosition(animation,frame)[2]])
                vertices = np.asarray(vertices)
            else:
                j = i - len(surface.headmesh)
                vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in surface.bodymesh[j]]
                vertices.append([surface.bodymesh[j][0].getPosition(animation,frame)[0],surface.bodymesh[j][0].getPosition(animation,frame)[1],surface.bodymesh[j][0].getPosition(animation,frame)[2]])
                vertices = np.asarray(vertices)
            triangle_plot.set_data(vertices[:,0],vertices[:,1])
            triangle_plot.set_3d_properties(vertices[:,2])

        for line, bone in zip(lines, animation.getBones(frame)):
            line.set_data([bone[0], bone[3]], [bone[1], bone[4]])
            line.set_3d_properties([bone[2], bone[5]])
        return surf+lines

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    scatters = []
    maxdata = -np.inf
    mindata = np.inf
    for joint in animation.getlistofjoints():
        position = joint.getPosition(frame = 0)
        #scatters.append(ax.plot([position[0]],[position[1]],[position[2]],'o', color='red', markersize=1)[0])
        if np.min(position)<mindata:
            mindata = np.min(position)
        if np.max(position)>maxdata:
            maxdata = np.max(position)

    surf = []
    for triangle in surface.headmesh:
        vertices = [[vert.getPosition(animation,0)[0],vert.getPosition(animation,0)[1],vert.getPosition(animation,0)[2]] for vert in triangle]
        vertices.append([triangle[0].getPosition(animation,0)[0],triangle[0].getPosition(animation,0)[1],triangle[0].getPosition(animation,0)[2]])
        vertices = np.asarray(vertices)
        surf.append(ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)[0])

    for triangle in surface.bodymesh:
        vertices = [[vert.getPosition(animation,0)[0],vert.getPosition(animation,0)[1],vert.getPosition(animation,0)[2]] for vert in triangle]
        vertices.append([triangle[0].getPosition(animation,0)[0],triangle[0].getPosition(animation,0)[1],triangle[0].getPosition(animation,0)[2]])
        vertices = np.asarray(vertices)
        surf.append(ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)[0])

    lines = []
    for bone in animation.getBones(0):
        lines.append(ax.plot([bone[0], bone[3]], [bone[1], bone[4]], [bone[2], bone[5]],'-o', color='black')[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(mindata,maxdata)
    ax.set_ylim(mindata,maxdata)
    ax.set_zlim(mindata,maxdata)
    ani = FuncAnimation(fig, update, frames=np.arange(animation.frames), fargs=(surf,lines) ,interval=1, blit=True)
    return ani

def PlotPoseAndSurface(animation, surface, axes = False, frame = 0):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    aux = animation.getBones(frame)
    bones = []
    for i in range(len(aux)):
        bones.append(ax.plot([aux[i,0], aux[i,3]], [aux[i,1], aux[i,4]], [aux[i,2], aux[i,5]],'-o', color='black', alpha=0.5)[0])


    origin = np.asarray([0,0,0])
    x = np.asarray([7,0,0])
    y = np.asarray([0,7,0])
    z = np.asarray([0,0,7])

    if axes:
        for joint in animation.getlistofjoints():
            if joint in animation.getskeletonmap().getJointsNoRootHips():
                if joint != animation.getskeletonmap().neck1 and joint != animation.getskeletonmap().neck and joint != animation.getskeletonmap().neck1  and joint != animation.getskeletonmap().spine1 and joint != animation.getskeletonmap().spine2:
                    matrix = joint.getGlobalTransform(frame)
                    local_origin = origin + matrix[:-1,-1]
                    rot = mathutils.shape4ToShape3(matrix)
                    local_x = np.dot(rot,x) + local_origin
                    local_y = np.dot(rot,y) + local_origin
                    local_z = np.dot(rot,z) + local_origin
                    ax.plot([local_origin[0], local_x[0]],[local_origin[1], local_x[1]],[local_origin[2], local_x[2]], color='red',linewidth=2.5)
                    ax.plot([local_origin[0], local_y[0]],[local_origin[1], local_y[1]],[local_origin[2], local_y[2]], color='green',linewidth=2.5)
                    ax.plot([local_origin[0], local_z[0]],[local_origin[1], local_z[1]],[local_origin[2], local_z[2]], color='blue',linewidth=2.5)

    if surface:
        surf = []
        for triangle in surface.headmesh:
            vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
            vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
            vertices = np.asarray(vertices)
            surf.append(ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)[0])

        for triangle in surface.bodymesh:
            vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
            vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
            vertices = np.asarray(vertices)
            surf.append(ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)[0])

#    vectors = []
#    for i in range(len(dispvectors[0])):
#        #[frame][vectorfromtriangle][p1orp2][xyz]
#        vectors.append(ax.plot([dispvectors[0][i][0][0], dispvectors[0][i][1][0]], [dispvectors[0][i][0][1], dispvectors[0][i][1][1]], [dispvectors[0][i][0][2], dispvectors[0][i][1][2]],'-', color='red')[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # mini,maxi = np.min(aux), np.max(aux)
    # ax.set_xlim(mini,maxi)
    # ax.set_ylim(mini,maxi)
    # ax.set_zlim(mini,maxi)
    # ax.set_axis_off()
    ax.set_xlim(-70,70)
    ax.set_ylim(0,140)
    ax.set_zlim(-70,70)
    ax.view_init(elev=100, azim=-90)
    ax.axis('off')
    plt.tight_layout()

#    ani = FuncAnimation(fig, update, frames=np.arange(len(data[0,0,:])), fargs=(bones, data, scatters, listofpoints, vectors, dispvectors),interval=1,
#                             blit=True)

    plt.show()


def CheckTargets(animation, joint, joint1, ego):
    target = np.asarray([ego.getTarget(frame) for frame in range(animation.frames)])
    position = np.asarray([joint.getPosition(frame) for frame in range(animation.frames)])
    position1 = np.asarray([joint1.getPosition(frame) for frame in range(animation.frames)])


    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(target[:,0], label='X', color = 'red', linestyle = '--')
    ax.plot(target[:,1], label='Y', color = 'green', linestyle = '--')
    ax.plot(target[:,2], label='Z', color = 'blue', linestyle = '--')
    ax.plot(position[:,0], color = 'red', linestyle = '-')
    ax.plot(position[:,1], color = 'green', linestyle = '-')
    ax.plot(position[:,2], color = 'blue', linestyle = '-')
    ax.plot(position1[:,0], color = 'black', linestyle = '-')
    ax.plot(position1[:,1], color = 'black', linestyle = '-')
    ax.plot(position1[:,2], color = 'black', linestyle = '-')

    plt.legend(title='Target position:')
    plt.show()


def PlotDispvectorsFrames(srcAnim, srcSurface, tgtAnim, tgtSurface, egoCoord, f1, f2, f3):
    fig, axs = plt.subplots(2, 3, figsize=(16,8))

    min_X, max_X = np.inf,np.NINF
    min_Y, max_Y = np.inf,np.NINF
    for animation, surface, line in zip([srcAnim, tgtAnim],[srcSurface, tgtSurface],range(2)):
        for frame, column in zip([f1,f2,f3],range(3)):
            aux = animation.getBones(frame)

            for i in range(len(aux)):
                axs[line, column].plot([aux[i,0], aux[i,3]], [aux[i,1], aux[i,4]],'-o', color='black')

            for triangle in surface.headmesh:
                vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1]] for vert in triangle]
                vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1]])
                vertices = np.asarray(vertices)
                axs[line, column].plot(vertices[:,0],vertices[:,1],'-o', color='red', markersize=1, alpha = 0.5)

            for triangle in surface.bodymesh:
                vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1]] for vert in triangle]
                vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1]])
                vertices = np.asarray(vertices)
                axs[line, column].plot(vertices[:,0],vertices[:,1],'-o', color='red', markersize=1, alpha = 0.5)
            mini_X,maxi_X = np.min([aux[:,0].min(),aux[:,3].min()]), np.max([aux[:,0].max(),aux[:,3].max()])
            mini_Y, maxi_Y = np.min([aux[:,1].min(),aux[:,4].min()]), np.max([aux[:,1].max(),aux[:,4].max()])
            if mini_X<min_X:
                min_X=mini_X
            if maxi_X>max_X:
                max_X=maxi_X
            if mini_Y<min_Y:
                min_Y=mini_Y
            if maxi_Y>max_Y:
                max_Y=maxi_Y

    lenNoLimb = len(srcSurface.headmesh)+len(srcSurface.bodymesh)
    for frame, column in zip([f1,f2,f3],range(3)):
        dispvector = np.asarray([np.asarray(egoCoord.framecoord[frame].dispvector[i]*egoCoord.framecoord[frame].tau[i])+egoCoord.framecoord[frame].refpoint[i] for i in range(lenNoLimb)])
        refpoint = np.asarray(egoCoord.framecoord[frame].refpoint[:lenNoLimb])
        norm_importance = egoCoord.framecoord[frame].importance/egoCoord.framecoord[frame].importance.max()
        for i in range(lenNoLimb):
            axs[0, column].plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]], color=[1*(1-norm_importance[i]),1*(1-norm_importance[i]),1])
        dispvector = np.asarray([np.asarray(egoCoord.framecoord[frame].tgt_dispvector[i])+egoCoord.framecoord[frame].tgt_refpoint[i] for i in range(lenNoLimb)])
        refpoint = np.asarray(egoCoord.framecoord[frame].tgt_refpoint[:lenNoLimb])
        norm_importance = egoCoord.framecoord[frame].importance/egoCoord.framecoord[frame].importance.max()
        for i in range(lenNoLimb):
            axs[1, column].plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]], color=[1*(1-norm_importance[i]),1*(1-norm_importance[i]),1])
    for line in range(2):
        for column in range(3):
            axs[line, column].set_xlim(min_X-5,max_X+5)
            axs[line, column].set_ylim(min_Y-10,max_Y+10)
            axs[line, column].set_frame_on(False)
            axs[line, column].tick_params(axis='both', which='both', length=0)
            axs[line, column].set_xticks([])
            if column!=0:
                axs[line, column].set_yticks([])
            #axs[line, column].set_axis_off()
    frames = [f1,f2,f3]
    font_size = 16
    for column in range(3):
        axs[1,column].set_xlabel(str.format('Frame %i' % frames[column]), fontsize=font_size)

    axs[0,0].set_ylabel('Source', fontsize=font_size)
    axs[1,0].set_ylabel('Target', fontsize=font_size)

    plt.tight_layout()
    plt.show()


def PlotDispvectorsFrames3D(animation, surface, egoCoord, frame, target = True, skipHead = False, skipBody = False):
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    aux = animation.getBones(frame)

    for i in range(len(aux)):
        ax.plot([aux[i,0], aux[i,3]], [aux[i,1], aux[i,4]],[aux[i,2], aux[i,5]],'-o', color='black')

    for triangle in surface.headmesh:
        vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
        vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
        vertices = np.asarray(vertices)
        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)

    for triangle in surface.bodymesh:
        vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
        vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
        vertices = np.asarray(vertices)
        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'-o', color='red', markersize=1, alpha = 0.5)
    mini_X,maxi_X = np.min([aux[:,0].min(),aux[:,3].min()]), np.max([aux[:,0].max(),aux[:,3].max()])
    mini_Y, maxi_Y = np.min([aux[:,1].min(),aux[:,4].min()]), np.max([aux[:,1].max(),aux[:,4].max()])
    mini_Z, maxi_Z = np.min([aux[:,2].min(),aux[:,5].min()]), np.max([aux[:,2].max(),aux[:,5].max()])

    lenNoLimb = len(surface.headmesh)+len(surface.bodymesh)
    #Print target or source ego coords refpoint
    if not target:
        dispvector = np.asarray([np.asarray(egoCoord.framecoord[frame].dispvector[i]*egoCoord.framecoord[frame].tau[i])+egoCoord.framecoord[frame].refpoint[i] for i in range(lenNoLimb)])
        refpoint = np.asarray(egoCoord.framecoord[frame].refpoint[:lenNoLimb])
        norm_importance = egoCoord.framecoord[frame].importance/egoCoord.framecoord[frame].importance.max()
        for i in range(lenNoLimb):
            # ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[np.clip(1*(1-norm_importance[i]),0,0.8),np.clip(1*(1-norm_importance[i]),0,0.8),1,0.8])
            ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[1*(1-norm_importance[i]),1*(1-norm_importance[i]),1,1])
            ax.scatter(refpoint[i,0], refpoint[i,1], refpoint[i,2], color='red', alpha=0.5)
    else:
        dispvector = np.asarray([np.asarray(egoCoord.framecoord[frame].tgt_dispvector[i])+egoCoord.framecoord[frame].tgt_refpoint[i] for i in range(len(egoCoord.framecoord[frame].tgt_refpoint))])
        refpoint = np.asarray(egoCoord.framecoord[frame].tgt_refpoint[:])
        norm_importance = egoCoord.framecoord[frame].importance/egoCoord.framecoord[frame].importance.max()
        for i in range(len(egoCoord.framecoord[frame].tgt_refpoint)):
            if skipHead:
                if i>len(surface.headmesh):
                    # ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[np.clip(1*(1-norm_importance[i]),0,1),np.clip(1*(1-norm_importance[i]),0,1),1,0.8])
                    ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[1*(1-norm_importance[i]),1*(1-norm_importance[i]),1,1])
                    ax.scatter(refpoint[i,0], refpoint[i,1], refpoint[i,2], color='red', alpha=0.5)
            elif skipBody:
                if i<len(surface.headmesh):
                    ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[np.clip(1*(1-norm_importance[i]),0,1),np.clip(1*(1-norm_importance[i]),0,1),1,1])
                    ax.scatter(refpoint[i,0], refpoint[i,1], refpoint[i,2], color='red', alpha=0.5)
            else:
                ax.plot([refpoint[i,0],dispvector[i,0]],[refpoint[i,1],dispvector[i,1]],[refpoint[i,2],dispvector[i,2]], color=[np.clip(1*(1-norm_importance[i]),0,1),np.clip(1*(1-norm_importance[i]),0,1),1,1])
                ax.scatter(refpoint[i,0], refpoint[i,1], refpoint[i,2], color='red', alpha=0.5)


        target = egoCoord.getTarget(frame)
        ax.scatter(target[0], target[1], target[2], color='red', marker="X", s=100, alpha=0.5)
        ee = animation.getskeletonmap().rhand.getPosition(frame)
        ax.scatter(ee[0], ee[1], ee[2], color='green', marker="o", s=80)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim(mini_X,maxi_X)
    # ax.set_ylim(mini_Y,maxi_Y)
    # ax.set_zlim(mini_Z,maxi_Z)
    ax.set_xlim(-100,100)
    ax.set_ylim(0,200)
    ax.set_zlim(-100,100)
    ax.view_init(elev=100, azim=-90)
    ax.axis('off')
    plt.show()




def MSc_printSurfaces(animation, surface, step = 45, frame = 0):
    """
    Imprime surperficies em diferentes posições
    Parameters
    ----------
    anim : TYPE
        DESCRIPTION.
    surf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(1, 6, figsize=(16,8))
    rot = mathutils.matrixR([0,-90+step,0])

    for i in range(6):
        rot = mathutils.matrixR([0,-90+step*i,0])
        for triangle in surface.bodymesh + surface.headmesh:
            vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
            vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
            vertices = [np.dot(rot,vertices[i]) for i in range(len(vertices))]
            vertices = np.asarray(vertices)
            if (vertices[:,2]<0).any():
                alpha = 0.3
            else:
                alpha = 1
            #zcolor = np.abs(vertices[:,2])/max(np.abs(vertices[:,2]))
            axs[i].plot(vertices[:,0],vertices[:,1],'-o', color='red', markersize=5, alpha = 0.5)
            axs[i].set_axis_off()
            axs[i].set_xlim(-25,25)
            axs[i].set_ylim(70,180)
            # print(vertices[:,2])
            # print('----------------------')


def MSc_printSurfacesAndPoses(animation, surface, step = 45, frame = 0, figs = 6):
    """
    Imprime surperficies em diferentes posições
    Parameters
    ----------
    anim : TYPE
        DESCRIPTION.
    surf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig, axs = plt.subplots(1, figs, figsize=(16,8))
    rot = mathutils.matrixR([0,-90+step,0])

    aux = animation.getBones(frame)
    print(aux.shape)


    for i in range(figs):
        rot = mathutils.matrixR([0,-90+step*i,0])


        for j in range(len(aux)):
            bones1 = np.asarray([np.dot(rot,aux[k,:3]) for k in range(len(aux))])
            bones2 = np.asarray([np.dot(rot,aux[k,3:]) for k in range(len(aux))])
            # if j==0:
            #     print(bones1.shape)
            #Plot skeleton
        for j in range(len(bones1)):
            axs[i].plot([bones1[j,0], bones2[j,0]], [bones1[j,1], bones2[j,1]],'-o', color='black')[0]

        for triangle in surface.bodymesh + surface.headmesh:
            vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
            vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
            vertices = [np.dot(rot,vertices[i]) for i in range(len(vertices))]
            vertices = np.asarray(vertices)
            if (vertices[:,2]<0).any():
                alpha = 0.3
            else:
                alpha = 1
            #zcolor = np.abs(vertices[:,2])/max(np.abs(vertices[:,2]))
            axs[i].plot(vertices[:,0],vertices[:,1],'-o', color='red', markersize=5, alpha = 0.5)
            axs[i].set_axis_off()
            axs[i].set_xlim(-35,35)
            axs[i].set_ylim(70,180)
            # print(vertices[:,2])
            # print('----------------------')

def MSc_printSurfaceAndCapsule(animation, surface, frame = 0, plotcapsule=True):
    """
    Imprime surperficies em diferentes posições
    Parameters
    ----------
    anim : TYPE
        DESCRIPTION.
    surf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(figsize=(12,12))

    aux = animation.getBones(frame)
    for i in range(len(aux)):
        axs.plot([aux[i,0], aux[i,3]], [aux[i,1], aux[i,4]],'-o', color='black')[0]

    for triangle in surface.bodymesh + surface.headmesh:
        vertices = [[vert.getPosition(animation,frame)[0],vert.getPosition(animation,frame)[1],vert.getPosition(animation,frame)[2]] for vert in triangle]
        vertices.append([triangle[0].getPosition(animation,frame)[0],triangle[0].getPosition(animation,frame)[1],triangle[0].getPosition(animation,frame)[2]])
        vertices = np.asarray(vertices)
        if (vertices[:,2]<0).any():
            alpha = 0.3
        else:
            alpha = 1
        #zcolor = np.abs(vertices[:,2])/max(np.abs(vertices[:,2]))
        axs.plot(vertices[:,0],vertices[:,1],'-o', color='red', markersize=5, alpha = 0.5)
        axs.set_axis_off()
        # print(vertices[:,2])
        # print('----------------------')

        if plotcapsule == True:
            for point in surface.points:
                if point.pointtype == 'limb':
                    if point.jointlock == 'RightUpLeg':
                        orientation,_ = animation.getskeletonmap().rupleg.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().rupleg.getPosition(frame)
                        p1 = animation.getskeletonmap().rlowleg.getPosition(frame)
                    elif point.jointlock == 'LeftUpLeg':
                        orientation,_ = animation.getskeletonmap().lupleg.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().lupleg.getPosition(frame)
                        p1 = animation.getskeletonmap().llowleg.getPosition(frame)
                    elif point.jointlock == 'RightLeg':
                        orientation,_ = animation.getskeletonmap().rlowleg.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().rlowleg.getPosition(frame)
                        p1 = animation.getskeletonmap().rfoot.getPosition(frame)
                    elif point.jointlock == 'LeftLeg':
                        orientation,_ = animation.getskeletonmap().llowleg.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().llowleg.getPosition(frame)
                        p1 = animation.getskeletonmap().lfoot.getPosition(frame)
                    elif point.jointlock == 'RightArm':
                        orientation,_ = animation.getskeletonmap().rarm.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().rarm.getPosition(frame)
                        p1 = animation.getskeletonmap().rforearm.getPosition(frame)
                    elif point.jointlock == 'RightForeArm':
                        orientation,_ = animation.getskeletonmap().rforearm.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().rforearm.getPosition(frame)
                        p1 = animation.getskeletonmap().rhand.getPosition(frame)
                    elif point.jointlock == 'LeftArm':
                        orientation,_ = animation.getskeletonmap().larm.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().larm.getPosition(frame)
                        p1 = animation.getskeletonmap().lforearm.getPosition(frame)
                    elif point.jointlock == 'LeftForeArm':
                        orientation,_ = animation.getskeletonmap().lforearm.getGlobalRotation(frame)
                        p0 = animation.getskeletonmap().lforearm.getPosition(frame)
                        p1 = animation.getskeletonmap().lhand.getPosition(frame)
                    capradius = point.radius
                    capaxis = p1-p0
                    caplength = np.linalg.norm(capaxis)/2 - capradius
                    center = (p1+p0)/2
                    #x = 0
                    zrange = capradius+caplength
                    z = np.arange(-zrange, zrange,0.1)
                    y = (1/4)*(np.abs(z-caplength) + np.abs(z+caplength) - 2*caplength)**2 - capradius**2
                    y2 = np.concatenate([np.sqrt(np.abs(y)),-np.sqrt(np.abs(y))])
                    z2 = np.concatenate([z,z[::-1]])
                    mRotate = mathutils.alignVectors([p1[0]-p0[0], p1[1]-p0[1], 0], [0,1,0], shape=3)
                    coordinates = np.asarray([np.dot(mRotate.T, np.asarray([y2[i],z2[i],0])) for i in range(y2.shape[0])])
                    x = coordinates[:,0] + center[0]
                    y = coordinates[:,1] + center[1]
                    axs.plot(x,y, color='gray')
    axs.set_xlim([-100,100])
    axs.set_ylim([0,200])

def MSc_PlotOrtho(which , step=0.01, epsilon = 0.0001, start = -np.pi, stop = np.pi):
    x = np.arange(start,stop,step)   # start,stop,step
    if which == 'cos':
        y = np.cos(x)
    elif which == 'molla':
        y = [np.cos(n) if np.cos(n)>epsilon else np.cos(epsilon) for n in x]
    elif which == 'molla2':
        y = [np.cos(n) if np.cos(n)>epsilon else epsilon for n in x]
    elif which == 'tonoli':
        y = (np.cos(x)+1)/2
    elif which == 'tonoli2':
        y = (np.cos(x)+1)
    elif which == 'teste':
        y = np.abs(np.cos(x))
    else:
        print('Opção errada')

    plt.axhline(0, color='gray')
    plt.axvline(np.pi/2, color='red')
    plt.axvline(0, color='gray')
    ax = plt.gca()
    if which == 'tonoli2':
        ax.set_ylim([-1.1,2.1])
        plt.axhline(1, color='red')
    else:
        plt.axhline(0.5, color='red')
        ax.set_ylim([-1.1,1.1])
    plt.plot(x,y)

def MSc_PlotCos(step):
    x = np.arange(0,4*np.pi,0.1)   # start,stop,step
    y = np.sin(x)
    plt.plot(x,y)


def MSc_PlotTrajectory(animation, animation2 = None, animation3 =None, joint = None):
    if not joint:
        joint = animation.getskeletonmap().rhand
    traj = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation.frames)])
    if animation2:
        joint = animation2.getskeletonmap().rhand
        traj2 = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation2.frames)])
    if animation3:
        joint = animation3.getskeletonmap().rhand
        traj3 = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation3.frames)])

    time = animation.frames/120
    t = np.arange(0,time,time/animation.frames)

    fig, ax = plt.subplots(figsize=(12,8), dpi=150)
    plt.plot(t,traj, label='Performer', color='black',linewidth=2.0)
    if animation2:
        plt.plot(t,traj2, label='Talita', linestyle='--', color='green',linewidth=2.0 )
    if animation3:
        plt.plot(t,traj3, label='Aragor', color='blue', linestyle='dashdot',linewidth=2.0 )

    ax.set_ylabel('Distance from the Origin ($cm$)')
    ax.set_xlabel('Time ($s$)')
    ax.legend()
    plt.tight_layout()
    fig.savefig('Trajectory', dpi=300)


def MSc_PlotTrajectoryIK(animation, animation2 = None, animation3 =None, joint = None):
    if not joint:
        joint = animation.getskeletonmap().rhand
    traj = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation.frames)])
    if animation2:
        joint = animation2.getskeletonmap().rhand
        traj2 = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation2.frames)])
    if animation3:
        joint = animation3.getskeletonmap().rhand
        traj3 = np.asarray([ np.linalg.norm(joint.getPosition(frame)) for frame in range(animation3.frames)])

    time = animation.frames/120
    t = np.arange(0,time,time/animation.frames)

    fig, ax = plt.subplots(figsize=(12,8), dpi=150)
    plt.plot(t,traj, label='Performer', color='black',linewidth=2.0 )
    if animation2:
        plt.plot(t,traj2, label='Initial Retargeting', linestyle='--', color='red',linewidth=2.0 )
    if animation3:
        plt.plot(t,traj3, label='Complete Retargeting', color='blue',linewidth=2.0 )

    ax.set_ylabel('Distance from the Origin ($cm$)')
    ax.set_xlabel('Time ($s$)')
    ax.legend()
    plt.tight_layout()
    fig.savefig('Trajectory', dpi=300)


def MSc_PlotImportance(animation, ego, joint=None):
    if not joint:
        joint = animation.getskeletonmap().rhand
        # ego = ego[0]

    # lenHead = 13
    # lenBody = 9
    # lenLimb = 6
    imphead = [sum(ego.framecoord[i].importance[:13]) for i in range(animation.frames)]
    impbody = [sum(ego.framecoord[i].importance[13:22]) for i in range(animation.frames)]

    time = animation.frames/120
    t = np.arange(0,time,time/animation.frames)

    fig, ax = plt.subplots(figsize=(8,8), dpi=150)
    plt.plot(t,imphead, label='Head', color='green')
    plt.plot(t,impbody, label='Body', color='blue')
    t1 = 450/120
    t2 = 650/120
    plt.plot([t1,t1],[0,1], color='red')
    plt.plot([t2,t2],[0,1], color='red')
    ax.text(t1+0.05,0.12, '(a)')
    ax.text(t2+0.05,0.12, '(b)')
    #plt.plot(t,traj3, label='Proposed Retargeting')
    ax.set_ylim([0.1,0.7])
    ax.set_ylabel('Importance Sum')
    ax.set_xlabel('Time ($s$)')
    ax.legend()
    plt.tight_layout()
    fig.savefig('Importance', dpi=300)
