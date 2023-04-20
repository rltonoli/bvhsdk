# Author: https://stackoverflow.com/questions/36665850/matplotlib-animation-inside-your-own-gui
###################################################################
#                                                                 #
#                     PLOTTING A LIVE GRAPH                       #
#                  ----------------------------                   #
#            EMBED A MATPLOTLIB ANIMATION INSIDE YOUR             #
#            OWN GUI!                                             #
#                                                                 #
###################################################################


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as matplotanim
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
from .. import anim
import time

#My imports
from mpl_toolkits.mplot3d import Axes3D


def start(animation):
    assert type(animation) == anim.Animation
    GUIPlayer(animation)
    #print('Computing positions...')
    #all_frames_bones_data = []
    #for frame in animation.frames:
    #    all_frames_bones_data.append(animation.getBones(frame))

class GUIPlayer():
    #def Plot3DAnimation(animation):
    """
    Plot animation as skeleton

    :type animation: Animation class object
    :param animation: Animation to be draw
    """
    def __init__(self, animation):
        self.animation = animation
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')

        self.play = True
        self.forward = True
        self.current_frame = 0
        self.current_fps = 0
        self.count_frame = 0
        self.count_frame_time = time.time()

        self.lines = []
        #maxdata = -np.inf
        #mindata = np.inf
        bones = animation.getBones(frame = 0)
        for bone in bones:
            self.lines.append(
                ax.plot([ bone[0], bone[3] ], [ bone[1], bone[4] ], [ bone[2], bone[5] ])
            )

        ax.set_xlim( (-150,150) )
        ax.set_ylim( (-50,200) )
        ax.set_zlim( (-150,150) )
        self.fps_text = ax.text(x= ax.get_xlim()[1]-1, y= ax.get_ylim()[1]-1, z=0, s=str(self.current_fps))
        #for joint in animation.getlistofjoints():
        #    position = joint.getPosition(frame = 0)
        #    scatters.append(ax.plot([position[0]],[position[1]],[position[2]],'o', color='red', markersize=1)[0])
        #    if np.min(position)<mindata:
        #        mindata = np.min(position)
        #    if np.max(position)>maxdata:
        #        maxdata = np.max(position)
        ax.view_init(elev=100, azim=-90)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        #ax.set_xlim(mindata,maxdata)
        #ax.set_ylim(mindata,maxdata)
        #ax.set_zlim(mindata,maxdata)

        axes_framedisplaytext = fig.add_axes([0.6, 0.05, 0.05, 0.05])
        axes_onebackward = fig.add_axes([0.65, 0.05, 0.05, 0.05]) #[left, bottom, width, height]
        axes_backward = fig.add_axes([0.7, 0.05, 0.05, 0.05]) #[left, bottom, width, height]
        axes_pause = fig.add_axes([0.75, 0.05, 0.05, 0.05])
        axes_forward = fig.add_axes([0.8, 0.05, 0.05, 0.05])
        axes_oneforward = fig.add_axes([0.85, 0.05, 0.05, 0.05])
        
        self.textbox_framedisplay = TextBox(axes_framedisplaytext, label='frame', initial='0')
        self.button_oneback = Button(axes_onebackward, label='$\u29CF$')
        self.button_backward = Button(axes_backward, label='$\u25C0$')
        self.button_pause = Button(axes_pause, label='$\u2016$')
        self.button_forward = Button(axes_forward, label='$\u25B6$')
        self.button_oneforw= Button(axes_oneforward, label='$\u29D0$')

        self.textbox_framedisplay.on_submit(self.onchange_textbox_framedisplay)
        self.button_oneback.on_clicked(self.click_onebackward)
        self.button_backward.on_clicked(self.click_backward)
        self.button_pause.on_clicked(self.click_pause)
        self.button_forward.on_clicked(self.click_forward)
        self.button_oneforw.on_clicked(self.click_oneforward)
        #ani = matplotanim.FuncAnimation(fig, self.update, frames=np.arange(animation.frames), fargs=([self.lines]) ,interval=1, blit=False)
        ani = matplotanim.FuncAnimation(fig, self.update, frames=self.funcplay(), fargs=() ,interval=1, blit=False)
        plt.show()
        #return ani

    def onchange_textbox_framedisplay(self, event):
        try:
            self.current_frame = int(self.textbox_framedisplay.text)
        except:
            self.textbox_framedisplay.set_val( str(self.current_frame) )

    def click_oneforward(self, event):
        self.current_frame += 1
        self.play = True

    def click_onebackward(self, event):
        self.current_frame -= 1
        self.play = True

    def click_backward(self, event):
        self.forward = False
        self.play = True

    def click_pause(self, event):
        self.play = not self.play

    def click_forward(self, event):
        self.forward = True
        self.play = True

    def funcplay(self):
        while True:
            if self.play:
                self.current_frame += self.forward - (not self.forward)
            if self.current_frame > self.animation.frames:
                self.current_frame = 0
            yield self.current_frame


    def update(self, frame):
        time1 = time.time()

        # TODO: maybe not the best way to skip update during pause
        if self.play:
            print(frame)
            
            if self.count_frame == 10:
                delta = time.time()-self.count_frame_time
                self.fps_text.set_text(s="fps: {0:.2f}".format( 10 / (delta)))
                self.count_frame = 0
                self.count_frame_time = time.time()
            self.count_frame += 1

            self.textbox_framedisplay.set_val( str(frame) )

            bones = self.animation.getBones(frame)
            for line, bone in zip(self.lines,bones):
                line[0].set_data([ bone[0], bone[3] ], [ bone[1], bone[4] ])
                line[0].set_3d_properties( [bone[2], bone[5]])


        return self.lines
    

def showinfo(animation):
    assert type(animation) == anim.Animation
    gui = GUIShowInfo(animation)
    return gui
    
class GUIShowInfo():


    def __init__(self, animation):
            self.animation = animation
            
            #fig, axs = plt.figure(3, 3, figsize=(12,8))
            fig = plt.figure(figsize=(12,8), tight_layout=True)
            gs = gridspec.GridSpec(3, 3, figure=fig)
            

            positions = np.empty(shape=(len(animation.getlistofjoints()), 3))
            offsets = np.empty(shape=(len(animation.getlistofjoints()), 3))
            for i, joint in enumerate(animation.getlistofjoints()):
                positions[i] = joint.getPosition(frame = 0)
                offsets[i] = joint.offset


            axs = []
            lab = {0: 'X axis', 1: 'Y axis', 2: 'Z axis'}
            for i in range(3):
                xyz = [[0,1], [1,2], [0,2]][i]
                axs.append(fig.add_subplot(gs[i, 1]))
                axs[-1].scatter(positions[:,xyz[0]], positions[:,xyz[1]], c='r', marker='o')
                axs[-1].set_aspect('equal')
                axs[-1].set_xlabel(lab[xyz[0]])
                axs[-1].set_ylabel(lab[xyz[1]])

            axs = []
            for i in range(3):
                xyz = [[0,1], [1,2], [0,2]][i]
                axs.append(fig.add_subplot(gs[i, 2]))
                axs[-1].scatter(offsets[:,xyz[0]], positions[:,xyz[1]], c='r', marker='o')
                axs[-1].set_aspect('equal')
                axs[-1].set_xlabel(lab[xyz[0]])
                axs[-1].set_ylabel(lab[xyz[1]])

            axs.append(fig.add_subplot(gs[:, 0]))
            for i, joint in enumerate(animation.root.printHierarchy()):
                axs[-1].text(0, -i*0.1, str(i) + " " + joint, fontsize=10)
            axs[-1].set_ylim(-len(animation.getlistofjoints())*0.1, 0.1)
            axs[-1].set_axis_off()


            plt.show()

            
