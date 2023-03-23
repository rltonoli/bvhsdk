# Author: https://stackoverflow.com/questions/36665850/matplotlib-animation-inside-your-own-gui
###################################################################
#                                                                 #
#                     PLOTTING A LIVE GRAPH                       #
#                  ----------------------------                   #
#            EMBED A MATPLOTLIB ANIMATION INSIDE YOUR             #
#            OWN GUI!                                             #
#                                                                 #
###################################################################


import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D


from .. import anim

matplotlib.use("Qt5Agg")

#My imports
from mpl_toolkits.mplot3d import Axes3D


def setCustomSize(x, width, height):
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(x.sizePolicy().hasHeightForWidth())
    x.setSizePolicy(sizePolicy)
    x.setMaximumSize(QtCore.QSize(width, height))


class CustomMainWindow(QtWidgets.QMainWindow):
    def __init__(self, animation):
        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("Anim Plot")

        # Create FRAME_A
        self.FRAME_A = QtWidgets.QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210, 210, 235, 255).name())
        self.LAYOUT_A = QtWidgets.QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)

        # Place the zoom button
        self.zoomBtn = QtWidgets.QPushButton(text='zoom')
        setCustomSize(self.zoomBtn, 100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0, 0))

        # Place the matplotlib figure
        self.myFig = CustomFigCanvas(animation)
        self.LAYOUT_A.addWidget(self.myFig, *(0, 1))

        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name='myDataLoop', target=dataSendLoop, daemon=True, args=(self.addData_callbackFunc, animation))
        myDataLoop.start()

        self.show()

    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self, animation):
        self.addedData = []

        self.animation = animation
        print('Matplotlib Version:', matplotlib.__version__)

        # The data
        self.n = np.linspace(0, self.animation.frames - 1, self.animation.frames)
        self.y = []

        # The window
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(111, projection='3d')
        print(self.ax1)
        self.ax1.view_init(elev=100, azim=-90, roll=0, vertical_axis='y')

        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit=True)

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        self.lines =[]
        self.ax1.cla()
        teste = self.animation.getBones(frame=0)
        for bone in teste:
            self.lines.append(
                self.ax1.plot([ bone[0], bone[3] ], [ bone[1], bone[4] ], [ bone[2], bone[5] ])
            )

    def addData(self, value):
        print(value.shape)
        self.addedData.append(value)

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom, top)
        self.draw()

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        print(framedata)
        
        #teste = self.animation.getBones()
        if len(self.addedData)>0:
            for bone, line in zip(self.addedData[0], self.lines):
                line[0].set_data( [bone[0], bone[3] ], [ bone[1], bone[4] ] )
                line[0].set_3d_properties( [ bone[2], bone[5] ] )
            del(self.addedData[0])

        #margin = 2
        #while(len(self.addedData) > 0):
        #    self.y = np.roll(self.y, -1)
        #    self.y[-1] = self.addedData[0]
        #    del(self.addedData[0])
#
        #self.line1.set_data(self.n[0:self.n.size - margin], self.y[0:self.n.size - margin])
        #self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        #self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        #self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(np.ndarray)


def dataSendLoop(addData_callbackFunc, animation):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    # Simulate some data
    n = np.linspace(0, 499, 500)
    y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
    i = 0

    
    for i in range(0,animation.frames):
        teste = animation.getBones(i)
        mySrc.data_signal.emit(teste)  # <- Here you emit a signal!
        time.sleep(0.1)
        i += 1

def start(animation):
    assert type(animation) == anim.Animation
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow(animation)

    sys.exit(app.exec_())

#if __name__ == '__main__':
#    app = QtWidgets.QApplication(sys.argv)
#    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
#    myGUI = CustomMainWindow()
#
#    sys.exit(app.exec_())
