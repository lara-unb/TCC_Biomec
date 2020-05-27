from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys

class Visualizer(object):
    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        # self.w.opts['distance'] = 40
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(0, 110, 1920, 1080)
        # self.w.show()

        self.g = gl.GLGridItem()
        self.w.addItem(self.g)

    def plot_data(self, keypoints):
        scatter_plot_item = gl.GLScatterPlotItem(pos=keypoints[:,:])
        self.w.addItem(scatter_plot_item)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    keypoints = np.random.randint(-10,10,size=(10,3))
    keypoints[:,2] = np.abs(keypoints[:,2])
    v = Visualizer()
    v.plot_data(keypoints)
    v.w.show()
    QtGui.QApplication.instance().exec_()