import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
class GraphWidget(pg.GraphicsWindow):
    def __init__(self, parent=None, **kargs):
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        pg.GraphicsWindow.__init__(self, **kargs)
        self.setParent(parent)
        self.setWindowTitle('Aquisition')
    
if __name__ == '__main__':
    w = GraphWidget()
    w.show()
    QtGui.QApplication.instance().exec_()