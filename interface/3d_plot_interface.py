from PyQt5 import QtWidgets, uic
import pyqtgraph.opengl as gl
import sys
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        #Load the UI Page
        uic.loadUi('MainWindow.ui', self)

        self.setup2d_graphs()
        self.setup3d_inv()

        self.plot([1,2,3,4,5,6,7,8,9,10], [30,32,34,32,33,31,29,32,35,45])

    def setup3d_inv(self):
        self.graphWidget.setCameraPosition(distance=50)
        ## Add a grid to the view
        g = gl.GLGridItem()
        g.scale(1, 1, 1)
        g.setDepthValue(100)  # draw grid after surfaces since they may be translucent
        self.graphWidget.addItem(g)

    def plot(self, hour, temperature):
        Z = np.ones((2, 2))
        p1 = gl.GLSurfacePlotItem(z=Z, shader='shaded', color=(0.5, 0.5, 1, 1))
        self.graphWidget.addItem(p1)

        # self.graphWidget.plot(hour, temperature)
        a = np.random.randn(150)
        b = np.random.randn(150)
        self.graphWidget_2D.plot(np.sort(a), np.sort(b))

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
 
if __name__ == '__main__':         
    main()