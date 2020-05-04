# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\interface.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(858, 716)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.img_label_upper = QtWidgets.QLabel(self.centralwidget)
        self.img_label_upper.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_label_upper.sizePolicy().hasHeightForWidth())
        self.img_label_upper.setSizePolicy(sizePolicy)
        self.img_label_upper.setMinimumSize(QtCore.QSize(400, 380))
        self.img_label_upper.setAutoFillBackground(False)
        self.img_label_upper.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.img_label_upper.setText("")
        self.img_label_upper.setObjectName("img_label_upper")
        self.horizontalLayout_7.addWidget(self.img_label_upper)
        self.window_plot_upper = GraphWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_plot_upper.sizePolicy().hasHeightForWidth())
        self.window_plot_upper.setSizePolicy(sizePolicy)
        self.window_plot_upper.setMinimumSize(QtCore.QSize(0, 400))
        self.window_plot_upper.setObjectName("window_plot_upper")
        self.horizontalLayout_7.addWidget(self.window_plot_upper)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.img_label_lower = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_label_lower.sizePolicy().hasHeightForWidth())
        self.img_label_lower.setSizePolicy(sizePolicy)
        self.img_label_lower.setMinimumSize(QtCore.QSize(400, 380))
        self.img_label_lower.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.img_label_lower.setText("")
        self.img_label_lower.setObjectName("img_label_lower")
        self.horizontalLayout_8.addWidget(self.img_label_lower)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.window_plot_lower = GraphWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_plot_lower.sizePolicy().hasHeightForWidth())
        self.window_plot_lower.setSizePolicy(sizePolicy)
        self.window_plot_lower.setObjectName("window_plot_lower")
        self.horizontalLayout_11.addWidget(self.window_plot_lower)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_11)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pp_button = QtWidgets.QPushButton(self.centralwidget)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../Artigo_remo/ema_fes_rowing_experiments/icons/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pp_button.setIcon(icon)
        self.pp_button.setObjectName("pp_button")
        self.horizontalLayout.addWidget(self.pp_button)
        self.speed_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.speed_label.sizePolicy().hasHeightForWidth())
        self.speed_label.setSizePolicy(sizePolicy)
        self.speed_label.setObjectName("speed_label")
        self.horizontalLayout.addWidget(self.speed_label)
        self.speed_n = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.speed_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.speed_n.setMinimum(0.25)
        self.speed_n.setSingleStep(0.25)
        self.speed_n.setProperty("value", 1.0)
        self.speed_n.setObjectName("speed_n")
        self.horizontalLayout.addWidget(self.speed_n)
        self.frame_slider = QtWidgets.QSlider(self.centralwidget)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("frame_slider")
        self.horizontalLayout.addWidget(self.frame_slider)
        self.frame_n = QtWidgets.QSpinBox(self.centralwidget)
        self.frame_n.setMaximum(99)
        self.frame_n.setObjectName("frame_n")
        self.horizontalLayout.addWidget(self.frame_n)
        self.time_video = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.time_video.sizePolicy().hasHeightForWidth())
        self.time_video.setSizePolicy(sizePolicy)
        self.time_video.setObjectName("time_video")
        self.horizontalLayout.addWidget(self.time_video)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 858, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuPlots = QtWidgets.QMenu(self.menubar)
        self.menuPlots.setObjectName("menuPlots")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionEdit_Video_Settings = QtWidgets.QAction(MainWindow)
        self.actionEdit_Video_Settings.setObjectName("actionEdit_Video_Settings")
        self.actionEdit_Plots = QtWidgets.QAction(MainWindow)
        self.actionEdit_Plots.setObjectName("actionEdit_Plots")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionPlots = QtWidgets.QAction(MainWindow)
        self.actionPlots.setObjectName("actionPlots")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_as)
        self.menuPlots.addAction(self.actionEdit_Plots)
        self.menuPlots.addAction(self.actionPlots)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuPlots.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pp_button.setText(_translate("MainWindow", "Play"))
        self.speed_label.setText(_translate("MainWindow", "Speed:"))
        self.time_video.setText(_translate("MainWindow", "00:00 / 00:00"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuPlots.setTitle(_translate("MainWindow", "Edit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionEdit_Video_Settings.setText(_translate("MainWindow", "Edit Video"))
        self.actionEdit_Plots.setText(_translate("MainWindow", "Video"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave_as.setText(_translate("MainWindow", "Save as"))
        self.actionPlots.setText(_translate("MainWindow", "Plots"))
from pyqtgraph_class import GraphWidget
