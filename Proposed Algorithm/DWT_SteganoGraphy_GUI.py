# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_Interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from random import seed
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QIcon, QPixmap
from numpy import Inf

import DWT_SteganoGraphy as stego
import cv2


class previewImageWindow(QWidget):
    def __init__(self, imagePath):
        super().__init__()
        # self.resize(811, 575)
        layout = QVBoxLayout()
        self.label = QtWidgets.QLabel()
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.setWindowTitle = imagePath
        self.resize(pixmap.width(), pixmap.height())
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.show()


class Ui_MainWindow(object):

    def onChangeTab(self, i):
        if i == 0:
            self.label.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#2975d6;\">Encode</span></p></body></html>")
        elif i == 1:
            self.label.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#2975d6;\">Decode</span></p></body></html>")
        else:
            self.label.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#2975d6;\">Get Performance</span></p></body></html>")

     # Function to Display Message/Error/Information
    def displayMsg(self, title, msg, ico_type=None):
        MsgBox = QMessageBox()
        MsgBox.setText(msg)
        MsgBox.setWindowTitle(title)
        if ico_type == 'err':
            ico = QMessageBox.Critical
        else:
            ico = QMessageBox.Information
        MsgBox.setIcon(ico)
        MsgBox.exec()

    # Function to Choose Input File
    def getFile(self):
        file_Path = QFileDialog.getOpenFileName(
            None, 'Open File', '', "Image Files(*.jpg *.jpeg *.png *.bmp)")[0]
        if file_Path != '':
            self.inputFilePath.setText(file_Path)

    # Function to Choose Input File for Compare
    def getResultFile(self):
        file_Path = QFileDialog.getOpenFileName(
            None, 'Open File', '', "Image Files(*.jpg *.jpeg *.png *.bmp)")[0]
        if file_Path != '':
            self.resultInputFilePath.setText(file_Path)

    # Function to Choose Fractional Input File

    def getFractionalFile(self):
        file_Path = QFileDialog.getOpenFileName(
            None, 'Open File', '', "Numpy Array File(*.npy)")[0]
        if file_Path != '':
            self.inputFractionalFilePath.setText(file_Path)

    # Function to Display save File dialog
    def saveFile(self):
        output_Path = QFileDialog.getSaveFileName(
            None, 'Save encoded file', '', "PNG File(*.png)")[0]
        return output_Path

    # Function to Display Save Fractional Dialog
    def save_FractionalFile(self):
        output_Path = QFileDialog.getSaveFileName(
            None, 'Save encoded file', '', "NUMPY File(*.npy)")[0]
        return output_Path

    # Function to Preview Image
    def previewFile(self):
        if(self.inputFilePath.text() == ''):
            self.displayMsg('Error: No file chosen',
                            'You must select input image file!', 'err')
        else:
            imagePath = self.inputFilePath.text()
            self.w = previewImageWindow(imagePath)

    # Function to Preview Result Image
    def resultPreviewFile(self):
        if(self.resultInputFilePath.text() == ''):
            self.displayMsg('Error: No file chosen',
                            'You must select input image file!', 'err')
        else:
            # global imagePath
            imagePath = self.resultInputFilePath.text()
            self.w = previewImageWindow(imagePath)

    # Function to Encode the Data & Save File
    def encode(self):
        input_Path = self.inputFilePath.text()
        text = self.encodeText.toPlainText()

        if input_Path == '':
            self.displayMsg('Error: No file chosen',
                            'You must select input image file!', 'err')
        elif text == '':
            self.displayMsg('Text is empty', 'Please enter some text to hide!')
        else:
            output_Path = self.saveFile()
            output_FractionalPath = self.save_FractionalFile()

            if output_Path == '':
                self.displayMsg('Operation cancelled',
                                'Operation cancelled by user!')
            if output_FractionalPath == '':
                self.displayMsg('Operation cancelled',
                                'Operation cancelled by user!')
            else:
                try:
                    seed_value = stego.encode(
                        input_Path, text, output_Path, output_FractionalPath, self.encodeProgressBar)
                except stego.FileError as fe:
                    self.displayMsg('File Error', str(fe), 'err')
                except stego.DataError as de:
                    self.displayMsg('Data Error', str(de), 'err')
                except stego.SkinNotDetected as se:
                    self.displayMsg("Skin Detection Error", str(se), 'err')
                except stego.LargestComponentNotFound as le:
                    self.displayMsg("Largest Component Error", str(le), 'err')
                except stego.NotEnoughCapasity as ce:
                    self.displayMsg("Capasity Error", str(ce), 'err')
                else:
                    self.displayMsg(
                        'Success', "Encoded Successfully!\n\nYour Seed Value is = '{}'".format(seed_value))
                    # self.encodeProgressBar.setValue(100)

    # Function to Decode the Data
    def decode(self):
        input_Path = self.inputFilePath.text()
        input_FractionalPath = self.inputFractionalFilePath.text()
        seed_value = self.decodeSeed.text()

        if input_Path == '':
            self.displayMsg('Error: No file chosen',
                            'You must select input image file!', 'err')
        elif input_FractionalPath == '':
            self.displayMsg('Error: No file chosen',
                            'You must select input Numpy Array file!', 'err')
        elif seed_value == '':
            self.displayMsg('Error: No Seed Given',
                            'You must enter Seed Value!', 'err')
        else:
            try:
                data = stego.decode(
                    input_Path, input_FractionalPath, seed_value, self.decodeProgressBar)
            except stego.FileError as fe:
                self.displayMsg('File Error', str(fe), 'err')
            except stego.SeedNotValid as se:
                self.displayMsg('Seed Error', str(se), 'err')
            except stego.SkinNotDetected as se:
                self.displayMsg("Skin Detection Error", str(se), 'err')
            except stego.LargestComponentNotFound as le:
                self.displayMsg("Largest Component Error", str(le), 'err')
            else:
                self.displayMsg('Success', 'Decoded Successfully!')
                self.decodeText.document().setPlainText(data)

    # Function to Comapare Image & Get Performance of Out Algorithm
    def compareImage(self):
        self.resultLabel_1.setText("Peak Signal-to-Noise Ratio (PSNR) : ")
        self.resultLabel_2.setText(
            "Structural Similarity Index Metric (SSIM):")
        self.resultLabel_3.setText("Universal Image Quality Index : ")
        input_Path_1 = self.inputFilePath.text()
        input_Path_2 = self.resultInputFilePath.text()
        if input_Path_1 == '':
            self.displayMsg('Error: No file chosen',
                            'You must select input image file!', 'err')
        elif input_Path_2 == '':
            self.displayMsg('Error: No file chosen',
                            'You must select input image file for Compare!', 'err')
        elif (cv2.imread(input_Path_1)).shape != (cv2.imread(input_Path_2)).shape:
            self.displayMsg('Error: Wrong files chosen',
                            'You must select Both Images with same Dimension for Compare!', 'err')
        else:
            comparison = cv2.imread(input_Path_1) == cv2.imread(input_Path_2)
            truth_Value = comparison.all()
            if (input_Path_1 == input_Path_2) or truth_Value == True:
                self.displayMsg('Error: Same files chosen',
                                'You must select 2 different Images to Compare!')
                PSNR = Inf
                SSIM = 1
                UQI = 1
            else:
                PSNR = stego.get_psnr(input_Path_1, input_Path_2)
                SSIM = stego.get_ssim(input_Path_1, input_Path_2)
                UQI = stego.get_uqi(input_Path_1, input_Path_2)

            self.resultLabel_1.setText(self.resultLabel_1.text() + str(PSNR))
            self.resultLabel_2.setText(self.resultLabel_2.text() + str(SSIM))
            self.resultLabel_3.setText(self.resultLabel_3.text() + str(UQI))

    # region

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(675, 627)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.inputFilePath = QtWidgets.QLineEdit(self.centralwidget)
        self.inputFilePath.setObjectName("inputFilePath")
        self.horizontalLayout_2.addWidget(self.inputFilePath)
        self.chooseFile = QtWidgets.QPushButton(self.centralwidget)
        self.chooseFile.setAutoFillBackground(False)
        self.chooseFile.setStyleSheet(
            "background-color: #7bc7de;color: rgb(255, 255, 255);")
        self.chooseFile.setAutoDefault(True)
        self.chooseFile.setObjectName("chooseFile")
        self.horizontalLayout_2.addWidget(self.chooseFile)
        self.preview = QtWidgets.QPushButton(self.centralwidget)
        self.preview.setStyleSheet("background-color: #009B81;\n"
                                   "color: rgb(255, 255, 255);")
        self.preview.setObjectName("preview")
        self.horizontalLayout_2.addWidget(self.preview)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_5.addWidget(self.line_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_5.addWidget(self.line_2)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.encodeTab = QtWidgets.QWidget()
        self.encodeTab.setObjectName("encodeTab")
        self.gridLayout = QtWidgets.QGridLayout(self.encodeTab)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.encodeTab)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_6.addWidget(self.label_6)
        self.label_5 = QtWidgets.QLabel(self.encodeTab)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_6.addWidget(self.label_5)
        self.encodeText = QtWidgets.QPlainTextEdit(self.encodeTab)
        self.encodeText.setObjectName("encodeText")
        self.verticalLayout_6.addWidget(self.encodeText)
        spacerItem2 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem2)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_7 = QtWidgets.QLabel(self.encodeTab)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_13.addWidget(self.label_7)
        self.encodeProgressBar = QtWidgets.QProgressBar(self.encodeTab)
        self.encodeProgressBar.setProperty("value", 0)
        self.encodeProgressBar.setObjectName("encodeProgressBar")
        self.horizontalLayout_13.addWidget(self.encodeProgressBar)
        self.verticalLayout_6.addLayout(self.horizontalLayout_13)
        spacerItem3 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem3)
        self.encodeBtn = QtWidgets.QPushButton(self.encodeTab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.encodeBtn.sizePolicy().hasHeightForWidth())
        self.encodeBtn.setSizePolicy(sizePolicy)
        self.encodeBtn.setStyleSheet("background-color: rgb(41, 117, 214);\n"
                                     "color: rgb(255, 255, 255);\n"
                                     "")
        self.encodeBtn.setObjectName("encodeBtn")
        self.verticalLayout_6.addWidget(
            self.encodeBtn, 0, QtCore.Qt.AlignHCenter)
        self.gridLayout_2.addLayout(self.verticalLayout_6, 0, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 0, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem5, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.encodeTab, "")
        self.decodeTab = QtWidgets.QWidget()
        self.decodeTab.setObjectName("decodeTab")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.decodeTab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem7 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem7, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.decodeTab)
        self.label_13.setObjectName("label_13")
        self.gridLayout_4.addWidget(self.label_13, 3, 0, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem8, 5, 0, 1, 1)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_15 = QtWidgets.QLabel(self.decodeTab)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_17.addWidget(self.label_15)
        self.decodeProgressBar = QtWidgets.QProgressBar(self.decodeTab)
        self.decodeProgressBar.setProperty("value", 0)
        self.decodeProgressBar.setObjectName("decodeProgressBar")
        self.horizontalLayout_17.addWidget(self.decodeProgressBar)
        self.gridLayout_4.addLayout(self.horizontalLayout_17, 8, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.decodeTab)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 0, 0, 1, 1)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_12 = QtWidgets.QLabel(self.decodeTab)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_15.addWidget(self.label_12)
        self.inputFractionalFilePath = QtWidgets.QLineEdit(self.decodeTab)
        self.inputFractionalFilePath.setObjectName("inputFractionalFilePath")
        self.horizontalLayout_15.addWidget(self.inputFractionalFilePath)
        self.chooseFractionalFile = QtWidgets.QPushButton(self.decodeTab)
        self.chooseFractionalFile.setStyleSheet("background-color: #7bc7de;\n"
                                                "color: rgb(255, 255, 255);")
        self.chooseFractionalFile.setObjectName("chooseFractionalFile")
        self.horizontalLayout_15.addWidget(self.chooseFractionalFile)
        self.gridLayout_4.addLayout(self.horizontalLayout_15, 1, 0, 1, 1)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_14 = QtWidgets.QLabel(self.decodeTab)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_16.addWidget(self.label_14)
        self.decodeSeed = QtWidgets.QLineEdit(self.decodeTab)
        self.decodeSeed.setObjectName("decodeSeed")
        self.horizontalLayout_16.addWidget(self.decodeSeed)
        self.decodeSeedCheckBox = QtWidgets.QCheckBox(self.decodeTab)
        self.decodeSeedCheckBox.setObjectName("decodeSeedCheckBox")
        self.horizontalLayout_16.addWidget(self.decodeSeedCheckBox)
        self.gridLayout_4.addLayout(self.horizontalLayout_16, 4, 0, 1, 1)
        self.decodeText = QtWidgets.QPlainTextEdit(self.decodeTab)
        self.decodeText.setObjectName("decodeText")
        self.gridLayout_4.addWidget(self.decodeText, 11, 0, 1, 1)
        self.decodeBtn = QtWidgets.QPushButton(self.decodeTab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.decodeBtn.sizePolicy().hasHeightForWidth())
        self.decodeBtn.setSizePolicy(sizePolicy)
        self.decodeBtn.setStyleSheet("background-color: rgb(41, 166, 74);\n"
                                     "color: rgb(255, 255, 255);\n"
                                     "")
        self.decodeBtn.setObjectName("decodeBtn")
        self.gridLayout_4.addWidget(
            self.decodeBtn, 6, 0, 1, 1, QtCore.Qt.AlignHCenter)
        spacerItem9 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem9, 7, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem10, 9, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.decodeTab)
        self.label_16.setObjectName("label_16")
        self.gridLayout_4.addWidget(self.label_16, 10, 0, 1, 1)
        self.horizontalLayout_3.addLayout(self.gridLayout_4)
        spacerItem11 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem11)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.decodeTab, "")
        self.resultTab = QtWidgets.QWidget()
        self.resultTab.setObjectName("resultTab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.resultTab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 622, 339))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.horizontalLayoutWidget)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem12 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem12, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem13 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem13)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_8 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_7.addWidget(self.label_8)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_17 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_19.addWidget(self.label_17)
        self.resultInputFilePath = QtWidgets.QLineEdit(
            self.horizontalLayoutWidget)
        self.resultInputFilePath.setObjectName("resultInputFilePath")
        self.horizontalLayout_19.addWidget(self.resultInputFilePath)
        self.resultChooseFile = QtWidgets.QPushButton(
            self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.resultChooseFile.sizePolicy().hasHeightForWidth())
        self.resultChooseFile.setSizePolicy(sizePolicy)
        self.resultChooseFile.setStyleSheet(
            "background-color: #7bc7de;color: rgb(255, 255, 255);")
        self.resultChooseFile.setObjectName("resultChooseFile")
        self.horizontalLayout_19.addWidget(self.resultChooseFile)
        self.resultPreview = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.resultPreview.sizePolicy().hasHeightForWidth())
        self.resultPreview.setSizePolicy(sizePolicy)
        self.resultPreview.setStyleSheet("background-color: #009B81;\n"
                                         "color: rgb(255, 255, 255);")
        self.resultPreview.setObjectName("resultPreview")
        self.horizontalLayout_19.addWidget(self.resultPreview)
        self.verticalLayout_9.addLayout(self.horizontalLayout_19)
        self.verticalLayout_7.addLayout(self.verticalLayout_9)
        self.horizontalLayout_10.addLayout(self.verticalLayout_7)
        spacerItem14 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem14)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        spacerItem15 = QtWidgets.QSpacerItem(
            20, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem15)
        self.resultBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.resultBtn.sizePolicy().hasHeightForWidth())
        self.resultBtn.setSizePolicy(sizePolicy)
        self.resultBtn.setStyleSheet(
            "background-color: #9a66cb;color: rgb(255, 255, 255);")
        self.resultBtn.setObjectName("resultBtn")
        self.verticalLayout.addWidget(
            self.resultBtn, 0, QtCore.Qt.AlignHCenter)
        spacerItem16 = QtWidgets.QSpacerItem(
            20, 17, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem16)
        self.line_5 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout.addWidget(self.line_5)
        spacerItem17 = QtWidgets.QSpacerItem(
            20, 17, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem17)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem18 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem18)
        self.label_23 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_4.addWidget(self.label_23)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem19 = QtWidgets.QSpacerItem(
            20, 17, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem19)
        self.resultLabel_1 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.resultLabel_1.setObjectName("resultLabel_1")
        self.verticalLayout.addWidget(
            self.resultLabel_1, 0, QtCore.Qt.AlignHCenter)
        spacerItem20 = QtWidgets.QSpacerItem(
            20, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem20)
        self.resultLabel_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.resultLabel_2.setObjectName("resultLabel_2")
        self.verticalLayout.addWidget(
            self.resultLabel_2, 0, QtCore.Qt.AlignHCenter)
        spacerItem21 = QtWidgets.QSpacerItem(
            20, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem21)
        self.resultLabel_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.resultLabel_3.setObjectName("resultLabel_3")
        self.verticalLayout.addWidget(
            self.resultLabel_3, 0, QtCore.Qt.AlignHCenter)
        spacerItem22 = QtWidgets.QSpacerItem(
            20, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem22)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 1, 1, 1)
        spacerItem23 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem23, 0, 2, 1, 1)
        self.tabWidget.addTab(self.resultTab, "")
        self.verticalLayout_5.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 675, 29))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuResults = QtWidgets.QMenu(self.menubar)
        self.menuResults.setObjectName("menuResults")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionComparison = QtWidgets.QAction(MainWindow)
        self.actionComparison.setObjectName("actionComparison")
        self.actionVisit_us = QtWidgets.QAction(MainWindow)
        self.actionVisit_us.setObjectName("actionVisit_us")
        self.menuHelp.addAction(self.actionAbout)
        self.menuHelp.addAction(self.actionVisit_us)
        self.menuResults.addAction(self.actionComparison)
        self.menubar.addAction(self.menuResults.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # endregion

        # Slots
        # now listen the currentChanged signal
        # self.tabWidget.blockSignals(False)
        self.tabWidget.currentChanged.connect(self.onChangeTab)  # changed!
        self.chooseFile.clicked.connect(self.getFile)
        self.preview.clicked.connect(self.previewFile)
        self.encodeBtn.clicked.connect(self.encode)
        self.chooseFractionalFile.clicked.connect(self.getFractionalFile)
        self.decodeBtn.clicked.connect(self.decode)
        self.resultChooseFile.clicked.connect(self.getResultFile)
        self.resultPreview.clicked.connect(self.resultPreviewFile)
        self.resultBtn.clicked.connect(self.compareImage)
        self.decodeSeedCheckBox.stateChanged.connect(lambda: self.decodeSeed.setEchoMode(
            QtWidgets.QLineEdit.Normal) if self.decodeSeedCheckBox.isChecked() else self.decodeSeed.setEchoMode(QtWidgets.QLineEdit.Password))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Steganography Software"))
        self.label_4.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Step 1:</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "Input Image File:"))
        self.chooseFile.setText(_translate("MainWindow", "Choose File"))
        self.preview.setText(_translate("MainWindow", "Preview"))
        self.label.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#2975d6;\">Encode</span></p></body></html>"))
        self.label_6.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Step 2 : </span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Enter text to hide :"))
        self.label_7.setText(_translate("MainWindow", "Progress :"))
        self.encodeBtn.setText(_translate("MainWindow", "Encode and Save"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.encodeTab), _translate("MainWindow", "Encode"))
        self.label_13.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Step 3 : </span></p></body></html>"))
        self.label_15.setText(_translate("MainWindow", "Progress : "))
        self.label_11.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Step 2 : </span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "Fractional File : "))
        self.chooseFractionalFile.setText(
            _translate("MainWindow", "Choose File"))
        self.label_14.setText(_translate("MainWindow", "Enter Seed :      "))
        self.decodeSeedCheckBox.setText(_translate("MainWindow", "Show Seed"))
        self.decodeSeed.setEchoMode(QtWidgets.QLineEdit.Password)
        self.decodeBtn.setText(_translate("MainWindow", "Decode"))
        self.label_16.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Decoded Data : </span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.decodeTab), _translate("MainWindow", "Decode"))
        self.label_8.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Step 2:</span></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "Choose Image : "))
        self.resultChooseFile.setText(_translate("MainWindow", "Choose File"))
        self.resultPreview.setText(_translate("MainWindow", "Preview"))
        self.resultBtn.setText(_translate("MainWindow", "Compare"))
        self.label_23.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Results :</span></p></body></html>"))
        self.resultLabel_1.setText(_translate(
            "MainWindow", "Peak Signal-to-Noise Ratio (PSNR) : "))
        self.resultLabel_2.setText(_translate(
            "MainWindow", "Structural Similarity Index Metric (SSIM): "))
        self.resultLabel_3.setText(_translate(
            "MainWindow", "Universal Image Quality Index : "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.resultTab), _translate("MainWindow", "Compare"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuResults.setTitle(_translate("MainWindow", "Results"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionComparison.setText(_translate("MainWindow", "Comparison"))
        self.actionVisit_us.setText(_translate("MainWindow", "Visit us"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())