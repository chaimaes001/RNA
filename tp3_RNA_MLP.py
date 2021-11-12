# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 18:06:20 2021
@author: Chaimae
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog


cl=MLPClassifier(activation="relu",solver="sgd",learning_rate_init=0.01,
                 n_iter_no_change=3)
#activation="relu", the default value
#,learning_rate="adaptive", default value=constant
#verbose=True,
#max_iter=(default=200) is enough for the learning
#tol=1e-3(default=1e-4, seen with verbose=true) the learning is done but the c.loss is bigger
#early_stopping=True:learning never done even with validation_fraction=0.9(max value)(default=0.1)
#n_iter_no_change=(default=10), learning done with 3 too =D

test=["A","A","B","B","C","C","D","D","E","E","F","F","G","H","H","I","I","J","J","K","L","M","M","N",
      "N","O","O","P","P","Q","R","S","S","T","U","U","V","W","X","Y","Z"]

da=[1,1,1,1,1,2,2,2,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,9,9,10,10,11,11,12,12,13,13,14,14,15,15,15,  #O
    16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26]
data=[]
def alphabet_learning():    
    for i in range(1,63): 
        image=plt.imread("apprentissage/"+str(i)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
        img=np.array(image)                
        img=img.flatten()                
        data.append(img)        
    return data   

d=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U",
    "V","W","X","Y","Z"]            
X=alphabet_learning()
#here learning step
cl.fit(X,da)

#calcul du taux d'apprentissage
cpt=0
for i in range(1,63): 
    image=plt.imread("apprentissage/"+str(i)+".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
    img=np.array(image)                
    r=int(cl.predict([img.flatten()]))       
    if(r==da[i-1]):
        cpt=cpt+1
print("le nombre des images d'apprentissage reconnues : ",cpt)
print("Le taux de reconnaissance des images d'apprentissage est : ", (cpt/62))

#the learning loss
print("La perte calculée durant l'apprentissage des alphabets : ",cl.loss_)

# E=cl.loss_curve_
# plt.plot(E)
# plt.ylabel("Erreur quadratique moyenne ")
# plt.xlabel("Nombre d'itération")
# plt.title(" alphabets ")
# plt.show()

#here le prétraitement des images test, et le calcul du taux de reconnaissance des images test 
def alphabet_test():
    cpt=0
    for i in range(1,42): 
        image=plt.imread("TEST/"+str(i)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
        img=np.array(image)                
        r=int(cl.predict([img.flatten()]))       
        if(d[r-1]==test[i-1]):
            cpt=cpt+1
    print("le nombre des images de test reconnues : ",cpt)
    print("Le taux de reconnaissance des images de test est : ", cpt/41)
    #return cpt

#to verify that the learning is done
# pred=cl.predict(da)
# print(pred)

alphabet_test()                               

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Reconn")
        MainWindow.resize(389, 250)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 20, 241, 20))
        self.label.setObjectName("label")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 50, 101, 23))
        self.pushButton.setObjectName("pushButton")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(150, 80, 71, 51))
               
        # self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_2.setGeometry(QtCore.QRect(150, 140, 75, 23))
        # self.pushButton_2.setObjectName("pushButton_2")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(56, 170, 271, 21))
        self.label_3.setObjectName("label_3")
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Reconnaissance des alphabets "))
        self.label.setText(_translate("MainWindow", "Le classifieur neuronal perceptron multicouche"))
        self.pushButton.setText(_translate("MainWindow", "Choisir une image"))
        #self.pushButton_2.setText(_translate("MainWindow", "Tester"))
        
        self.pushButton.clicked.connect(self.openPicture)
        #self.pushButton_2.clicked.connect(self.tester)

        
    def openPicture(self):
        nom_image = QFileDialog.getOpenFileName()
        self.path = nom_image[0]
        path1 = self.path
        picture = QtGui.QPixmap(path1)
        picture1 = picture.scaled(50, 50, QtCore.Qt.KeepAspectRatio)

        self.label_2.setPixmap(QtGui.QPixmap(picture1))
        self.label_2.adjustSize()
        
        image = plt.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
        img=np.array(image)                
        r=int(cl.predict([img.flatten()]))
        self.label_3.setText("C'est la lettre : "+str(d[r-1]))
        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
