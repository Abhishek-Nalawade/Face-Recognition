import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

class LoadData:
    def __init__(self):
        self.Xtrain = 1
        self.Ytrain = 1
        self.Xtest = 1
        self.Ytest = 1

    def data_preprocessing(self, x, question, cross_val, cross_val_ind):
        X = x['face']
        #cv2.imshow("first", x['face'][:,:,8])
        #print(X.shape)
        X = np.moveaxis(X,-1,0)
        X = np.reshape(X, (200,3, X.shape[1], X.shape[2]))
        X = np.reshape(X, (200,3, (X.shape[2]*X.shape[3])))
        #print(X.shape)
        if question == 1:       # for face labelling data
            faces_per_class = 2
            self.Xtrain = X[:,1:,:]
            self.Xtest = X[:,0,:]
            #cv2.imshow("before1 ",np.reshape(self.Xtest[10,:], (24,21)))
            #cv2.waitKey(0)

            #cv2.imshow("before ",np.reshape(self.Xtrain[11,0,:], (24,21)))
            #cv2.imshow("after", np.reshape(self.Xtrain[11,1,:], (24,21)))
            #print(self.Xtrain.shape,"   ",self.Xtest.shape)

            # plt.imshow(np.reshape(self.Xtrain[11,0,:],(24,21)), cmap='gray')
            # plt.show()

            Y = np.arange(self.Xtrain.shape[0])
            Y = np.reshape(Y, (1, Y.shape[0]))
            self.Ytrain = Y
            self.Ytest = Y

        if question == 2:       # for facial expression data
            train_samples = int(0.6 * X.shape[0])
            faces_per_class = train_samples
            test_samples = X.shape[0] - train_samples
            if cross_val == "OFF":
                self.Xtrain = X[:train_samples,:2,:]
                a = self.Xtrain.shape
                #print("pre preprocessed ",self.Xtrain.shape)
                #cv2.imshow("before ",np.reshape(self.Xtrain[1,0,:], (24,21)))
                #cv2.imshow("after", np.reshape(self.Xtrain[1,1,:], (24,21)))
                self.Xtrain = np.moveaxis(self.Xtrain, 1, 0)
                # self.Xtrain = np.reshape(self.Xtrain, (self.Xtrain.shape[0]*self.Xtrain.shape[1], self.Xtrain.shape[2]))
                # self.Xtrain = np.reshape(self.Xtrain, (a[1], a[0], self.Xtrain.shape[1]))

                #print("preprocessed ",self.Xtrain.shape)
                #cv2.imshow("before1 ",np.reshape(self.Xtrain[0,1,:], (24,21)))
                #cv2.imshow("after1", np.reshape(self.Xtrain[1,1,:], (24,21)))

                #cv2.waitKey(0)
                self.Ytrain = np.ones((self.Xtrain.shape[0], self.Xtrain.shape[1]))
                self.Ytrain[1,:] = (-1) * self.Ytrain[1,:]

                self.Xtest = X[train_samples:,:2,:]

                #a = self.Xtest.shape

                self.Xtest = np.moveaxis(self.Xtest, 1, 0)

                #self.Xtest = np.reshape(self.Xtest, (a[1], a[0],a[2]))

                self.Ytest = np.ones((self.Xtest.shape[0], self.Xtest.shape[1]))
                self.Ytest[1,:] = (-1) * self.Ytest[1,:]
                # print("Xtrain ", self.Xtrain.shape)
                # print("Ytrain ", self.Ytrain.shape)
            elif cross_val == "ON":
                if cross_val_ind == 1:
                    self.Xtrain = X[:train_samples,:2,:]
                    self.Xtrain = np.moveaxis(self.Xtrain, 1, 0)
                    self.Ytrain = np.ones((self.Xtrain.shape[0], self.Xtrain.shape[1]))
                    self.Ytrain[1,:] = (-1) * self.Ytrain[1,:]
                    self.Xtest = X[train_samples:,:2,:]
                    self.Xtest = np.moveaxis(self.Xtest, 1, 0)
                    self.Ytest = np.ones((self.Xtest.shape[0], self.Xtest.shape[1]))
                    self.Ytest[1,:] = (-1) * self.Ytest[1,:]

                elif cross_val_ind==2:
                    self.Xtrain = X[test_samples:,:2,:]
                    self.Xtrain = np.moveaxis(self.Xtrain, 1, 0)
                    self.Ytrain = np.ones((self.Xtrain.shape[0], self.Xtrain.shape[1]))
                    self.Ytrain[1,:] = (-1) * self.Ytrain[1,:]
                    self.Xtest = X[:test_samples,:2,:]
                    self.Xtest = np.moveaxis(self.Xtest, 1, 0)
                    self.Ytest = np.ones((self.Xtest.shape[0], self.Xtest.shape[1]))
                    self.Ytest[1,:] = (-1) * self.Ytest[1,:]


        #cv2.imshow("window",X[2,2])
        #cv2.waitKey(0)
        #cv2.imshow("before1 ",np.reshape(self.Xtrain[11,0,:], (24,21)))
        ##cv2.imshow("after1", np.reshape(self.Xtrain[11,1,:], (24,21)))
        #cv2.imshow("after12", np.reshape(self.Xtest[11,:], (24,21)))
        #cv2.waitKey(0)
        print("\nLoaded data from data.mat file")
        return faces_per_class

    def pose_processing(self, x, question=1):
        #print(x)
        X = x['pose']
        #plt.imshow(X[:,:,10,1], cmap='gray')
        #plt.show()
        # for i in range(X.shape[2]):
        #     plt.imshow(np.reshape(X[:,:,i,14],(48,40)), cmap='gray')
        #     plt.show()

        X = np.moveaxis(X,-1,0)
        X = np.moveaxis(X,-1,1)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
        # for i in range(X.shape[1]):
        #     plt.imshow(np.reshape(X[14,i,:],(48,40)), cmap='gray')
        #     plt.show()

        if question == 1:
            faces_per_class = round(0.6*X.shape[1])
            self.Xtrain = X[:,:faces_per_class,:]
            self.Xtest = X[:,faces_per_class:,:]
            #print(self.Xtrain.shape,"   ",self.Xtest.shape)

            Y = np.arange(self.Xtrain.shape[0])
            Y = np.reshape(Y, (1, Y.shape[0]))
            Y1 = np.repeat(Y,8)
            Y2 = np.repeat(Y, 5)
            self.Ytrain = Y1
            self.Ytest = Y2
            self.Ytrain = np.reshape(self.Ytrain, (1,self.Ytrain.shape[0]))
            self.Ytest = np.reshape(self.Ytest, (1,self.Ytest.shape[0]))
            #print(self.Ytest)
            #print("faces_per_class ", faces_per_class)
        print("\nLoaded data from pose.mat file")
        return faces_per_class


    def illumination_preprocessing(self, x, question=1):
        #print(x)
        X = x['illum']
        X = np.moveaxis(X,-1,0)
        X = np.moveaxis(X,-1,1)
        #print(X.shape)

        if question == 1:
            faces_per_class = round(0.6*X.shape[1])
            self.Xtrain = X[:,:faces_per_class,:]
            self.Xtest = X[:,faces_per_class:,:]
            #print(self.Xtrain.shape,"   ",self.Xtest.shape)

            Y = np.arange(self.Xtrain.shape[0])
            Y = np.reshape(Y, (1, Y.shape[0]))
            Y1 = np.repeat(Y,13)
            Y2 = np.repeat(Y, 8)
            self.Ytrain = Y1
            self.Ytest = Y2
            self.Ytrain = np.reshape(self.Ytrain, (1,self.Ytrain.shape[0]))
            self.Ytest = np.reshape(self.Ytest, (1,self.Ytest.shape[0]))
            #print(self.Ytrain.shape)
            #print("faces_per_class ", faces_per_class)

        print("\nLoaded data from illumination.mat file")
        return faces_per_class


    def get_data(self, file, question, cross_val="OFF", cross_val_ind=1):
        x = loadmat('../%s.mat' %file)
        #print(x['face'].shape)
        #print(type(x['face'][0]))
        if file == "data":
            faces_per_class = self.data_preprocessing(x, question, cross_val, cross_val_ind)
        elif file == "pose":
            faces_per_class = self.pose_processing(x, question)
        elif file == "illumination":
            faces_per_class = self.illumination_preprocessing(x, question)
        return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, faces_per_class
