import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

class PCA1:
    def __init__(self, dimensions):
        self.project_onto = dimensions
        self.covariance = 1


    def center_data(self, Xtrain, faces_per_class):
        print("\nPerforming PCA........ ")
        sum1 = np.sum(Xtrain, axis = 1)

        sum1 = np.sum(sum1, axis = 0)

        global_mean = (1/(Xtrain.shape[0]*Xtrain.shape[1])) * sum1
        global_mean = np.reshape(global_mean, (1,global_mean.shape[0]))
        #print("global mean ",global_mean.shape)
        #plt.imshow(np.reshape(global_mean,(24,21)), cmap='gray')
        #plt.show()
        #cv2.imshow("windos", np.reshape(global_mean,(24,21)))
        #cv2.waitKey(0)

        temp_Xtrain = np.reshape(Xtrain, ((Xtrain.shape[0]*Xtrain.shape[1]), Xtrain.shape[2]))
        #print("temp Xtrain ", temp_Xtrain.shape)
        centered_X = temp_Xtrain - global_mean

        return centered_X

    def compute_Covariance(self, centered_X ,decorrelate):
        centered_X = centered_X.T
        #print("from covariance here ", centered_X.shape)
        self.covariance = (1/centered_X.shape[1]) * np.dot(centered_X, centered_X.T)
        self.covariance = self.covariance + (decorrelate*np.eye(self.covariance.shape[0]))
        #print("covariance ",self.covariance.shape)
        return

    def project_data(self, centered_X, faces_per_class, test = "OFF"):
        centered_X = centered_X.T
        #print("Centered Xtrain ", centered_X.shape)
        #U, S, V = np.linalg.svd(self.covariance)
        eigval, U = np.linalg.eig(self.covariance)

        U = U.T
        idxs = np.argsort(eigval)[::-1]
        eigval = eigval[idxs]
        U = U[idxs]

        dimensions = U[0:self.project_onto]
        dimensions = dimensions.T
        dimensions = np.real(dimensions)


        # plt.imshow(np.reshape(dimensions[:,2],(24,21)), cmap='gray')
        # plt.show()
        #print("dimensions ",dimensions.shape)
        #print("before ", centered_X.shape)
        projected_X = np.dot(dimensions.T, centered_X)
        #print("projected_X ",projected_X.shape)

        if test == "ON":
            projected_X = projected_X.T
            #print("New Xtest ",projected_X.shape)
        else:
            projected_X = projected_X.T
            shape1 = int(centered_X.shape[1]/faces_per_class)
            projected_X = np.reshape(projected_X, (shape1,faces_per_class,projected_X.shape[1]))
            #print("New Xtrain ",projected_X.shape)

        return projected_X
