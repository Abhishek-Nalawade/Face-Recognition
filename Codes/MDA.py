import numpy as np
import cv2
import matplotlib.pyplot as plt

class MDA:
    def __init__(self, dimensions):
        self.local_means = 1
        self.global_mean = 1
        self.project_onto = dimensions
        self.between_scatter = 1
        self.within_scatter = 1

    def compute_class_means(self, Xtrain, display = "OFF"):
        print("\nPerforming MDA........ ")
        sum = np.sum(Xtrain, axis = 1)
        no_of_faces_each_class = Xtrain.shape[1]
        self.local_means = (1/no_of_faces_each_class) * sum

        sum1 = np.sum(self.local_means, axis = 0)
        no_classes = Xtrain.shape[0]
        prior = 1/no_classes
        self.global_mean = prior * sum1

        #cv2.imshow("mean face",np.reshape(self.global_mean, (24,21)))
        #cv2.waitKey(0)
        self.global_mean = np.reshape(self.global_mean, (self.global_mean.shape[0],1))
        if(display == "ON"):
            print("local means ",self.local_means.shape)
            print("global mean ", self.global_mean.shape)
        return


    def compute_scatter_matrices(self, Xtrain, decorrelate, display = "OFF"):
        z = np.zeros((1,Xtrain.shape[0]))
        #global_means = self.global_mean + z
        diff = self.local_means.T - self.global_mean
        prior = (1/Xtrain.shape[0])
        self.between_scatter = (prior) * np.dot(diff, diff.T)

        local_means_repeated = np.repeat(self.local_means.T, Xtrain.shape[1], axis = 1)

        temp_Xtrain = np.reshape(Xtrain, ((Xtrain.shape[0]*Xtrain.shape[1]), Xtrain.shape[2]))
        temp_Xtrain = temp_Xtrain.T

        no_of_classes = Xtrain.shape[0]
        no_examples_each_class = Xtrain.shape[1]
        within_diff = temp_Xtrain - local_means_repeated
        self.within_scatter = (1/no_of_classes) * (1/no_examples_each_class) * np.dot(within_diff, within_diff.T)

        if(display == "ON"):
            print("between scatter ",self.between_scatter.shape)
            print("repeated local means ",local_means_repeated.shape)
            print("temp_Xtrain shape ", temp_Xtrain.shape)
            print("within scatter ",self.within_scatter.shape)
        #cv2.imshow("old ",np.reshape(Xtrain[2,1,:], (24,21)))
        #cv2.imshow("new ",np.reshape(temp_Xtrain[:,5], (24,21)))
        #cv2.waitKey(0)

        self.within_scatter = self.within_scatter + (decorrelate * np.eye(self.within_scatter.shape[0]))

        return temp_Xtrain

    def project_data(self, temp_Xtrain, faces_per_class, test = "OFF"):
        within_scatter_inv = np.linalg.inv(self.within_scatter)
        #print("within_scatter_inv ",within_scatter_inv.shape)
        U, S, V = np.linalg.svd(np.dot(within_scatter_inv, self.between_scatter))
        dimensions = U[:,:self.project_onto]
        #print("dimensions ",dimensions.shape)

        # plt.imshow(np.reshape(dimensions[:,1],(24,21)), cmap='gray')
        # plt.show()

        projected_X = np.dot(dimensions.T, temp_Xtrain)
        #print("Projected X ",projected_X.shape)
        if(test == "ON"):
            projected_X = projected_X.T
            #print("New Xtest ",projected_X.shape)
        else:
            projected_X = projected_X.T
            shape1 = int(temp_Xtrain.shape[1]/faces_per_class)

            projected_X = np.reshape(projected_X, (shape1,faces_per_class,projected_X.shape[1]))
            #print("New Xtrain ",projected_X.shape)
        return projected_X
