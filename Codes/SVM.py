import numpy as np
import cv2
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, regularization, kernel_type, iterations = 1000, step_size = 0.001):
        self.step_size = step_size
        self.regularization = regularization
        self.no_of_iterations = iterations
        self.kernel_type = kernel_type


    def radialBasisFunction(self, X, Y, sigma):
        #print(sigma)
        dist = np.linalg.norm(X-Y, axis=1)
        rbf = (dist**2)/(sigma**2)
        rbf = np.exp(-rbf)
        #print(rbf.shape)
        return rbf

    def polynomialKernel(self, X, Y, d):
        #print("FROM HERRE ", X.shape,"    ",Y.shape)
        Y = Y.T
        col = (np.dot(X,Y) + 1)**d
        return col

    def trainKernel(self, Xtrain, Ytrain, sigma):
        # print("hiiiiiiiii ",Xtrain.shape)
        # print("hiiiiiiiii ",Ytrain.shape)
        a = Xtrain.shape
        temp_Xtrain = np.reshape(Xtrain, (a[0]*a[1],a[2]))
        b = Ytrain.shape
        temp_Ytrain = np.reshape(Ytrain, (b[0]*b[1],1))
        #print(temp_Ytrain)
        K = np.zeros((a[0]*a[1], a[0]*a[1]))
        #print(K.shape)
        for i in range(temp_Xtrain.shape[0]):
            Xt = temp_Xtrain[i,:]
            Xt = np.reshape(Xt, (1,Xt.shape[0]))
            if self.kernel_type == "RBF":
                K[i,:] = self.radialBasisFunction(Xt, temp_Xtrain, sigma)
            elif self.kernel_type == "polynomial":
                K[i,:] = self.polynomialKernel(Xt, temp_Xtrain, sigma)

        #print(K)
        #print("from SVM ",temp_Ytrain.shape)
        Y1 = np.dot(temp_Ytrain, temp_Ytrain.T)
        #print(Y1.shape)
        P = matrix(Y1 * K)
        q = matrix(-np.ones((temp_Xtrain.shape[0])))
        G = matrix(np.vstack((np.eye(temp_Xtrain.shape[0]) * -1, np.eye(temp_Xtrain.shape[0]))))
        h = matrix(np.hstack((np.zeros(temp_Xtrain.shape[0]), np.ones(temp_Xtrain.shape[0]) * self.regularization)))
        A = matrix(temp_Ytrain, (1, temp_Xtrain.shape[0]), "d")
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol["x"])

        #print(alphas)

        threshold = -0.00001
        # idx = np.arange(temp_Xtrain.shape[0])[np.ravel(alphas>threshold)]
        # alphas = alphas[idx]
        #print(idx)
        support = ((alphas>threshold)).flatten()
        Y1 = alphas[support]*temp_Ytrain[support]
        #print(Y1)
        W = np.dot(Y1.T,temp_Xtrain[support])

        b = (1/Y1[0]) - (np.dot(W,temp_Xtrain[support][0]))
        #print(b)

        W = W.T
        return W, b


    def trainLinear(self, Xtrain, Ytrain):
        #print("Xtrain ", Xtrain.shape)
        #print("Ytrain ", Ytrain.shape)

        W = np.zeros((Xtrain.shape[2],1))
        b = 0

        for i in range(self.no_of_iterations):
            for j in range(Xtrain.shape[1]):
                #positive example
                X = Xtrain[0,j]
                X = np.reshape(X, (X.shape[0],1))
                Y1 = Ytrain[0,j]
                condition = Y1 * (np.dot(W.T, X) - b)
                if condition >= 1:
                    W = W - (self.step_size * 2 * self.regularization * W)
                else:
                    W = W - (self.step_size * ((2 * self.regularization * W)-(np.dot(X, Y1))))
                    b = b - (self.step_size * Y1)

                #negative example
                X = Xtrain[1,j]
                X = np.reshape(X, (X.shape[0],1))
                Y1 = Ytrain[1,j]
                #print(Y1)
                condition = Y1 * (np.dot(W.T, X) - b)
                if condition >= 1:
                    W = W - (self.step_size * 2 * self.regularization * W)
                else:
                    W = W - (self.step_size * ((2 * self.regularization * W)-(np.dot(X, Y1))))
                    b = b - (self.step_size * Y1)

        #print("for this ",W.shape)
        return W, b

    def assign_labels(self, W, b, Xtest, Ytest):
        temp_Xtest = np.reshape(Xtest, ((Xtest.shape[0]*Xtest.shape[1]), Xtest.shape[2]))
        temp_Ytest = np.reshape(Ytest, (Ytest.shape[0]*Ytest.shape[1]))

        #print(temp_Ytest.shape)

        output = np.dot(temp_Xtest, W)
        #print(output)
        prediction = np.sign(output)

        return prediction, temp_Ytest

    def compute_accuracy(self, prediction, temp_Ytest):
        score = 0
        #print(temp_Ytest)
        for i in range(prediction.shape[0]):
            #print(prediction[i][0],"    ",temp_Ytest[i])
            if prediction[i][0] == temp_Ytest[i]:
                score += 1
        accuracy = (score * 100)/prediction.shape[0]
        print("\nAccuracy is: ",accuracy)
        return accuracy
