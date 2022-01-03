import numpy as np
import cv2
import matplotlib.pyplot as plt
from SVM import *

class AdaBoost:
    def __init__(self, iterations, regularization):
        self.iterations = iterations
        self.combined_classifiers = list()
        self.ai = list()
        self.base_classifier = SVM(regularization, kernel_type="gradient_descent", step_size=0.001)

    def initialize(self, Xtrain):
        print("\nInitializing Boosted SVM.........")
        a = Xtrain.shape
        weights = (1/(a[0]*a[1])) * np.ones((a[0]*a[1],1))

        return weights

    def evaluate_combined_models(self, Xtest, Ytest):
        final_prediction = 0
        best_accuracy = 0
        x = list()
        accuracies = list()
        temp_classifier = list()
        for i in range(len(self.ai)):
            W = self.combined_classifiers[i][0]
            b = self.combined_classifiers[i][1]
            prediction, temp_Ytest = self.base_classifier.assign_labels(W, b, Xtest, Ytest)
            final_prediction = final_prediction + (prediction * self.ai[i])
            pred_labels = np.sign(final_prediction)
            accuracy = self.base_classifier.compute_accuracy(pred_labels, temp_Ytest)
            temp_classifier.append([W,b])
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_classifier = temp_classifier
            # accuracies.append(accuracy)
            # x.append(i)
        # accuracies = np.array(accuracies)
        # x = np.array(x)
        # plt.plot(x, accuracies, 'bo')
        # plt.plot(x, accuracies)
        # plt.show()
        return best_classifier



    def boost(self, Xtrain, Ytrain, Xtest, Ytest, weights):
        #base_classifier = SVM(regularization[0], kernel_type="gradient_descent", step_size=0.001)
        for i in range(self.iterations):
            if i == 0:
                X = Xtrain
                Y = Ytrain
            else:
                no_samples = Xtrain.shape[0]*Xtrain.shape[1]
                w_temp = np.ravel(weights)
                #print(w_temp.shape)
                idx = np.random.choice(no_samples, no_samples, p=w_temp)

                t = Xtrain.shape

                temp_Xtrain = np.reshape(Xtrain,(t[0]*t[1],t[2]))
                temp_Ytrain = np.reshape(Ytrain, (t[0]*t[1],1))
                X = temp_Xtrain[idx,:]
                Y = temp_Ytrain[idx]
                X = np.reshape(X, (t[0],t[1],t[2]))
                Y = np.reshape(Y, (t[0],t[1]))
                #print(X.shape,"    ",Y.shape)


            W, b = self.base_classifier.trainLinear(X, Y)

            prediction, temp_Y = self.base_classifier.assign_labels(W, b, X, Y)
            #print(prediction)
            #self.base_classifier.compute_accuracy(prediction, temp_Ytest)
            temp_Y = np.reshape(temp_Y, (temp_Y.shape[0],1))

            wrong_pred_idx = prediction != temp_Y               #returns all boolean values at corresponding indices according to the condition

            incorrect = weights[wrong_pred_idx]
            total_error = np.sum(incorrect)

            if total_error == 0:
                total_error = 0.00001
            performance = (1/2) * np.log((1-total_error)/total_error)

            #saving classifiers
            self.ai.append(performance)
            self.combined_classifiers.append([W,b])



            #updating weights
            #print(weights)
            #print("------------------------------------------")
            weights = np.exp(-(temp_Y * prediction * performance))

            weights = (1/np.sum(weights))*weights

            best_classifier = self.evaluate_combined_models(Xtest, Ytest)
        return best_classifier

    def compute_final_accuracy(self, best_classifier, Xtest, Ytest):
        final_prediction = 0
        for i in range(len(best_classifier)):
            W = best_classifier[i][0]
            b = best_classifier[i][1]
            prediction, temp_Ytest = self.base_classifier.assign_labels(W, b, Xtest, Ytest)
            final_prediction = final_prediction + (prediction * self.ai[i])
        pred_labels = np.sign(final_prediction)
        print("\nFinal best: ")
        accuracy = self.base_classifier.compute_accuracy(pred_labels, temp_Ytest)
        return accuracy
