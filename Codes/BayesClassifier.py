import numpy as np
import cv2

class BayesClassifier:
    def __init__(self, name):
        self.fileData = name

    def compute_ML_mean(self, Xtrain):
        print("\nInitializing Bayes Classifier.........")
        sum1 = np.sum(Xtrain, axis = 1)
        no_of_faces_each_class = Xtrain.shape[1]
        mean = (1/no_of_faces_each_class) * sum1
        #print("mean ", mean.shape)
        #cv2.imshow("window",np.reshape(mean[10], (24,21)))
        #cv2.waitKey(0)
        return mean

    def compute_ML_covariance(self, Xtrain, mean, no_of_faces_each_class, decorrelate):
        covariances = list()
        disturbance = decorrelate * np.eye(Xtrain.shape[2])
        for i in range(Xtrain.shape[0]):
            for j in range(no_of_faces_each_class):
                Xi1 = Xtrain[i,j,:]
                #print("now ",Xi1.shape)
                Xi1 = np.reshape(Xi1, (Xi1.shape[0], 1))
                if j == 0:
                    Xi = Xi1
                else:
                    Xi = np.concatenate((Xi,Xi1), axis = 1)

            mean1 = np.reshape(mean[i,:], (mean.shape[1],1))
            diff = Xi - mean1
            covariance = np.dot(diff, diff.T)
            covariance = (covariance/no_of_faces_each_class) + disturbance
            covariances.append(covariance)
        covariances = np.array(covariances)
        #print("covariances ",covariances.shape)
        return covariances

    def classify_data(self, mean, covariance, Xtest, Xtrain, question_no, data):
        #print("Xtest shape ", Xtest.shape)
        if question_no == 2 or data == "pose" or data == "illumination":
            a = Xtest.shape
            if len(a) == 3:
                Xtest = np.reshape(Xtest, (a[0]*a[1],a[2]))
        test_labels = list()
        prior = 1/Xtrain.shape[0]
        for i in range(Xtest.shape[0]): #each test case
            classes = list()
            for j in range(mean.shape[0]):         #compared with each class
                mean1 = np.reshape(mean[j,:], (mean.shape[1],1))
                Xt = np.reshape(Xtest[i,:] , (Xtest.shape[1],1))
                #print("here ",mean1)
                diff = Xt - mean1
                cov_inv = np.linalg.inv(covariance[j,:,:])
                covdet = np.linalg.det(covariance[j,:,:])
                covdet = covdet**(1/2)

                #likelihood = (1/((2*np.pi)**(Xtest.shape[1]/2))*covdet) * np.exp((-(1/2))*(np.dot(diff.T, np.dot(cov_inv, diff))))
                likelihood = -((Xtest.shape[1]/2)*np.log(2*np.pi)) - (np.log(covdet)/2) - ((1/2)*(np.dot(diff.T, np.dot(cov_inv, diff))))
                posterior = likelihood + np.log(prior)

                classes.append(posterior[0][0])
            classes = np.array(classes)
            #print(classes)
            test_labels.append(np.argmax(classes))
            #print(classes)
            #print("class ", np.argmax(classes))
            print("Going through example: ",i)
        print("done")
        return test_labels

    def compute_accuracy(self, test_labels, Ytest, data, question_no):
        score = 0

        if question_no == 2:
            a = Ytest.shape
            Ytest = np.reshape(Ytest, (a[0]*a[1],1))
            Ytest = Ytest.T
            #print(Ytest.shape)
            Ytest[Ytest[:,:]==1] = 2
            Ytest[Ytest[:,:]==-1] = 1
            Ytest[Ytest[:,:]==2] = 0


        for i in range(len(test_labels)):
            if data == "data" and question_no == 1:
                if i==test_labels[i]:
                    score += 1
            elif data == "pose" or data == "illumination" or question_no == 2:
                print("Labels and prediction: ",Ytest[0,i],"  ==  ",test_labels[i])
                if Ytest[0,i]==test_labels[i]:
                    score += 1
        accuracy = (score * 100)/len(test_labels)
        print("Accuracy is: ",accuracy)
        return
