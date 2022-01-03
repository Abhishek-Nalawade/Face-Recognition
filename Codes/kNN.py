import numpy as np
import cv2
import matplotlib.pyplot as plt

class kNN:
    def __init__(self, no_of_NN):
        self.no_of_NN = no_of_NN

    def assign_labels(self, test_case, faces_per_class):
        #print("faces ",faces_per_class)
        #print("before ",test_case)
        test_case = np.array(test_case)
        test_case = test_case/faces_per_class
        test_case = test_case.astype(np.int)
        #print(test_case)

        vote = list()
        for i in range(self.no_of_NN):
            poss = np.where(test_case == test_case[i])
            #print(poss[0])
            vote.append(len(poss[0]))
        #print(vote)
        max_val = max(vote)
        #print("here ",max_val)
        if max_val > 1 or self.no_of_NN == 1:
            ind = vote.index(max_val)
            pred_class = test_case[ind]
        else:
            pred_class = None
        #print(ind)
        return pred_class

    def euclidean_distance(self, temp_Xtrain, Xtest, faces_per_class, question_no, data, data_reduction):
        prediction = list()
        if (question_no == 2) or  (question_no == 2 and data_reduction == "OFF") or (data=="pose" and data_reduction=="OFF") or (data == "illumination" and data_reduction=="OFF"):
            a = Xtest.shape
            Xtest = np.reshape(Xtest, (a[0]*a[1],a[2]))
        for i in range(Xtest.shape[0]):
            #plt.imshow(np.reshape(Xtest[i,:], (24,21)), cmap='gray')
            #plt.show()
            #cv2.imshow("window ",np.reshape(Xtest[i,:], (24,21)))
            #cv2.waitKey(0)
            test = np.reshape(Xtest[i], (1,Xtest.shape[1]))
            diff = temp_Xtrain - test
            diff_sq = diff**2
            euclidean_dist = (np.sum(diff_sq, axis = 1))**(1/2)
            #euclidean_dist = np.reshape(euclidean_dist, (euclidean_dist.shape[0],1))
            sorted_euclidean_dist = np.sort(euclidean_dist)

            min = sorted_euclidean_dist[:self.no_of_NN]

            #print(min)
            test_case = list()
            for j in range(self.no_of_NN):
                test_case.append(np.where(euclidean_dist == min[j])[0][0])
            #print("test ",test_case)

            pred = self.assign_labels(test_case, faces_per_class)
            prediction.append(pred)
            #print(prediction)
            # if i==10:
            #     break
            #print("prediction ",pred)

        #print(len(prediction))
        #print(prediction)
        return prediction


    def find_nearestNeighbors(self, Xtrain, Ytrain, Xtest, Ytest, faces_per_class, question_no, data, data_reduction):

        #cv2.imshow("before", np.reshape(Xtrain[4,0,:],(24,21)))
        temp_Xtrain = np.reshape(Xtrain, ((Xtrain.shape[0]*Xtrain.shape[1]), Xtrain.shape[2]))
        #print("temp Xtrain ", temp_Xtrain.shape)
        #cv2.imshow("window", np.reshape(temp_Xtrain[8,:],(24,21)))
        #cv2.waitKey(0)

        prediction = self.euclidean_distance(temp_Xtrain, Xtest, faces_per_class, question_no, data, data_reduction)

        return prediction

    def compute_accuracy(self, test_labels, Ytest, faces_per_class, data, question_no):
        score = 0

        if question_no == 2:
            a = Ytest.shape
            Ytest = np.reshape(Ytest, (a[0]*a[1],1))
            Ytest = Ytest.T
            Ytest[Ytest[:,:]==1] = 2
            Ytest[Ytest[:,:]==-1] = 1
            Ytest[Ytest[:,:]==2] = 0


        if data == "data" and question_no == 1:
            for i in range(len(test_labels)):
                if(i == test_labels[i]):
                    score += 1
        elif data == "pose" or data == "illumination" or question_no == 2:
            for i in range(len(test_labels)):
                #print("after after all ",Ytest[0,i],"     ",test_labels[i])
                if Ytest[0,i] == test_labels[i]:
                    score += 1
        accuracy = (score * 100)/len(test_labels)
        print("\nAccuracy is for %s NN is: "%self.no_of_NN,accuracy)
        return
