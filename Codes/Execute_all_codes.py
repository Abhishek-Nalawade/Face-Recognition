from LoadData import *
from MDA import *
from PCA import *
from BayesClassifier import *
from kNN import *
from SVM import *
from AdaBoost import *


def runMDA(Xtrain, Xtest, faces_per_class, data, question_no, dimensions, method="other"):
    discriminant = MDA(dimensions)
    #print("question is ",question_no)
    if method == "SVM" or question_no == 2:
        Xtest = np.reshape(Xtest, (Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2]))
    discriminant.compute_class_means(Xtrain, display="OFF")
    temp_X = discriminant.compute_scatter_matrices(Xtrain, decorrelate=0.01, display="OFF")
    Xtrain = discriminant.project_data(temp_X, faces_per_class)


    if data == "data":
        Xtest = discriminant.project_data(Xtest.T,faces_per_class, test="ON")
    elif data == "pose" or data == "illumination":
        a = Xtest.shape
        Xtest = np.reshape(Xtest, (a[0]*a[1], a[2]))
        Xtest = discriminant.project_data(Xtest.T,faces_per_class, test="ON")

    #cv2.imshow("windoe",np.reshape(Xtrain[0,0,:], ()))
    #cv2.waitKey(0)
    if method == "SVM" or question_no == 2:
        Xtest = np.reshape(Xtest, (2,int(Xtest.shape[0]/2),Xtest.shape[1]))
    return Xtrain, Xtest

def runPCA(Xtrain, Xtest, faces_per_class, data, question_no, dimensions, method="other"):
    representer = PCA1(dimensions)
    if method == "SVM" or question_no == 2:
        Xtest = np.reshape(Xtest, (Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2]))
    centered_X = representer.center_data(Xtrain, faces_per_class)
    representer.compute_Covariance(centered_X, decorrelate = 0.2)  #0.2
    Xtrain = representer.project_data(centered_X, faces_per_class)

    print("Xtest from PCA ",Xtest.shape)
    if data == "data":
        Xtest = representer.project_data(Xtest, faces_per_class, test = "ON")
    elif data == "pose" or data == "illumination":
        a = Xtest.shape
        Xtest = np.reshape(Xtest, (a[0]*a[1], a[2]))
        Xtest = representer.project_data(Xtest,faces_per_class, test="ON")

    if method == "SVM" or question_no == 2:
        Xtest = np.reshape(Xtest, (2,int(Xtest.shape[0]/2),Xtest.shape[1]))
    return Xtrain, Xtest


def runBayesClassifier(file, question_no, dimensions=150, data_reduction="OFF"):
    data = LoadData()
    Xtrain, Ytrain, Xtest, Ytest, faces_per_class = data.get_data(file, question=question_no)

    if data_reduction == "MDA":
        Xtrain, Xtest = runMDA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = dimensions)
    elif data_reduction == "PCA":
        Xtrain, Xtest = runPCA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 270)

    #print(Xtrain)
    if question_no == 2 and data_reduction == "PCA":
        decorrelate = 0.025
    else:
        decorrelate = 0.9
    classify = BayesClassifier(file)
    mean = classify.compute_ML_mean(Xtrain)
    covariance = classify.compute_ML_covariance(Xtrain, mean, faces_per_class, decorrelate) #0.9
    classified_labels = classify.classify_data(mean, covariance, Xtest, Xtrain, question_no, file)
    classify.compute_accuracy(classified_labels, Ytest, file, question_no)
    return

def run_kNN(file, question_no, data_reduction="OFF"):
    data = LoadData()
    Xtrain, Ytrain, Xtest, Ytest, faces_per_class = data.get_data(file, question=question_no)
    #print("hi ", Xtest.shape)
    print("\nInitializing K-NN.........")
    if data_reduction == "MDA":
        Xtrain, Xtest = runMDA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 150)
    elif data_reduction == "PCA":
        Xtrain, Xtest = runPCA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 270)

    K = [1,3,5,7,9,11]
    for k in K:
        classify = kNN(k)
        predicted = classify.find_nearestNeighbors(Xtrain, Ytrain, Xtest, Ytest, faces_per_class, question_no, file, data_reduction)
        classify.compute_accuracy(predicted, Ytest, faces_per_class, file, question_no)
    return

def run_LinearSVM(file, question_no, data_reduction="OFF"):
    data = LoadData()
    Xtrain, Ytrain, Xtest, Ytest, faces_per_class = data.get_data(file, question=question_no)

    if data_reduction == "MDA":
        Xtrain, Xtest = runMDA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 150, method="SVM")
    elif data_reduction == "PCA":
        Xtrain, Xtest = runPCA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 270, method="SVM")


    iterations = 1000
    regularization = [0.001, 0.002, 0.03, 0.04, 0.05, 0.06, 0.7, 0.8, 1]
    accuracies = list()
    for i in range(len(regularization)):
        classifier = SVM(regularization[i], iterations, step_size=0.001)

        W, b = classifier.trainLinear(Xtrain, Ytrain)
        prediction, temp_Ytest = classifier.assign_labels(W, b, Xtest, Ytest)
        accuracies.append(classifier.compute_accuracy(prediction, temp_Ytest))

    regularization = np.array(regularization)
    accuracies = np.array(accuracies)

    plt.plot(regularization, accuracies, 'bo')
    plt.plot(regularization, accuracies)
    plt.show()
    return


def run_KernelSVM(file, question_no, kernel_type, data_reduction="OFF"):
    data = LoadData()
    Xtrain, Ytrain, Xtest, Ytest, faces_per_class = data.get_data(file, question=question_no, cross_val="ON", cross_val_ind=2)

    if data_reduction == "MDA":
        Xtrain, Xtest = runMDA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 200, method="SVM")
    elif data_reduction == "PCA":
        Xtrain, Xtest = runPCA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 270, method="SVM")



    regularization = [0.001, 0.002, 0.03, 0.04, 0.05, 0.06, 0.7, 0.8, 20]
    accuracies = list()
    sigma = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("\nInitializing Kernel SVM.........")
    for i in range(len(sigma)):
        classifier = SVM(regularization[8], kernel_type)
        if kernel_type == "RBF":
            W,b = classifier.trainKernel(Xtrain, Ytrain, sigma[i])
        elif kernel_type == "polynomial":
            W,b = classifier.trainKernel(Xtrain, Ytrain, r[i])
        prediction, temp_Ytest = classifier.assign_labels(W, b, Xtest, Ytest)
        accuracies.append(classifier.compute_accuracy(prediction, temp_Ytest))

    if kernel_type == "polynomial":
        sigma = r

    sigma = np.array(sigma)
    accuracies = np.array(accuracies)

    plt.plot(sigma, accuracies, 'bo')
    plt.plot(sigma, accuracies)
    plt.show()

    return

def run_AdaBoost(file, question_no, data_reduction="OFF"):
    data = LoadData()
    Xtrain, Ytrain, Xtest, Ytest, faces_per_class = data.get_data(file, question=question_no)
    #print("hi ", Xtest.shape)
    if data_reduction == "MDA":
        Xtrain, Xtest = runMDA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 150)
    elif data_reduction == "PCA":
        Xtrain, Xtest = runPCA(Xtrain, Xtest, faces_per_class, file, question_no, dimensions = 270)

    iterations = 10
    accuracies = list()
    regularization = [0.04, 0.05, 0.06, 0.7, 0.8, 20]
    #for i in range(len(regularization)):
    booster = AdaBoost(iterations, regularization[0])
    weights = booster.initialize(Xtrain)
    best_classifier = booster.boost(Xtrain, Ytrain, Xtest, Ytest, weights)
    accuracies.append(booster.compute_final_accuracy(best_classifier, Xtest, Ytest))

    # regularization = np.array(regularization)
    # accuracies = np.array(accuracies)
    #
    # plt.plot(regularization, accuracies, 'bo')
    # plt.plot(regularization, accuracies)
    # plt.show()
    return

print("The following modules are available: \n1) Bayes Classifier \n2) kNN Classifier")
print("3) Linear SVM \n4) Kernel SVM \n5) Boosted SVM")
request = int(input("Please enter the number corresponding to the classifier to run it: "))
print("\nPlease select the dataset to test on \n1) Data \n2) Pose \n3) Illumination")
dataset = int(input("Enter the nmuber corresponding to the dataset: "))
dataset = dataset - 1
dataset_names = ["data", "pose", "illumination"]
print("\nPlease enter the type for data reduction \n0) Without data reduction \n1) MDA \n2) PCA")
data_red = int(input("Please enter the number corresponding to the data reduction method: "))
reduction_methods = ["OFF", "MDA", "PCA"]

if request == 1:
    if dataset_names[dataset] != "data":
        question = 1
        dimensions = 150
    else:
        print("\nFollowing classification is available: \n1) Identifying subject label")
        print("2) Neutral v/s Facial expression ")
        question = int(input("Enter the number corresponding to the classification "))
        if question == 2:
            dimensions = 1
        else:
            dimensions = 150
    runBayesClassifier(dataset_names[dataset], question, dimensions, data_reduction=reduction_methods[data_red])
elif request == 2:
    if dataset_names[dataset] != "data":
        question = 1
    else:
        print("\nFollowing classification is available: \n1) Identifying subject label")
        print("2) Neutral v/s Facial expression ")
        question = int(input("Enter the number corresponding to the classification "))
    run_kNN(dataset_names[dataset], question, data_reduction=reduction_methods[data_red])
elif request == 3:
    run_LinearSVM("data", 2, data_reduction=reduction_methods[data_red])
elif request == 4:
    print("\nFollowing kernels are available: \n1) Radial Basis Function Kernel ")
    print("2) Polynomial Kernel")
    inp = int(input("Enter the number corresponding to the Kernel "))
    inp = inp - 1
    kernels = ["RBF", "polynomial"]
    run_KernelSVM("data", 2, kernels[inp], data_reduction=reduction_methods[data_red])
elif request == 5:
    run_AdaBoost("data", 2, data_reduction=reduction_methods[data_red])




#runBayesClassifier("data", 1, data_reduction="MDA") #question_no #works for MDA

#############running on top right terminal
#runBayesClassifier("pose", 1, data_reduction="MDA")              #works for MDA

##########runnig on top left terminal
#runBayesClassifier("illumination",1, data_reduction="MDA")       #works for PCA MDA

#works for MDA PCA =75.625% with decorrelate=0.025 and for OFF decorrelate=0.9
# runBayesClassifier("data", 2, data_reduction="MDA")              #works for PCA MDA


#run_kNN("data", 1, data_reduction="MDA")                         #works for MDA
#run_kNN("pose",1, data_reduction="MDA")                          #works for MDA
#run_kNN("illumination",1, data_reduction="MDA")                  #works for MDA
#run_kNN("data",2, data_reduction="MDA")                          #works for PCA MDA
#run_LinearSVM("data", 2, data_reduction="OFF")                   #works for PCA MDA
#run_KernelSVM("data", 2, "polynomial", data_reduction="MDA")     #works for PCA MDA
#run_KernelSVM("data", 2, "RBF", data_reduction="MDA")            #works for PCA MDA
#run_AdaBoost("data", 2, data_reduction="MDA")                     #done
