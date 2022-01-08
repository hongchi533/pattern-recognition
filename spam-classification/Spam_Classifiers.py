import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
import math
import scipy.io # package for reading data in Matlab's .mat file format

# Data loading
spam_data = scipy.io.loadmat('spamData.mat')
X_TRAIN = np.array(spam_data["Xtrain"])
Y_TRAIN = np.array(spam_data["ytrain"])
N_train = Y_TRAIN.shape[0]
X_TEST = np.array(spam_data["Xtest"])
Y_TEST = np.array(spam_data["ytest"])
N_test = Y_TEST.shape[0]
print(" X_TRAIN_SIZE: \n",X_TRAIN.shape)
print(" Y_TRAIN_SIZE: \n",Y_TRAIN.shape)
print(" X_TEST_SIZE: \n",X_TEST.shape)
print(" Y_TEST_SIZE: \n",Y_TEST.shape)

#print(X_TRAIN.dtype)
#print(Y_TRAIN.dtype)

# Data Processing
# if a feature is greater than 0, it’s simply set to 1, if it’s less than or equal to 0, it’s set to 0.
X_TRAIN_BNY = np.where(X_TRAIN>0,1,0)
X_TEST_BNY = np.where(X_TEST>0,1,0)

# transform each feature using log(xij + 0.1)
X_TRAIN_LOG = np.log(X_TRAIN + 0.1)
X_TEST_LOG = np.log(X_TEST + 0.1)

#################Q1. Beta Naive Bayes#################
print(">>>>>>>>>>>>Beta Naive Bayes<<<<<<<<<<<<")
# produce Beta(a,a) parameters {0,0.5,1,1.5,....,100}
beta_A = np.linspace(0,100,num=201)
beta_B = np.linspace(0,100,num=201)

print("Beta Hyperparameters:")
print(beta_A)

# Estimate Class Prior using ML
# Number of Class1 / # Number of samples
class_1_prior_ML = (Y_TRAIN.sum()) / (Y_TRAIN.shape[0])

C1_Prior_ML = math.log(class_1_prior_ML)  # for numerical stability, computing log
C0_Prior_ML = math.log(1 - class_1_prior_ML)  # Class0 prior = 1-Class1 prior, bernoulli


def train(beta_1,beta_2):

    # assume a prior Beta(α,α) on the feature distribution
    # calculate posterior predictive distribution for every class and each feature

    # for class 1
    # for each sample, if corresponding ylabel is 0, its values of features become 0
    # then, remove samples belonging to class0
    x_train_class1 = X_TRAIN_BNY*Y_TRAIN
    x_train_class1 = x_train_class1[~np.all(x_train_class1 == 0, axis=1)]

    # calculate posterior predictive distribution=(N1+a)/(N+a+b) of each feature
    # class1_1 = D x 1 (D: number of features)
    class1_1 = (np.sum(x_train_class1, axis=0) + beta_1) / (
                    x_train_class1.shape[0] + beta_1 + beta_2)
    # log computing
    global C1_1 , C1_0
    C1_1 = np.log(class1_1)
    C1_0 = np.log(1-class1_1+1e-20) # for feature = 1 in class 0

    # for class 0
    x_train_class0 = X_TRAIN_BNY*np.logical_not(Y_TRAIN).astype(int)
    # remove samples belonging to class1
    x_train_class0 = x_train_class0[~np.all(x_train_class0 == 0, axis=1)]

    # similarly, for each feature, calculate posterior predictive distribution
    class0_1 = (np.sum(x_train_class0, axis=0) + beta_1) / (
                    x_train_class0.shape[0] + beta_1 + beta_2)

    # log computing
    global C0_1 , C0_0
    C0_1 = np.log(class0_1)
    C0_0 = np.log(1-class0_1+1e-20)  # for feature = 0 in class 0


def predict_beta(X_Samples):
    # predict target class label of test data x using MAP
    # calculate and store posterior log(p(y | x, parameters)) of each class for each sample

    # the class prior plus the sum of the likelihoods of all features of the test sample
    # beta_bayes_c1 = N x 1 (N : number of samples)
    beta_bayes_c1 = C1_Prior_ML + X_Samples.dot(C1_1) + np.logical_not(X_Samples).astype(int).dot(C1_0)
    beta_bayes_c0 = C0_Prior_ML + X_Samples.dot(C0_1) + np.logical_not(X_Samples).astype(int).dot(C0_0)

    #compare log(p(y = 1 | x, parameters)) and  log(p(y = 0 | x, parameters))
    beta_bayes_prob = np.vstack((beta_bayes_c0, beta_bayes_c1))
    return np.argmax(beta_bayes_prob, axis=0)   # x belong to class c which posterior is bigger

# function of calculating error rate using mean method
def error_rate (targets , predictions):
    absolute_differences = np.absolute(predictions - targets)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences*100

# iterating by beta Hyperparameter
ERROR_TRAIN = []
ERROR_TEST = []
i=0     #idx of beta Hyperparameter list

for beta_a, beta_b in zip(beta_A, beta_B):

    # train
    train(beta_a, beta_b)

    # predict training dataset
    predict_train = (predict_beta(X_TRAIN_BNY))

    # calculate and store the error rate corresponding to each beta parameter
    ERROR_TRAIN.append(error_rate(predict_beta(X_TRAIN_BNY), Y_TRAIN.reshape(Y_TRAIN.shape[0])))

    # predict Testing dateset
    ERROR_TEST.append(error_rate(predict_beta(X_TEST_BNY), Y_TEST.reshape(Y_TEST.shape[0])))
    if i==0 or i==2 or i==20 or i==200 :    # print result for a = 1, 10, 100
        print("--------------------------------------------------------------")
        print("Beta({},{})".format(beta_a, beta_b))
        print("Training error rate: {}% ".format(ERROR_TRAIN[i]))
        print("Testing error rate: {}% ".format(ERROR_TEST[i]))


    i=i+1

# plot error rate corresponding to beta parameters
plt.figure()
plt.plot(beta_A, ERROR_TRAIN, 'b-', linewidth=1, label='Training')
plt.plot(beta_A, ERROR_TEST, 'g-', linewidth=1, label='Testing')
plt.xlabel('Beta(a,a)', fontsize=11)
plt.ylabel('Error Rate %', fontsize=11)
plt.title('Training / Testing Error Rates(%) Against Beta Distribution')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.show()
print("==============================================================================================")

################ Q2. Gaussian Naive Bayes ################
print(">>>>>>>>>>>>Gaussian Naive Bayes<<<<<<<<<<<<")
# estimate both the class prior probability and the class conditional mean and variance of each feature using ML
# Class 1
X_TRAIN_Class1 = X_TRAIN_LOG * Y_TRAIN
X_TRAIN_Class1 = X_TRAIN_Class1[~np.all(X_TRAIN_Class1 == 0, axis=1)]

Class1_mean = np.mean(X_TRAIN_Class1, axis=0)
Class1_var = np.mean(np.square(X_TRAIN_Class1 - Class1_mean), axis=0)

# Class 0
X_TRAIN_Class0 = X_TRAIN_LOG * np.logical_not(Y_TRAIN).astype(int)
X_TRAIN_Class0 = X_TRAIN_Class0[~np.all(X_TRAIN_Class0 == 0, axis=1)]

Class0_mean = np.mean(X_TRAIN_Class0, axis=0)
Class0_var = np.mean(np.square(X_TRAIN_Class0 - Class0_mean), axis=0)


# predict
def Predict_2(X_Samples):

    # calculate feature likelihood using Gaussian distribution

    class1_cond = np.log((1 / (np.sqrt(2 * np.pi * Class1_var))) *
                         np.exp(-((X_Samples - Class1_mean) ** 2 / (2 * Class1_var))))


    class0_cond = np.log((1 / (np.sqrt(2 * np.pi * Class0_var))) *
                         np.exp(-((X_Samples - Class0_mean) ** 2 / (2 * Class0_var))))


    # calculate posterior probability and compare

    gaussian_bayes_c1 = C1_Prior_ML + np.sum(class1_cond, axis=1)
    gaussian_bayes_c0 = C0_Prior_ML + np.sum(class0_cond, axis=1)

    for i in range(gaussian_bayes_c1.shape[0]):
        if (gaussian_bayes_c1[i] == gaussian_bayes_c0[i]):
            print('equal')

    gaussian_bayes_prob = np.vstack((gaussian_bayes_c0, gaussian_bayes_c1))
    return np.argmax(gaussian_bayes_prob, axis=0)

ERROR_TRAIN_GAUSSIAN = []
ERROR_TEST_GAUSSIAN = []

ERROR_TRAIN_GAUSSIAN = error_rate(Predict_2(X_TRAIN_LOG), Y_TRAIN.reshape(Y_TRAIN.shape[0]))
ERROR_TEST_GAUSSIAN = error_rate(Predict_2(X_TEST_LOG), Y_TEST.reshape(Y_TEST.shape[0]))


print("Training error rate: {}% ".format(ERROR_TRAIN_GAUSSIAN))
print("Testing error rate: {}% ".format(ERROR_TEST_GAUSSIAN))
print("==============================================================================================")

################ Q3. Logistic Regression ################

print(">>>>>>>>>>>>Logistic Regression<<<<<<<<<<<<")
# produce lambd hyperparameter
lambd1 = np.linspace(1,9,num=9)
lambd2 = np.linspace(10,100,num=19)
lambd = np.hstack((lambd1,lambd2))
print("L2 regularization parameters:")
print(lambd)

# fit a logistic regression model with bias term and with L2 regularization
# initialize weights and bias to 0
weights = np.zeros(X_TRAIN.shape[1])
print("initial weights:")
print(weights)
bias = np.zeros(1)
print("initial bias:")
print(bias)


error_train = []
error_train_final = []
error_test = []


for  L2_lambd in zip(lambd):


    # iteration and update weights and bias until convergence
    for epoch in range(20):

        # calculate u=sigm(wx+b)
        x0 = 1
        z = X_TRAIN_LOG.dot(weights) + x0 * bias
        u = 1/(1 + np.exp(-z))

        # to introduce bias term, , denote new feature vector x (length (D+1)) with x0=1
        x0 = np.full((X_TRAIN.shape[0]), 1)
        X = np.column_stack((x0, X_TRAIN_LOG)) #X:N x D+1

        S = np.diag(u * (1 - u))

        # exclude bias from L2 regularization, set the first diagonal element to 0
        I = np.diag(np.concatenate([np.array([0]), np.full((X_TRAIN.shape[1]), 1)]))


        # to exclude bias from L2 regularization, denote new weights as a (D+1)X1 vector with first term setting to 0.
        # calculate gradient and second derivative H
        G = X.T.dot(u-Y_TRAIN.reshape(Y_TRAIN.shape[0])) + L2_lambd * np.concatenate([np.array([0]), weights])
        H = X.T.dot(S).dot(X) + L2_lambd * I

        # calculate inverse matrix of H
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # if H is not invertible
            H_inv = H

        descent = H_inv.dot(G)

        # update weights and bias
        bias = bias - descent[0]
        weights = weights - descent[1:]

        # calculate new u for prediction
        x0 = 1
        z = X_TRAIN_LOG.dot(weights) + x0 * bias
        u = 1 / (1 + np.exp(-z))

        # compare p(y=1|x) and p(y=0|x)
        predict_train = np.where(u>=0.5,1,0)

        error_train.append(error_rate(Y_TRAIN.reshape(Y_TRAIN.shape[0]), predict_train))


        n_epoch=epoch+1

        # judge convergence by the error rate no longer changing

        if len(error_train) >= 3:
            if error_train[-1]==error_train[-2]==error_train[-3]:
                error_train_final.append(error_train[-1])

                # predict the test set with weights that are no longer updated
                x0 = 1
                z_test = X_TEST_LOG.dot(weights) + x0 * bias
                u_test = 1 / (1 + np.exp(-z_test))

                predict_test = np.where(u_test >= 0.5, 1, 0)
                error_test.append(error_rate(Y_TEST.reshape(Y_TEST.shape[0]), predict_test))

                if L2_lambd[0] == 1 or L2_lambd[0]==10 or L2_lambd[0]==100:

                    # print results

                    print("--------------------------------------------------------------")
                    print("L2 Regularization Parameter lambda = ", L2_lambd[0])
                    print("iteration number:",n_epoch)
                    print("Training Error Rate: {}%".format(error_train_final[-1]))
                    print("Test Error Rate: {}%".format(error_test[-1]))

                break


# print lowest error rate
print("----------------------------------------------------------------------------------------------")
train_min_idx = np.argmin(error_train_final, axis=0)
test_min_idx = np.argmin(error_test, axis=0)
print("lambda {} achieved best performance on train dataset".format(lambd[train_min_idx]))
print("minimum training error rate: {}%".format(error_train_final[train_min_idx]))
print("lambda {} achieved best performance on test dataset".format(lambd[test_min_idx]))
print("minimum testing error rate: {}%".format(error_test[test_min_idx]))


# plot error rate corresponding to lambda
plt.figure()
plt.plot(lambd, error_train_final, 'g-', linewidth=1.5, label='Training')
plt.plot(lambd, error_test, 'b-', linewidth=1.5, label='Testing')
plt.xlabel('lambda', fontsize=11)
plt.ylabel('Error Rate %', fontsize=11)
plt.title('Training / Testing Error Rates(%) versus L2 Regularization Parameter')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.show()
print("==============================================================================================")

################ Q4. K-Nearest Neighbors ################
print(">>>>>>>>>>>>KNN Classifier<<<<<<<<<<<<")
# produce K hyperparameter
K1 = np.linspace(1,9,num=9)
K2 = np.linspace(10,100,num=19)
K = np.hstack((K1,K2))

print("K values")

print(K)


# use Euclidean Distance to measure distance between neighbors
def dist(x_samples):

    # calculate Euclidean Distance from the predicting sample to each training set sample in D dimensional space
    x_train_square=np.sum(np.square(X_TRAIN_LOG),axis=1)
    x_test_square=np.sum(np.square(x_samples),axis=1)
    distance=np.sqrt(x_test_square.reshape(x_test_square.shape[0],1)-2*np.dot(x_samples,X_TRAIN_LOG.T)+x_train_square+1e-5)

    return distance


def predict(X_samples,k):

    # find K nearest train samples around x
    n=X_samples.shape[0]
    distance = dist(X_samples)
    k_idx = np.argpartition(distance, k, axis=1)[:, :k]
    k_idx=k_idx.reshape(n*k)

    # find corresponding labels of K neighbors
    labels=Y_TRAIN.reshape(Y_TRAIN.shape[0])[k_idx]
    labels=labels.reshape(n,k)

    # calculate the number of neighbors points belonging to class ܿ1
    k1=np.sum(labels,axis=1)

    # if ݇ k1 > k0, x is predicted as spam email, and vice versa
    return np.where(k1/k>=0.5,1,0)

error_train_KNN = []
error_test_KNN = []

# begin predicting by setting different hyperparameter k
for k in zip(K):

    k=int(k[0])
    predict_Ytrain=predict(X_TRAIN_LOG,k)
    error_train_KNN.append(error_rate(Y_TRAIN.reshape(Y_TRAIN.shape[0]),predict_Ytrain))

    predict_Ytest=predict(X_TEST_LOG,k)
    error_test_KNN.append(error_rate(Y_TEST.reshape(Y_TEST.shape[0]),predict_Ytest))

    if k==1 or k==10 or k==100:


        print("K = ", k)
        print("TRAINING ERROR RATE: {}% ".format(error_train_KNN[-1]))
        print("TESTING ERROR RATE: {}% ".format(error_test_KNN[-1]))



print("----------------------------------------------------------------------------------------------")
# print lowest error rate
test_min_idx = np.argmin(error_test_KNN, axis=0)
print("K = {} achieved best performance on test dataset".format(K[test_min_idx]))
print("minimum testing error rate: {}%".format(error_test_KNN[test_min_idx]))

# plot error rate corresponding to K
plt.figure()
plt.plot(K, error_train_KNN, 'b-', linewidth=1.5, label='Training')
plt.plot(K, error_test_KNN, 'r-', linewidth=1.5, label='Testing')
plt.xlabel('K', fontsize=11)
plt.ylabel('Error Rate %', fontsize=11)
plt.title('KNN Classifier Performance')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.show()
print("==============================================================================================")









