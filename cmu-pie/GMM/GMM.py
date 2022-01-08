import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
import scipy.io # package for reading data in Matlab's .mat file format

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# Data loading
face_data = scipy.io.loadmat('../facedataset.mat')
X_TRAIN = np.array(face_data["train_data"])
X_TEST = np.array(face_data["test_data"])


# Call GMM package-GaussianMixture
def GMM (data):
    # 3 Gaussian components
    model = GaussianMixture(n_components=3)
    model.fit(data)
    Predict = model.predict(data)
    Prob = model.predict_proba(data)
    # return clustering results with probabilities
    return Predict,Prob


# call PCA package to do dimensionality reduction
def PCa (data,d):
    pca= PCA(n_components = d)
    pca.fit(data)
    x_red=pca.fit_transform(data)
    return x_red

# -----------------raw data----------------- #
raw_data=np.concatenate((X_TRAIN,X_TEST),axis=0)
Predict_raw,Prob_raw=GMM(raw_data)

# find the image belonging to each cluster with maximum probability
for i in range(3):
    max_face_raw = Prob_raw[:,i].argsort()[-1]
    face_1=raw_data[max_face_raw,:].reshape(32,32).T
    plt.subplot(1,3,i+1)
    plt.imshow(face_1, cmap ='gray')

plt.suptitle('raw data')
plt.show()


# -----------------200d data----------------- #
data_200=PCa(raw_data,200)
Predict_200,Prob_200=GMM(data_200)

for i in range(3):
    max_face_200 = Prob_200[:,i].argsort()[-1]
    face_2=raw_data[max_face_200,:].reshape(32,32).T
    plt.subplot(1,3,i+1)
    plt.imshow(face_2, cmap ='gray')
plt.suptitle('PCA-200d')
plt.show()


# -----------------80d data----------------- #
data_80=PCa(raw_data,80)
Predict_80, Prob_80=GMM(data_80)


for i in range(3):
    max_face_80 = Prob_80[:,i].argsort()[-1]
    face_3=raw_data[max_face_80,:].reshape(32,32).T

    plt.subplot(1,3,i+1)
    plt.imshow(face_3, cmap ='gray')
plt.suptitle('PCA-80d')
plt.show()



# Visualization
# Use PCA to reduce raw data to 2D for visualization
x_vis=PCa(raw_data,2)
# ax.scatter(X_train[:, 0], X_train[:, 1], s=0.8, c=y_train)
# Plot the test data with crosses
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('GMM clustering results on raw data')
ax.scatter(x_vis[:, 0],x_vis[:, 1],  c=Predict_raw)
plt.show()

# Use PCA to reduce 200D data to 2D for visualization
x_vis=PCa(data_200,2)
# ax.scatter(X_train[:, 0], X_train[:, 1], s=0.8, c=y_train)
# Plot the test data with crosses
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('GMM clustering results on data reduced to 200d')
ax.scatter(x_vis[:, 0],x_vis[:, 1],  c=Predict_200)

plt.show()

# Use PCA to reduce 80D data to 2D for visualization
x_vis=PCa(data_80,2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('GMM clustering results on data reduced to 80d')
ax.scatter(x_vis[:, 0],x_vis[:, 1],  c=Predict_80)
plt.show()

