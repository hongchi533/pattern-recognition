!! MAKE SURE "libsvm-3.24"  toolbox is in the <Your_Project_Name\SVM>, you can download libsvm-3.25 package, but remember to modify the path. e.g addpath('libsvm-3.24\matlab')->addpath('libsvm-3.25\matlab') in 'SVM.m'.

Here is the content of this project, make sure you do arrange your documents in such way (after data preparation):

├── Readme.md                   
├── PIE
│   ├── 0
│   ├── 1
│   ├── ...
│   └── 67
├── CNN_PIE
│   ├── test
│   │   ├── 0
│   │   ├── 1
│   │   ├── 5
│   │   ├── ...
│   │   └── 65
│   └── train
│   │   ├── 0
│   │   ├── 1
│   │   ├── 5
│   │   ├── ...
│   │   └── 65
├── PCA
│   ├── PCA_CLA.m
│   ├── PCA_VIS.m
│   └── nearest_neighbor.m
├── LDA
│   ├── LDA_cla.m
│   ├── LDA_VIS.m
│   └── nearest_neighbor.m
├── SVM
│   ├── libsvm-3.24
│   │   ├── ...
│   │   └── matlab
│   └── SVM.m
├── CNN
│   └── CNNPIE.py
├── GMM
│   └── GMM.py
├── img2gray.m
├── img26_loader.m
├── cnndataloader.m
└── facedataset.mat

1. Environment:

Python --version >= 3.7
Matlab --version >= 2016b
Third Party: Pillow, Numpy, matplotlib,  scipy.io, pytorch, libsvm, sci-kit learn.
*You can obtain the corresponding installation commands of the pytorch on official website 'https://pytorch.org/'  according to your server configuration.
*The SVM toolbox can be downloaded	from https://www.csie.ntu.edu.tw/~cjlin/libsvm/

2. Interactive:

*Selfies have been converted to grayscale image and stored in  `./PIE/0/`, so you don't need run 'img2gray.m'.

 (1) Open matlab
 (2) Run the 'img26_loader.m'  to get 'facedataset.mat'. Make sure include CMU PIE dataset in the folder PIE in the same path.
 (3) Run the 'cnndataloader.m'  to get 'CNN_PIE' folder
 (4) Run 'PCA_CLA.m',  'PCA_VIS.m',  'LDA_VIS.m',  'LDA_cla.m', 'SVM.m' directly. The dataset will be automatically read from the previously generated 'facedataset.mat' file. Before run SVM.m, make sure 'libsvm-3.24' folder is in the same path.

 (5) Open Anaconda Prompt or  Python environment prompt
 (6) Open project Your_Project_Name
 (7) Navigate to ..\Your_Project_Name\CNN\ by typing:
                                  > cd CNN
 (8) In terminal, type:
                                 > python CNNPIE.py
 If your pytorch is installed in anaconda virtual environment, type:
                                                        > conda activate your_env_name
                                                        > python CNNPIE.py

 (9) Return to the previous directory:
                                    > cd ../                                             
 (10) Navigate to ..\Your_Project_Name\GMM\ by typing: > cd GMM
 (11) In terminal, type:
                                 > python GMM.py

3. Description:
This project implemented several algorithms: PCA/LDA/SVM/GMM/CNN, on dataset combined with CMU PIE dataset and 10 selfies. Model performance are visualized.

'img2gray.m' can convert my selfies  to grayscale images and save them into `./PIE/0/`.
'img26_loader.m' can randomly choose 25 out of the 67 subjects from CMU PIE set and use 70% of the provided images for training and use the remaining 30% for testing. Besides, it can resize 10 my photos into the same resolution (32*32)  as CMU PIE set and save them all as .mat file. 
'cnndataloader.m' can combine chosen 26 subjects into the 'CNN_PIE' folder for code 'CNNforPIE.py'.
'PCA_VIS.m' can use PCA rule for feature dimensionality reduction and visualization.
'PCA_CLA.m' can use PCA rule  for feature dimensionality reduction and classification.
'LDA_VIS.m' can use LDA rule  for feature dimensionality reduction and visualization.
'LDA_cla.m' can use LDA rule  for feature dimensionality reduction and classification.
'nearest_neighbor.m' is a function file, can be called by PCA and LDA to implement classification.
'SVM.m' can train svm model to fulfill classification.
'GMM.m' can fulfill 3 components GMM model for EM-GMM clustering.
'CNNPIE.py' can train a CNN using pytorch with two convolutional layers and two fully connected layer (the last is output layer), the network architecture is specified as: 20(conv)-50(conv)-500(fc)-26(fc).