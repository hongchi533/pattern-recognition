close all;
clear all;

% load data
load('../facedataset.mat');

% load libSVM(libsvm-{version}\matlab)
addpath('libsvm-3.24\matlab')

% implement PCA based on svd
% centralize the data
mean_train = mean(train_data,1);
train_cen = train_data - mean_train;
% svd
S= train_cen'*train_cen;
[W,D,V] = svd(S);
lam = diag(D);


% train svm model with raw data
% train_data=(mapminmax(train_data'))';
% test_data=(mapminmax(test_data'))';

% train svm model using raw images
model_1 = svmtrain(train_label',train_data,'-t 0 -c 0.01 -q');
model_2 = svmtrain(train_label',train_data,'-t 0 -c 0.1 -q');
model_3 = svmtrain(train_label',train_data,'-t 0 -c 1 -q');

% do predict on test dataset, get accuracy
[predict_label, accuracy_1,dec_values] = svmpredict(test_label', test_data, model_1);
[~, accuracy_2, ~] = svmpredict(test_label', test_data, model_2);
[~, accuracy_3, ~] = svmpredict(test_label', test_data, model_3);



% train svm model after PCA pre-processing	with dimensionality	 of	 80	
train_80 = train_data*W(:,1:80); 
test_80 = test_data*W(:,1:80);

model_4 = svmtrain(train_label', train_80, '-t 0 -c 0.01');
model_5 = svmtrain(train_label', train_80, '-t 0 -c 0.1');
model_6 = svmtrain(train_label', train_80, '-t 0 -c 1');

% do predict on test dataset, get accuracy
[~, accuracy_4, ~] = svmpredict(test_label', test_80, model_4); 
[~, accuracy_5, ~] = svmpredict(test_label', test_80, model_5); 
[~, accuracy_6, ~] = svmpredict(test_label', test_80, model_6); 



% train svm model after PCA pre-processing	with dimensionality	 of	 200

train_200 = train_data*W(:,1:200); 
test_200 = test_data*W(:,1:200); 

model_7 = svmtrain(train_label', train_200, '-t 0 -c 0.01');
model_8 = svmtrain(train_label', train_200, '-t 0 -c 0.1');
model_9 = svmtrain(train_label', train_200, '-t 0 -c 1');


[~, accuracy_7, ~] = svmpredict(test_label', test_200, model_7); 
[~, accuracy_8, ~] = svmpredict(test_label', test_200, model_8); 
[~, accuracy_9, ~] = svmpredict(test_label', test_200, model_9); 


fprintf('Accuracy using the	raw	face images with C is 0.01 is: %f\n', accuracy_1(1));
fprintf('Accuracy using the	raw	face images with C is 0.1 is: %f\n', accuracy_2(1));
fprintf('Accuracy using the	raw	face images with C is 1 is: %f\n', accuracy_3(1));

fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 80 with C is 0.01 is: %f\n', accuracy_4(1));
fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 80 with C is 0.1 is: %f\n', accuracy_5(1));
fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 80 with C is 1 is: %f\n', accuracy_6(1));

fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 200 with C is 0.01 is: %f\n', accuracy_7(1));
fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 200 with C is 0.1 is: %f\n', accuracy_8(1));
fprintf('Accuracy after PCA pre-processing	with dimensionality	 of	 200 with C is 1 is: %f\n', accuracy_9(1));



