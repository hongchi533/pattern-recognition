close all;
clear all;
load('../facedataset.mat');

% get eigen vectors calculated based on PCA principle.
[Cell_all] =  PCA(train_data);
eigen_vectors=  Cell_all{1};

% reduce dimensionality to 40 
train_40d = train_data * eigen_vectors(:, 1:40);
test_40d = test_data * eigen_vectors(:, 1:40);

% do classification using nearest neighbor 
[pie_acc1, self_acc1] = nearest_neighbor(train_40d, train_label, test_40d, test_label);
display(strcat('Dimensionality 40:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc1*100), '%', ' | Accuracy on my own photos is:', num2str(self_acc1*100), '%'));

% reduce dimensionality to 80 and do classification 
% do classification using nearest neighbor
train_80d = train_data * eigen_vectors(:, 1:80);
test_80d = test_data * eigen_vectors(:, 1:80);
[pie_acc2, self_acc2] = nearest_neighbor(train_80d, train_label, test_80d, test_label);
display(strcat('Dimensionality 80:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc2*100), '%',' | Accuracy on my own photos is:', num2str(self_acc2*100), '%'));

% reduce dimensionality to 200 
% do classification using nearest neighbor
train_200d = train_data * eigen_vectors(:, 1:200);
test_200d = test_data * eigen_vectors(:, 1:200);
[pie_acc3, self_acc3] = nearest_neighbor(train_200d, train_label, test_200d, test_label);
display(strcat('Dimensionality 200:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc3*100),'%',' | Accuracy on my own photos is:', num2str(self_acc3*100), '%'));

% PCA function based on svd
%[U,S,V] = svd(A) performs a singular value decomposition of matrix A, such that A = U*S*V'.
function [Cell_all] = PCA( dataSet )  
    %[m,n] = size(dataSet);  
    data_mean = mean(dataSet);
    X = dataSet - data_mean ;
    S=X'*X;
    %   svd
    [U,D,V] = svd(S);
    d=diag(D);
    Cell_all={U,d};
end  
