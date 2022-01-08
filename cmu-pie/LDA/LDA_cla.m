close all;
clear all;
load('../facedataset.mat');

train_size=size(train_data,1);

% mean of features of whole dataset
mean_total=mean(train_data);

Mui=zeros(26,1024);
Sw=zeros(1024,1024);
Sb=zeros(1024,1024);

% the folder I choose
class=[22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47, 55, 1, 65, 5, 60, 0]; 

for i=1:26
    % find train_data belonging to class i
    ind=find(train_label==class(i));
    Xi=train_data(ind,:);
    % calculate mean of train data belonging to class i
    Mui(i,:)=mean(Xi);
    class_ni(i,:)=size(Xi,1);
    % obtain number of train data belonging to class i
    ni= class_ni(i,:);
    Pi=ni/train_size;
    % calculate Sw and Sb
    Si=(Xi-Mui(i,:))'*(Xi-Mui(i,:))./ni;
    Sw=Sw+Pi.*Si;
    Sb=Sb+Pi.*(Mui(i,:)-mean_total)'*(Mui(i,:)-mean_total);
end


% it returns sorted results
S=pinv(Sw)*Sb;
% W contain eigenvector and Lam contain eigenvalue 
[W, Lam, V]=svd(S);

% [W, D] = eig(Sb , Sw);
%  all_eigen_values = sum(D, 1);
% [~, I] = sort(all_eigen_values, 'descend');
% W= W(:, I);
% [W, D] = eig(Sb , Sw);
%  all_eigen_values = sum(D, 1);
%  [~, I] = sort(all_eigen_values, 'descend');
%  W= W(:, I);

% reduce dimensionality to 2
train_2d_lda = train_data * W(:, 1:2);
test_2d_lda= test_data * W(:, 1:2);
% do classification using nearest neighbor
[pie_acc, self_acc] = nearest_neighbor(train_2d_lda, train_label, test_2d_lda, test_label);
display(strcat('Dimensionality 2:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc*100), '%', ' | Accuracy on my own photos is:', num2str(self_acc*100), '%'));

% reduce dimensionality to 3 
train_3d_lda = train_data * W(:, 1:3);
test_3d_lda = test_data * W(:, 1:3);
% do classification using nearest neighbor
[pie_acc, self_acc] = nearest_neighbor(train_3d_lda, train_label, test_3d_lda, test_label);
display(strcat('Dimensionality 3:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc*100), '%', ' | Accuracy on my own photos is:', num2str(self_acc*100), '%'));

% reduce dimensionality to 9 do classification using nearest
% do classification using nearest neighbor
train_9d_lda = train_data * W(:, 1:9);
test_9d_lda = test_data * W(:, 1:9);
[pie_acc, self_acc] = nearest_neighbor(train_9d_lda, train_label, test_9d_lda, test_label);
display(strcat('Dimensionality 9:',' Accuracy on the CMU PIE test images is:', num2str(pie_acc*100), '%', ' | Accuracy on my own photos is:', num2str(self_acc*100), '%'));
















        
    
    
    
    
    