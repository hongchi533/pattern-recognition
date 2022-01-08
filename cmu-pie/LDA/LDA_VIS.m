close all;
clear all;
load('../facedataset.mat');

% Visualization of 500 samples 
train_size=size(train_data_500,1);

% mean of features of whole dataset 
mean_total=mean(train_data_500);
Mui=zeros(26,1024);
Sw=zeros(1024,1024);
Sb=zeros(1024,1024);

% folder I choose
class=[22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47, 55, 1, 65, 5, 60,0]; 

for i=1:26
    % find train_data belonging to class i
    ind=find(train_label_500==class(i));
    Xi=train_data_500(ind,:);
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



S=pinv(Sw)*Sb;
% W contains eigenvectors and S contains eigenvalues 
[W, S, V]=svd(S);
W_new=W;


% reduce the dimensionality	of	vectorized	images to 2	and	3 respectively
train_2d=train_data_500*W_new(:,1:2);
train_3d=train_data_500*W_new(:,1:3);

% 2d visualization
figure(1);
hold on
s1 = scatter(train_2d(1:498,1),train_2d(1:498,2),15,train_label_500(1,1:498),'filled');
s2 = scatter(train_2d(499:500,1),train_2d(499:500,2),40,'r','pentagram','filled');

title('2D visualization of LDA');
legend([s2],{'My photos'},'Location','northwest');
hold off

figure(2);
hold on

% 3d visualization
s3 = scatter3(train_3d(1:498,1),train_3d(1:498,2),train_3d(1:498,3),15,train_label_500(1,1:498),'filled');
s4 = scatter3(train_3d(499:500,1),train_3d(499:500,2),train_3d(499:500,3),40,'r','pentagram','filled');

grid on
title('3D visualization of LDA');
legend(s4,'My photos');
view([1 2 1]);
hold off



        
    
    
    
    
    