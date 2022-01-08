close all;
clear all;

load('../facedataset.mat');

% Visualization of 500 samples 
% get eigen vectors using principle of PCA, the function is defined at the bottom
[Cell_all] =  PCA(train_data_500);
eigen_vectors=  Cell_all{1};

% Randomly	sample	500	images in imagesloader.m, 
% reduce the dimensionality	of	vectorized	 images	 to	 2	 and	 3	respectively
FinalData_2 = train_data_500*eigen_vectors(:,1:2);  
FinalData_3 = train_data_500*eigen_vectors(:,1:3); 

% visualization
figure(1);
hold on
s1 = scatter(FinalData_2(1:498,1),FinalData_2(1:498,2),5,train_label_500(1,1:498),'filled'); % plot projected CMU PIE data in 2d
s2 = scatter(FinalData_2(499:500,1),FinalData_2(499:500,2),40,'r','pentagram','filled'); % plot projected my photos data point in 2d

title('2D visualization of PCA');
legend(s2,'My photos');
hold off

figure(2);
hold on

s3 = scatter3(FinalData_3(1:498,1),FinalData_3(1:498,2),FinalData_3(1:498,3),5,train_label_500(1,1:498),'filled');% plot projected CMU PIE data in 3d
s4 = scatter3(FinalData_3(499:500,1),FinalData_3(499:500,2),FinalData_3(499:500,3),40,'r','pentagram','filled');% plot projected my data in 3d

grid on
title('3D visualization of PCA');
%legend([s3 s4],{'CMU PIE data','My photos'});
legend(s4,'My photos');
view([1 2 1]);
hold off

[m,n] = size(eigen_vectors);
eigenfice=cell(1,m);

% rebuild eigenfaces
for i=1:m
    eigenfice{1,i} = eigen_vectors(:,i);
    eigenfice{1,i} = reshape(eigenfice{1,i},32,32);
end
% plot eigenface1,2,3
figure(3);
for i=1:3
    subplot(2,5,i);p=eigenfice{i};imshow(p,[]);
    title(sprintf('Eigenface%d',i));
end 

% PCA function based on eig
function [Cell_all] = PCA( dataSet )  
    [m,n] = size(dataSet);  
    data_Mean = mean(dataSet);  
    data_Cen=dataSet-data_Mean;
    dataCov = cov(data_Cen);  
    [V, D] = eig(dataCov);  
    % By default, eig does not always return sorted eigenvalues ??and eigenvectors. 
    % We can use the sort function to sort the eigenvalues ??in descending order and reorder the corresponding eigenvectors.
    [d, index] = sort(diag(D),'descend');  
     d=reshape(diag(D),1,1024);
    eigen_vectors = V(:,index);
    eigen_values = d(1, index);
    Cell_all={eigen_vectors,eigen_values};

end  

% PCA function based on svd
% function [Cell_all] = PCA( dataSet )  
%     train_mean = mean(dataSet);
%     X = dataSet - train_mean ;
% %     S=X'*X;
%     %   svd
%     [U,D,V] = svd(X');
%      d=diag(D);
%     Cell_all={U,d};
% end  


