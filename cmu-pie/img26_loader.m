function  load_images()
% For PCA,LDA,SVM   
    folder_count = 25;
    image_count = 170;
    train_ratio = 0.7;
    train_count = round(image_count * train_ratio);
    test_count = round(image_count * (1 - train_ratio));
    train_data = zeros(folder_count * train_count+7,1024);
    test_data = zeros(folder_count * test_count+3,1024);
    train_data_500=zeros(500,1024);
    train_label_500=zeros(500,1024);
    class_25=[22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47, 55, 1, 65, 5, 60];   %the folder number I choose
   
    % load CMU PIE images
    for i=1:25

        folder_name = strcat('PIE\', num2str(class_25(i)), '\');
        pic = zeros(image_count,1024);
        for j=1:image_count
            pic_name = strcat(folder_name, num2str(j), '.jpg');
            pic(j, :) = reshape(imread(pic_name), [1, 1024]);    
        end
        
        pic = double(pic)/255;
        
        % shuffle the dataset
        rng(3);
        perm = randperm(170);
        train_data((i-1)*train_count+1:i*train_count,: ) = pic( perm(1:train_count),:);
        test_data((i-1)*test_count+1:i*test_count,:) = pic( perm(train_count+1:image_count),:);
    end
    
    % load own photo
    self_images = zeros(10, 1024);
    for i=1:10
       
        pic_name = strcat('PIE\0\', num2str(i), '.jpg');
        self_images(i, :) = reshape(imread(pic_name), [1, 1024]);
        
    end
    self_images = double(self_images)/255;
    
    train_data( folder_count*train_count+1:folder_count*train_count+7,:) = self_images(1:7, :);
    test_data(folder_count*test_count+1:folder_count*test_count+3,: ) = self_images(8:10, :);
    
    train_label = zeros(1, folder_count*train_count+7);
    test_label = zeros(1, folder_count*test_count+3);
    for i=1:25
        train_label((i-1)*train_count+1:i*train_count) = class_25(i)*ones(1, train_count);
        test_label((i-1)*test_count+1:i*test_count) = class_25(i)*ones(1,test_count);
    end
    train_label(folder_count*train_count+1:folder_count*train_count+7) = 0*ones(1, 7);
    test_label(folder_count*test_count+1:folder_count*test_count+3) = 0*ones(1, 3);

    % randomly sample 500 images from the CMU PIE training set and my own photos
    rng(5);
    rand_idx_498 = randperm((size(train_label,2)-7),498);
    rng(6);
    rand_idx_self_2 = (size(train_label,2)-7) + randperm(7,2);
    
    train_data_500=train_data(rand_idx_498,:);
    train_data_500(499:500,:)=train_data(rand_idx_self_2,:);
    
    train_label_500=train_label(1,rand_idx_498);
    train_label_500(1,499:500)=train_label(1,rand_idx_self_2);
    save('facedataset.mat','train_data','train_label','test_data','test_label','train_data_500','train_label_500');
    

% %For CNN
% %randomly transfer images from PIE folders  into CNN_PIE folders
%    Class_20 = [22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47];%Folder number I choose
% rand_num=randperm(170);
%     for i=1:20
%          folder_Path=dir(strcat('E:\PR-CA2\PIE\PIE\',num2str(Class_20(i)),'\*.jpg'));
%         train_Path=strcat('.\CNN_PIE\train\',num2str(Class_20(i)),'\');
%         test_Path=strcat('.\CNN_PIE\test\',num2str(Class_20(i)),'\');
%         mkdir(train_Path)
%         mkdir(test_Path)
%         for j=1:119
%         picname=strcat('E:\PR-CA2\PIE\PIE\',num2str(Class_20(i)),'\',folder_Path(rand_num(j)).name);
%         copyfile(picname,train_Path);
%         end
%         for j=120:170
%         picname=strcat('E:\PR-CA2\PIE\PIE\',num2str(Class_20(i)),'\',folder_Path(rand_num(j)).name);
%         copyfile(picname,test_Path);
%         end   
%     end
% 
%     %ADD selfie
%     folder_Path=dir(strcat('E:\PR-CA2\PIE\PIE\',num2str(0),'\*.jpg'));
%     train_Path=strcat('.\CNN_PIE\train\',num2str(0),'\');
%     test_Path=strcat('.\CNN_PIE\test\',num2str(0),'\');
%     mkdir(train_Path)
%     mkdir(test_Path)
%     rand_num2=randperm(10);
%         for j=1:7
%         picname=strcat('E:\PR-CA2\PIE\PIE\0\','\',folder_Path(rand_num2(j)).name);
%         copyfile(picname,train_Path);
%         end
%         for j=8:10
%         picname=strcat('E:\PR-CA2\PIE\PIE\0\','\',folder_Path(rand_num2(j)).name);
%         copyfile(picname,test_Path);
%         end   

 end
