%Class_20 = [22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47];% 20 Folders numbers I choose
Class_25=[22, 49, 26, 53, 40, 64, 44, 37, 9, 8, 62, 58, 25, 57, 6, 33, 52, 19, 48, 47, 55, 1, 65, 5, 60];
rng(3);
rand_num=randperm(170);

% divided by 7:3

    for i=1:25
         folder_Path=dir(strcat('PIE\',num2str(Class_25(i)),'\*.jpg'));
        train_Path=strcat('.\CNN_PIE\train\',num2str(Class_25(i)),'\');
        test_Path=strcat('.\CNN_PIE\test\',num2str(Class_25(i)),'\');
        mkdir(train_Path)
        mkdir(test_Path)
        for j=1:119
        picname=strcat('PIE\',num2str(Class_25(i)),'\',folder_Path(rand_num(j)).name);
        copyfile(picname,train_Path);
        end
        for j=120:170
        picname=strcat('PIE\',num2str(Class_25(i)),'\',folder_Path(rand_num(j)).name);
        copyfile(picname,test_Path);
        end   
    end

    % add my own photos into train and test folders
    folder_Path=dir(strcat('PIE\',num2str(0),'\*.jpg'));
    train_Path=strcat('.\CNN_PIE\train\',num2str(0),'\');
    test_Path=strcat('.\CNN_PIE\test\',num2str(0),'\');
    mkdir(train_Path)
    mkdir(test_Path)
    
    rng(1);
    rand_num2=randperm(10);
        for j=1:7
        picname=strcat('PIE\0\','\',folder_Path(rand_num2(j)).name);
        copyfile(picname,train_Path);
        end
        for j=8:10
        picname=strcat('PIE\0\','\',folder_Path(rand_num2(j)).name);
        copyfile(picname,test_Path);
        end   
