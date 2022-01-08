clc;clf;clear;close all;

% transfer selfies to grayscale 32*32 format as CMU PIE
Path=dir('mine\*.jpg');

for i=1:10
    % load selfies
    self_pic=imread(strcat('mine\',Path(i).name));
    % convert to grayscale format
    self_pic=rgb2gray(imresize(self_pic,[32,32]));
    
    % store to a new folder 
    image_name=num2str(i);
     str0='PIE\0\';
    str1=image_name;
     str2='.jpg'  
     save_path=[str0,str1,str2];
        
  imwrite(self_pic,save_path);
    
   
end

