function [pie_acc, self_acc] = nearest_neighbor(train_images, train_label, test_images, test_label)

% find the nearest training point for the testing point (k=1)
Idx = knnsearch(train_images,test_images);

% extract corresponding labels
Class =train_label(:,Idx);

% calculate the classification	accuracy on	the	CMU	PIE	test images	and	my own photo separately
pie_acc = sum(Class(1:51*25) ==  test_label(1:51*25)) / (size(test_label, 2)-3);
self_acc = sum(Class(51*25+1:51*25+3) == test_label(51*25+1:51*25+3)) / 3;

end

    
