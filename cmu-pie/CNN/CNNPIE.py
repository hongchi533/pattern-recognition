import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets
from PIL import Image


torch.manual_seed(10) # Set the seed for each experiment to be consistent
Batch_size = 10 # Set batchsize

# Load data
def readImg(path):
    return Image.open(path)

# When using pytorch's nn.module, you can know from the instructions that its standard input is "B, C, H, W", which means:

# B - batchsize，For example, when you use dataloder set the batchsize is 64 then this item is 64.
# C - channel,which is the number of channels of the input matrix, is 1 if you enter an gray picture.
# H - high, which is the height of the input matrix.
# W - width，which is the width of the input matrix.

data_transform = transforms.Compose([transforms.ToTensor()])
# transfer the image to Tensor,normalize to [0,1]

# ImageFolder assumes that all files are saved by folder,
# that pictures of the same category are stored under the same folder,
# that the folder is named class name,
# and that label is sorted in the order of folder names and stored in a dictionary.

train_data = datasets.ImageFolder(root='../CNN_PIE/train', transform=data_transform, loader=readImg)
train_loader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=0)

# batch_size(int, optional): the number of samples in per batch
# shuffle(bool, optional): at the beginning of each epoch, the data is reordered
test_data = datasets.ImageFolder(root='../CNN_PIE/test', transform=data_transform, loader=readImg)
test_loader = DataLoader(test_data, batch_size=1278, shuffle=True, num_workers=0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first conv layer
        self.conv1 = nn.Sequential(
            # input[1,32,32]
            nn.Conv2d(
                in_channels=1,    #
                out_channels=20,  # the number of filters
                kernel_size=5,    # 5x5
                stride=1,         #
                #padding=2,        # padding 0 to the outside of the feature image
            ),
            #  output[20,(32+2*padding-kernel_size)/stride+1,(32+2*padding-kernel_size)/stride+1]
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # After pooling, the output is [20, 14, 14], then next convolution layer.
        )
        # second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,    # 同上
                out_channels=50,
                kernel_size=5,
                stride=1,
                #padding=2
            ),
            # output [50,10,10]
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output[50,5,5]
        )
        # third layer -fully connected layer, input [50 x 5 x 5], output [500]
        self.fc = nn.Sequential(
            nn.Linear(50*5*5, 500),
            nn.ReLU())
        # fourth layer -fully connected layer, input [500], output [26]
        self.output = nn.Linear(in_features=500, out_features=26)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 50,5,5]
        x = x.view(x.size(0), -1) # Reserve the batchsize, multiply the following dimensions together .batch, 50 x 8 x 8
        x = self.fc(x)
        output = self.output(x)   # ouput [batch,26]
        return output

# Set hyperparameters
epoches = 35
learning_rate = 1e-3
weight_decay=1e-6


def main():

    cnn = CNN()
    print(cnn)

    # Define optimizers and loss functions
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate,weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    # Cycle
    for epoch in range(epoches):
        print("Process the {} epoch".format(epoch))
        # train
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
			# To display the accuracy in real time.
            if step % 100 == 0:

                for test_x, test_y in test_loader:

                    test_output = cnn(test_x)
                    predict_y = torch.argmax(test_output, dim=1)
                    total = test_y.size(0)

                accuracy = ((predict_y  == test_y).sum()).item()/ float(test_y.size(0))*100
                print('Epoch end: ', epoch,'Step: ', step, '| train loss: %.5f' % loss.item(), '| test accuracy: %.3f' % accuracy)

        # after one epoch, test the performance on test set

        for test_x, test_y in test_loader:
            test_output = cnn(test_x)
            predict_y = torch.argmax(test_output, dim=1)
            total = test_y.size(0)

        print('Test size: ',total)

        accuracy_total = ((predict_y == test_y).sum()).item() / float(test_y.size(0))
        print('Step: ', step,': Finished one epoch | ','Test Accuracy= {:.3%}\n'.format(accuracy_total))


if __name__ == "__main__":
    main()


