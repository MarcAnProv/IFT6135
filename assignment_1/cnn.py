import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import torchvision.datasets as datasets
import time

"""
Adapted from "https://github.com/pytorch/examples/blob/master/mnist/main.py#L39"
"""

# Dataset loading
# Dataset is an abstract base class (abstract base classes exist to be inherited, but never instantiated)
class MNIST_dataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    # Magic method : They're special methods that you can define to add "magic" to your classes
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y 


    def __len__(self):
        return len(self.data)

def loader(data, labels, batch_size):
        dataset = MNIST_dataset(np.expand_dims(data, axis=1), labels)
        # DataLoader is an iterator that can batch and shuffle the data
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return loader

# Defining the structure of the CNN
# nn.Module is the base class for all the neural network modules
class MNIST_cnn(nn.Module):
    # Define each layer
    def __init__(self):
        # Super is used to initialize the members of the base class
        super(MNIST_cnn, self).__init__()
        # 2D Conv (input channels, output channels, kernel size, stride, padding, dilation, groups -> number of blocked connections)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 30, 4, 1)
        self.conv3 = nn.Conv2d(30, 40, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Linear (input features, output features, bias=True)
        self.fc1 = nn.Linear(40*3*3, 500)
        self.fc2 = nn.Linear(500, 10)

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        # flatten feature map
        x = x.view(-1, 3*3*40)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output 

    
# Define the training method
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    # define the operation batch wise
    for batch_idx, (data, target) in enumerate(train_loader):
        # data to cpu or gpu
        data, target = data.to(device), target.to(device)
        # clear the gradient in the optimizer at the beginning of each backpropagation
        optimizer.zero_grad()
        # get out
        output = model(data)
        # define loss
        loss = F.nll_loss(output, target)
        # backpropagation to get the gradients
        loss.backward()
        # update the parameters
        optimizer.step()

        # training loss
        train_loss += loss.item()

        # show training progress
        if batch_idx % 500 == 0:
            # loss.item() gets the a scalar value held in the loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    return train_loss / len(train_loader)


# Define the test method
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # disabling gradient calculation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # calculate the right classification
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average the loss (batch-wise)
    test_loss /= len(test_loader.dataset)

    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

    return test_loss


def main():

    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TP1 MNIST CNN')
    #Training args
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    #parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                       # help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # set seed for random number generator
    torch.manual_seed(args.seed)

    # use GPU if available
    device = torch.device('cuda' if use_cuda else 'cpu')


    # MNIST dataset
    f = gzip.open('mnist.pkl.gz')
    data = pickle.load(f, encoding='latin1')

    train_data, train_labels = data[0]
    train_data = train_data.reshape(50000,28,28)
    valid_data, valid_labels = data[1]
    valid_data = valid_data.reshape(10000,28,28)
    test_data, test_labels = data[2]
    test_data = test_data.reshape(10000,28,28)

    # create dataloader
    train_loader = loader(train_data, train_labels, args.batch_size)
    valid_loader = loader(valid_data, valid_labels, args.batch_size)
    test_loader = loader(test_data, test_labels, args.batch_size)

    # create neural network object
    model = MNIST_cnn().to(device)

    # print model summary
    summary(model, (1, 28, 28))

    # choose optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loss = []
    validation_loss = []

    start_time = time.time()
    # Start training
    for epoch in range(1, args.epochs+1):
        # training loss
        t_loss = train(args, model, device, train_loader, optimizer, epoch)
        # validation loss
        v_loss = test(args, model, device, valid_loader)

        train_loss.append(t_loss)
        validation_loss.append(v_loss)
    print("--- %s seconds ---" % (time.time() - start_time))

    # plot training and validation loss for each epoch
    x = list(range(1, 11))
    plt.figure()
    plt.title('training and validation loss for each epoch')
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.plot(x, train_loss, label='training loss')
    plt.plot(x, validation_loss, label='validation loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # test set
    print("Test accuracy:")
    v_loss = test(args, model, device, test_loader)

    # Save the trained model(which means parameters)
    if(args.save_model):
        torch.save(model.state_dict(), 'hmw1_cnn_mnist')

if __name__ == '__main__':
    main()









