import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from module import Net
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using GPU: %s'%torch.cuda.is_available())
n_epochs = 10
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
train_data = torchvision.datasets.MNIST('./datasets', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data,batch_size=batch_size_train, shuffle=True)
test_data = torchvision.datasets.MNIST('./datasets', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = DataLoader(test_data,batch_size=batch_size_test, shuffle=True)

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = []  #[i * len(test_loader.dataset) for i in range(n_epochs + 1)]
check_test = []
check_pred = []

def train():
    network.to(device)
    for i in range(1,n_epochs+1):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data.to(device))
            target = target.cuda()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data),len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),loss.item()))
                train_losses.append(loss.item())
                train_counter.append((n_epochs*batch_idx * 64))
                torch.save(network.state_dict(), './model.pth')
                torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    network_state_dict = torch.load('model.pth')
    network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('optimizer.pth')
    optimizer.load_state_dict(optimizer_state_dict)
    network.eval()
    correct = 0
    with torch.no_grad():
        for index,(data,target) in enumerate(test_loader):
            test_loss = 0
            output = network(data.to(device))
            target = target.cuda()
            test_loss += F.nll_loss(output, target)
            # print(output)
            pred = output.data.max(1, keepdim=True)[1]
            # print(pred)
            correct += pred.eq(target.data.view_as(pred)).sum()
            # test_loss /= batch_size_test
            if index % log_interval == 0:
                check_test.append(data[0])
                check_pred.append(pred[0].data.tolist())
                test_losses.append(test_loss.item())
                test_counter.append((index * 64))

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def lookpic():
    # print(check_test)
    print(check_pred)
    print(test_losses)
    for i in range(0,len(check_test)):
        check_test[i] = check_test[i].transpose(dim0=2,dim1=0)
        check_test[i] = check_test[i].transpose(dim0=1, dim1=0)
        plt.imshow(check_test[i])
        plt.show()
        input('qwq')

if __name__ == "__main__":
    # train()
    test()
    lookpic()
    print(len(check_target))
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    # print(check_test)
    # print(check_target)
    print(test_counter)
    print(test_losses)
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()