import numpy
import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

print(train_data)
print(test_data)
print(train_data.data.size())
print(train_data.targets.size())

import matplotlib.pyplot as plt

#plt.imshow(train_data.data[10000], cmap='gray')
#plt.title('%i' % train_data.targets[10000])
#plt.show()

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(
        train_data, 
        batch_size=100, 
        shuffle=True, 
        num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(
        test_data, 
        batch_size=100, 
        shuffle=True, 
        num_workers=1),
}
loaders

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

cnn = CNN()
print(cnn)
loss_func = nn.CrossEntropyLoss()
loss_func #what does this even do
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
optimizer #idk what this does

from torch.autograd import Variable

num_epochs = 3

def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()
            
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1,
total_step, loss.item()))
            pass
        
        pass
    
    pass

train(num_epochs, cnn, loaders)

def test():
    # Test the model
    cnn.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass

    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
        

    
pass

test()

print("-------------------")
sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:30].numpy()
actual_number 

test_output, last_layer = cnn(imgs[:30])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

A = np.array(actual_number)
B = np.array(pred_y)

print(np.subtract(A,B))

print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')
