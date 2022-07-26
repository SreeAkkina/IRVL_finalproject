import time
import torchvision.models as models
import torchvision.transforms as transforms 
import os
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn as nn
from utils.evalutils import AverageMeters
# lr means learning rate 

DATADIR = 'training_set/'  # defining the path to the data

class CatDogDataset(Dataset):
    def __init__(self, dataset_directory, transform=None):  # initiates
        self.datalocation = dataset_directory
        self.classes = ['cats', 'dogs']  # self is basically the instance of the class in java terms
        self.data = []
        self.last_cat = 0
        self.transform = transform

        if not os.path.exists("cache/data.pt"):  # if this path thing doesn't exist, then enter the loop and basically create one 
            for i in self.classes:
                path = os.path.join(dataset_directory,i)
                for idx, img in enumerate(tqdm(os.listdir(path))):
                    data_img = Image.open(os.path.join(path, img)).resize((256, 256))  # resizes the image into a more pleasant size
                    data_img = transforms.PILToTensor()(data_img)  # transforms the image to a tensor
                    self.data.append(data_img)  # adds that back to the original data
                    if i == 'cats': self.last_cat = idx  # finds the index of the last cat so the computer knows where to separate cat and dog data files, even after randomization
        else:  # if it already exists, then do some stuff with the data
            data_dict = torch.load("cache/data.pt")
            self.data = data_dict["data"]
            self.last_cat = data_dict["last_cat"]
            return

        self.data = torch.stack(self.data)
        torch.save({
            "data": self.data,
            "last_cat": self.last_cat
        }, "cache/data.pt")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = transforms.RandomCrop(224)(img).float()  # data augmentation
        img = self.transform(img)
        if idx <= self.last_cat:
            label = 0
        else:
            label = 1
        return img, label
# creating arguments that you can edit in terminal ((neil) ➜  Documents python day6neil_train.py --batch_size=16)
parser = argparse.ArgumentParser(description='A Cat and Dog Classifier')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--momentum", default=0.9)
args = parser.parse_args()

# normalizing the colors using standard deviation to make it easier for the computer
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# applying to normalize function to our dataset
dataset = CatDogDataset(transform=normalize)

# loading the dataset and the model Resnet50 which is pretrained
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
model = models.resnet50(pretrained=True)

# fully connected layer:
model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias=True),
    nn.Sigmoid()
)

# running on gpu
model = model.cuda()

# finding the local minimum 
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # look @ image - the transitions btwn states are rotating in opp directions so it's going in a straight line
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)  # change the lr over time depending on needs of the optimizer
criterion = nn.BCELoss()  # loss function that you are trying to optimize

# training takes place:
for epoch in range(args.epochs):
    # training loop 
    pbar = tqdm(dataloader)  # progress bar
    avg_meters = AverageMeters()
    time_meters = AverageMeters()

    for idx, (img, label) in enumerate(pbar):  # converts a list into an index with values corresponding to that index
        end = time.time()
        img = img.cuda()
        label = label.cuda()
        label = label.reshape((label.shape[0], 1)).float()
        pred = model(img)
        loss = criterion(pred, label)  # compare predicted label vs actual label - based on that result, loss function will optimize it
        optimizer.zero_grad()  # optimizer resets gradients to 0. note: gradients correspond to each of the parameters
        
        # gradients are directional derivatives for the parameters...

        loss.backward()  # optimizer will take gradients and use backward pass to change the associated parameters
        optimizer.step()

        # logging metrics to assess the performance better:

        # numerical loss value
        loss = loss.item()
        time_meters.add_loss_value("batch_time", time.time() - end)
        avg_meters.add_loss_value("total_loss", loss)
        # pbar with loss
        pbar.set_postfix({ 
            'Batch Time': time_meters.average_meters['batch_time'].avg, 
            'Loss' : avg_meters.average_meters['total_loss'].avg,
            'Epoch': epoch
        })

    if epoch == args.epochs-1:
        torch.save(  # saves the model but serializing it into a binary form that can be read by pytorch later
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            },
            f"checkpoints/{epoch}.pth"
        )