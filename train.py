import torch
from torch._C import ThroughputBenchmark
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MobileNetSkipAdd
from utils import MyDataset


def calculateError(outputs, labels):
    diffMatrix = torch.abs(outputs-labels.data)
    nElement = torch.sum(outputs==labels.data)+torch.sum(outputs!=labels.data)

    MSE = torch.sum(torch.pow(diffMatrix, 2)) / nElement
    RMSE = torch.sqrt(MSE)
    MAE = torch.sum(diffMatrix)

    return MSE, RMSE, MAE


def train(model, writer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare dataset
    train_set = MyDataset("./dataset/train/origin/", "./dataset/train/depth/")
    test_set = MyDataset("./dataset/test/origin/", "./dataset/test/depth/")

    train_loader = DataLoader(train_set, batch_size=8, 
                              shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=2, 
                              shuffle=False, drop_last=True)

    dataloaders = {'train': train_loader, 'validation': test_loader}

    model = model.to(device)
    
    loss_fn = nn.L1Loss()
    learning_rate = 1e-3

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1400], gamma=0.5, last_epoch=-1)

    num_epochs = 1500

    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            running_loss = 0.0
            running_mse = 0.0
            running_rmse = 0.0
            running_mae = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.permute(0,3,1,2).float()
                labels = labels.permute(0,3,1,2).float()
                # print(inputs.shape)
                # print(labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels) # output=(batch_size, tag_num), labels=(batch_size, 1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # _, y_preds = torch.max(outputs, 1)
                running_loss += loss.detach()*inputs.size(0)
                # running_corrects += (torch.sum(outputs==labels.data)) / (torch.sum(outputs==labels.data)+torch.sum(outputs!=labels.data))
                # mse, rmse, mae = calculateError(outputs, labels)
                # running_mse += mse
                # running_rmse += rmse
                # running_mae += mae
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_mse = running_mse.float() / len(dataloaders[phase].dataset)
            # epoch_rmse = running_rmse.float() / len(dataloaders[phase].dataset)
            # epoch_mae = running_mae.float() / len(dataloaders[phase].dataset)

            scheduler.step()

            writer.add_scalar(phase+'/loss', epoch_loss.item(), epoch)
            # writer.add_scalar(phase+'/mse', epoch_mse.item(), epoch)
            # writer.add_scalar(phase+'/rmse', epoch_rmse.item(), epoch)
            # writer.add_scalar(phase+'/mae', epoch_mae.item(), epoch)
            
        if (epoch+1)%50==0:
            torch.save(model, './model/'+str(epoch)+'.pkl')


if __name__ == '__main__':
    net = MobileNetSkipAdd(output_size=(480,480), pretrained=False)
    writer = SummaryWriter('log')
    train(net, writer)
    writer.close()