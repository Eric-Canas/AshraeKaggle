from ASHRAEDataset import ASHRAEDataset
import torch
import Models as models
from torch.utils import data
import warnings
import math
from torch import nn
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from NetworkUtils import save_model, load_model, pearson_correlation
from torch.utils.data import DataLoader
import time
import os

def train_step(model, data, epoch, criterion, optimizer, device = 'cuda:0', verbose = False, writer=None, verbose_each=200):
    model.train()
    losses = []
    pearsons = []
    datalen = len(data)
    batch_time = time.time()
    for i, (x,y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        # print(i_batch)
        output = model(x)[...,0]
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pearson = pearson_correlation(x=y, y=output).item()
        pearsons.append(pearson)
        if verbose and i % verbose_each == 0:
            this_batch_time = time.time()-batch_time
            percent_done = i/datalen
            remaining_time = (this_batch_time*(datalen/verbose_each))*(1-percent_done)
            batch_time = time.time()
            print('Train -> Batch {0}/{1} ({2}%) of epoch {3}. Loss: {4}, Pearson: {5} --> Remaining Time: {6}:{7}'.
                  format(i,datalen, int((percent_done)*100), epoch, loss.item(), np.round(pearson,6),
                         int(remaining_time/60),int(remaining_time)%60))
    mean_losses = np.mean(np.array(losses))
    mean_pearsons = np.mean(np.array(pearsons))
    if writer is not None:
        writer.add_scalar('Train-Loss', mean_losses, epoch)
        writer.add_scalar('Train-Pearson', mean_pearsons, epoch)
    print('TRAIN -> EPOCH {0}, MEAN LOSS: {1}, MEAN PEARSON: {2}'.format(epoch, mean_losses, mean_pearsons))

def validation_step(model, data, epoch, criterion, device = 'cuda:0', writer=None):
    model.eval()
    losses = []
    pearsons = []
    with torch.no_grad():
        for i, (x,y) in enumerate(data):
            x = x.to(device)
            y = y.to(device)
            output = model(x)[...,0]
            loss = criterion(output, y)
            losses.append(loss.item())
            pearson = pearson_correlation(x=y, y=output).item()
            pearsons.append(pearson)
    mean_loss = np.mean(losses)
    mean_pearson = np.mean(np.array(pearsons))
    if writer is not None:
        writer.add_scalar('Validation-Loss', mean_loss, epoch)
        writer.add_scalar('Validation-Pearson', mean_pearson, epoch)
    print('VALIDATION -> EPOCH {0}, MEAN LOSS {1}, STD LOSS {2}, MEAN PEARSON {3}'.format(epoch, mean_loss, np.std(np.array(losses)), mean_pearson))

def train(lr=0.5,momentum = 0.9, gpu = 0, epochs = 500, file_name='DefaultFileName',
                 charge=None, save = True, batch_size = 20000, epochs_for_saving=20):

    #Stablishing the device
    device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn(message="Executing on CPU!", category=ResourceWarning)

    #Generating the model
    model = models.OneLayerRegressor()
    model = model.to(device)

    #Training Parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    #Generating the Dataset
    dataset = ASHRAEDataset()
    train_len, validation_len = int(math.ceil(0.96 * len(dataset))), int(math.ceil(0.03 * len(dataset)))
    train, validation, test = data.random_split(dataset, (train_len,
                                                          validation_len,
                                                          len(dataset)-train_len-validation_len))
    #Pass to Dataloader for reading batches
    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    validation = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    #Writer for plotting graphic in tensorboard
    writer = SummaryWriter(comment=file_name)
    print('Starting the training...')
    print("Batch Size: " + str(batch_size))
    print("Running in: " + device)

    for i in range(epochs):
        train_step(model=model, data=train, criterion=criterion, optimizer=optimizer, epoch=i, device=device,
                   writer=writer, verbose=True)
        validation_step(model=model, data=validation, criterion=criterion, epoch=i, device=device, writer=writer)
        if save and i%epochs_for_saving==0:
            model = model.cpu()
            save_model(model=model,file_name=file_name)
            model = model.to(device)
    writer.close()