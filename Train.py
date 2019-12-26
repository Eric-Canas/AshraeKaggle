from Dataset import Dataset
import torch
from torch import functional as F
from torch.utils import data
import warnings
import math
from torch import nn
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from NetworkUtils import save_model, load_model
from torch.utils.data import DataLoader
import os

def train_step(model, data, epoch, criterion, optimizer, device = 'cuda:0', verbose = False, writer=None):
    model.train()
    losses = []
    for i, (x,y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        # print(i_batch)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and i % 50 == 0:
            print('Train -> Batch {0} of epoch {1}. Loss: {2}'.format(i, epoch, loss.item()))
    mean = np.mean(np.array(losses))
    if writer is not None:
        writer.add_scalar('Train-Loss', mean, epoch)
    print('TRAIN -> EPOCH {0}, MEAN LOSS: {1}'.format(epoch, mean))

def validation_step(model, data, epoch, criterion, device = 'cuda:0', writer=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x,y) in enumerate(data):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            losses.append(loss.item())
    mean = np.mean(losses)
    if writer is not None:
        writer.add_scalar('Validation-Loss', mean, epoch)
    losses = np.array(losses)
    print('VALIDATION -> EPOCH {0}, MEAN LOSS {1}, STD LOSS {2}'.format(epoch, mean, np.std(losses)))

def train(lr=0.001,momentum = 0.9, gpu = 0, epochs = 500, file_name='DefaultFileName',
                 charge=None, save = True, batch_size = 50):

    device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn(message="Executing on CPU!", category=ResourceWarning)
    #TODO: Define the model
    model = None

    print("Batch Size: "+str(batch_size))
    print("Running in: "+device)

    writer = SummaryWriter(comment=file_name)
    dataset = Dataset()
    train_len, validation_len = int(math.ceil(0.96 * len(dataset))), int(math.ceil(0.03 * len(dataset)))
    train, validation, test = data.random_split(dataset, (train_len,
                                                          validation_len,
                                                          len(dataset)-train_len-validation_len))
    #WHY IT IS NOT WORKING? TwT
    #train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4),
    #validation = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)



    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr, momentum=momentum)#
    for i in range(epochs):
        train_step(model=model, data=train, criterion=criterion, optimizer=optimizer, epoch=i, device=device,
                   writer=writer)
        validation_step(model=model, data=validation, criterion=criterion, epoch=i, device=device, writer=writer)
        if save and i%20==0:
            model = model.cpu()
            save_model(model=model,file_name=file_name)
            model = model.to(device)
    writer.close()