import torch
import os

def save_model(model, path='./models', file_name='default_model.pth'):
    path = os.path.join(path, file_name)
    torch.save(model.state_dict(), path)

def load_model(model, path='./models', file_name='default_model.pth'):
    path = os.path.join(path, file_name)
    model.load_state_dict(torch.load(path))
    return model
