import torch
import os

def save_model(model, path='./Models', file_name='default_model.pth'):
    path = os.path.join(path, file_name+'.pth')
    torch.save(model.state_dict(), path)

def load_model(model, path='./AcceptableModels', file_name='default_model.pth'):
    path = os.path.join(path, file_name)
    model.load_state_dict(torch.load(path))
    return model

def pearson_correlation(x,y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val