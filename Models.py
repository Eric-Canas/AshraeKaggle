import torch.nn as nn
import torch.nn.functional as F
from ASHRAEDataset import INPUT_LEN
class OneLayerRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN, hidden_size=INPUT_LEN*2, output_size=1):
        super(OneLayerRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)  # hidden layer
        self.output = nn.Linear(hidden_size, output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.output(x))  # linear output
        return x