from torch import nn 
import torch

class ClassificationModel(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_units: int, 
                 output_shape: int, 
                 dropout: bool) -> None:
        super().__init__()

        self.linear_input = nn.Linear(input_size, hidden_units) 
        if dropout:
            self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_units, hidden_units) 
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear_output = nn.Linear(hidden_units, output_shape)

    def forward(self, x): 
        if self.dropout: 
            x = self.linear_input(x)
            x = self.leaky_relu(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.linear_output(x)
            x = self.sigmoid(x)
        else:
            x = self.linear_input(x)
            x = self.relu(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.linear_output(x)
            x = self.sigmoid(x)
        return x

