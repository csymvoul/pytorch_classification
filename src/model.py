from torch import nn 
import torch

class ClassificationModel(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_units: int, 
                 output_shape: int, 
                 dropout: bool) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(input_size, hidden_units) 
        if dropout:
            self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_units, output_shape)

    def forward(self, x): 
        if self.dropout: 
            x = self.layer_1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.layer_2(x)
        else:
            x = self.layer_1(x)
            x = self.relu(x)
            x = self.layer_2(x)
        return x

