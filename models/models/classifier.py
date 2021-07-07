import torch
import torch.nn as nn


    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(2048, 8631)
    def forward(self, x):
        return self.linear(x)
