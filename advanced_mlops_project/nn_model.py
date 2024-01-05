import torch
import torch.nn as nn
from torchsummary import summary


class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


class FullyConnectedNeuralNetwork:
    def __init__(self, size_h, size_w, embedding_size, num_classes):
        self.model = nn.Sequential()
        self.model.add_module("flatten", Flatten())
        self.model.add_module("dense1", nn.Linear(3 * size_h * size_w, 256))
        self.model.add_module("dense1_relu", nn.ReLU())
        self.model.add_module("dropout1", nn.Dropout(0.1))
        self.model.add_module("dense3", nn.Linear(256, embedding_size))
        self.model.add_module("dense3_relu", nn.ReLU())
        self.model.add_module("dropout3", nn.Dropout(0.1))
        self.model.add_module("dense4_logits", nn.Linear(embedding_size, num_classes))

    def show_model(self):
        # print model summary
        summary(self.model, (3, self.model.size_h, self.model.size_w), device="cpu")
        pass
