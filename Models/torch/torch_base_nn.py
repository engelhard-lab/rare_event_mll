import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, layers, activation, preset_weights=None):
        super().__init__()
        self.flatten = nn.Flatten()
        sequence = []
        l = 1
        while l < len(layers) - 1:
            sequence.append(nn.Linear(layers[l-1], layers[l]))
            if activation == 'relu':
                sequence.append(nn.ReLU())
            l += 1
        sequence.append(nn.Linear(layers[l-1], layers[l]))
        sequence.append(nn.Sigmoid())
        self.forward_pass = nn.Sequential(*sequence)
        if preset_weights is not None:
            self.preset_weights = preset_weights
            self.preset_layer = 0
            self.n_steps = len(layers) - 1
            self.apply(self._init_weights)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.forward_pass(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.preset_layer < self.n_steps:
                module.weight.data = self.preset_weights[self.preset_layer]
                module.bias.data = self.preset_weights[self.preset_layer+1]
            else:
                module.weight.data = self.preset_weights[self.preset_layer][0, :].unsqueeze(0)
                module.bias.data = self.preset_weights[self.preset_layer+1][0].unsqueeze(0)
            self.preset_layer += 2


