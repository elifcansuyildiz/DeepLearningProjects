import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        return x