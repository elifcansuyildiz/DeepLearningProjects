import torch.nn as nn

class CNN(nn.Module):
    """
    Variation of LeNet: a simple CNN model
    for handwritten digit recognition
    """
    def __init__(self):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        conv1 = nn.Conv2d(in_channels=3, out_channels=16,  kernel_size=5, stride=1, padding=0)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.layer1 = nn.Sequential(
                conv1, relu1, maxpool1
            )
      
        # layer 2
        conv2 = nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=5, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.layer2 = nn.Sequential(
                conv2, relu2, maxpool2
            )
        
        # fully connected classifier
        in_dim = 32 * 5 * 5
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
        
        return
        
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2_flat = out2.view(cur_b_size, -1)
        y = self.fc(out2_flat)
        return y

class CNN_with_dropout(nn.Module):
    """ 
    Variation of LeNet: a simple CNN model
    for handwritten digit recognition
    """
    def __init__(self):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        drop1 = nn.Dropout2d(p=0.2)
        conv1 = nn.Conv2d(in_channels=3, out_channels=32,  kernel_size=5, stride=1, padding=0)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.layer1 = nn.Sequential(
                drop1, conv1, relu1, maxpool1
            )
      
        # layer 2
        drop2 = nn.Dropout2d(p=0.5)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=5, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.layer2 = nn.Sequential(
                drop2, conv2, relu2, maxpool2
            )
        
        # fully connected classifier
        in_dim = 64 * 5 * 5
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
        
        return
        
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2_flat = out2.view(cur_b_size, -1)
        y = self.fc(out2_flat)
        return y

class CNN_with_poolings(nn.Module):
    """
    Varation of LeNet: a simple CNN model
    for handwritten digit recognition
    """
    def __init__(self, pooling_layers):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        conv1 = nn.Conv2d(in_channels=3, out_channels=16,  kernel_size=5, stride=1, padding=0)
        relu1 = nn.ReLU()
        maxpool1 = pooling_layers[0]
        self.layer1 = nn.Sequential(
                conv1, relu1, maxpool1
            )
      
        # layer 2
        conv2 = nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=5, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool2 = pooling_layers[1]
        self.layer2 = nn.Sequential(
                conv2, relu2, maxpool2
            )
        
        # fully connected classifier
        in_dim = 32 * 5 * 5
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
        
        return
        
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2_flat = out2.view(cur_b_size, -1)
        y = self.fc(out2_flat)
        return y

class CNN_for_weight_visualization(nn.Module):
    """
    Varation of LeNet: a simple CNN model
    for handwritten digit recognition
    """
    def __init__(self):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,  kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
      
        # layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # fully connected classifier
        in_dim = 32 * 5 * 5
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
        
        return
        
    def forward(self, x, save_layer_outputs=False):
        """ Forward pass """
        cur_b_size = x.shape[0]
        
        x = self.conv1(x)
        if save_layer_outputs:
            self.conv1_out = x.detach().cpu().numpy()
        x = self.relu1(x)
        if save_layer_outputs:
            self.relu1_out = x.detach().cpu().numpy()
        x = self.maxpool1(x)
        if save_layer_outputs:
            self.maxpool1_out = x.detach().cpu().numpy()
        
        x = self.conv2(x)
        if save_layer_outputs:
            self.conv2_out = x.detach().cpu().numpy()
        x = self.relu2(x)
        if save_layer_outputs:
            self.relu2_out = x.detach().cpu().numpy()
        x = self.maxpool2(x)
        if save_layer_outputs:
            self.maxpool2_out = x.detach().cpu().numpy()

        x_f = x.view(cur_b_size, -1)
        y = self.fc(x_f)
        return y