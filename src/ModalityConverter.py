from torch import nn


class ModalityConverter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModalityConverter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


class ConvolutionalModalityConverter(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvolutionalModalityConverter, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x
