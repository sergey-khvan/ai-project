import torch
import torch.nn as nn

VGG16 = [
    64, 64, 'M', 128, 128, 'M',
    256, 256, 256, 'M', 512, 512,
    512, 'M', 512, 512, 512, 'M'
    ]

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes = 1000):
        super().__init__()
        self.in_channels = in_channels
        self.conv_stack = self.create_conv(VGG16)
        self.fc_stack = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x,1)
        x = self.fc_stack(x)
        return x

    def create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for s in architecture:
            if isinstance(s,int):
                out_channels = s
                layers += [nn.Conv2d(in_channels, out_channels,
                                     kernel_size=3,stride=1,padding=1),
                           nn.BatchNorm2d(s),
                           nn.ReLU()
                           ]
                in_channels = s
            elif s == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
