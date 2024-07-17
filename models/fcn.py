import torch.nn as nn
import torch.nn.functional as F


# class conv(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.conv = nn.Conv2d(**kwargs)
#
#     def forward(self, x):
#         return F.leaky_relu(self.conv(x))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        ]
        if padding is None:
            layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        else:
            layers.extend([
                nn.ZeroPad2d(padding),
                nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=0),
            ])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FCN(nn.Module):
    def __init__(self, nrof_classes, in_channels=1):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2, padding=0)
        )

        self.block1 = Block(in_channels=32, out_channels=64)
        self.block2 = Block(in_channels=64, out_channels=128, padding=(1, 0, 0, 0))
        self.block3 = Block(in_channels=128, out_channels=256, padding=(1, 0, 0, 0))

        self.output_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2, 3), stride=1),  # padding=(0, 1)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 5), stride=1),  # padding=(0, 2)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(in_channels=512, out_channels=nrof_classes, kernel_size=(1, 7), stride=1)  # padding=(0, 3)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.output_block(x)
        return out.squeeze(2)


if __name__ == '__main__':
    from torchsummary import summary

    model = FCN(33)
    # print(model)
    w = 500
    print(summary(model, (1, 32, w)))
