import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2

        return out

class add_fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return x + y

class conv_fusion(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=ch*2, out_channels=ch, kernel_size=1)
    
    def forward(self, x, y):
        z = torch.cat([x, y], 1)
        z = self.conv(z)

        return z

class Unet(nn.Module):
    def __init__(self, ch, step=2, fusion=add_fusion):
        super().__init__()

        self.out_ch = ch
        self.fea_ch = 64
        self.step = step

        self.head = nn.Sequential(
            nn.Conv2d(self.out_ch, self.fea_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.tail = nn.Sequential(
            nn.Conv2d(self.fea_ch, self.out_ch, kernel_size=1),
            nn.Tanh()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.fea_ch, self.fea_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.fea_ch, self.fea_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        layer_num = 5
        encoder_layers = [ResBlock(self.fea_ch, self.fea_ch) for i in range(layer_num)]
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = [ResBlock(self.fea_ch, self.fea_ch) for i in range(layer_num)]
        self.decoder = nn.ModuleList(decoder_layers)

        fusion_layers = [fusion(self.fea_ch) for i in range(layer_num)]
        self.fusions = nn.ModuleList(fusion_layers)

    def initial_x(self, y):
        bs, row, col = y.shape
        x = torch.zeros(bs, self.out_ch, row, row).to(y.device)
        for i in range(self.out_ch):
            x[:, i, :, :] = y[:, :, self.step * i:self.step * i + col - (self.out_ch - 1) * self.step]

        return x

    def forward(self, x, mask=None):
        x = self.initial_x(x)
        x1 = self.head(x)
        xs = []
        for layer in self.encoder:
            x1 = layer(x1)
            xs.append(x1)
        x2 = self.bottleneck(x1)
        for layer, fusion in zip(self.decoder, self.fusions):
            x2 = layer(fusion(xs.pop(), x2))
        x3 = self.tail(x2)
        out = x + x3
        return out