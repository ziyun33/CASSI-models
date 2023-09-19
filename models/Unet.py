import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=ch_in, bias=bias)
        self.pwconv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pwconv(self.dwconv(x))
    

class PartialConv(nn.Module):
    def __init__(self, ch, n_div=4, bias=True) -> None:
        super().__init__()
        self.ch_in = ch
        self.ch_pconv = ch // n_div
        self.pconv = nn.Conv2d(in_channels=self.ch_pconv, out_channels=self.ch_pconv, kernel_size=3, stride=1, padding=1, bias=bias)
        self.pwconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.ch_pconv, self.ch_in-self.ch_pconv], dim=1)
        x1 = self.pconv(x1)
        x = self.pwconv(torch.cat((x1, x2), 1))

        return x
    

class ConvBlock(nn.Module):
    def __init__(self, ch, conv, bn=True) -> None:
        super().__init__()
        self.conv = conv
        if bn is True:
            self.bn = nn.BatchNorm2d(ch)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, convblocks, activate=nn.ReLU()) -> None:
        super().__init__()

        self.conv1, self.conv2 = convblocks
        self.act = activate

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.act(x + x2) if x.shape == x2.shape else self.act(x2)

        return out


class add_fusion(nn.Module):
    def __init__(self, ch=None) -> None:
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
    def __init__(self, 
                 ch, 
                 layer_num=5, 
                 updown=[1, 3], dwconv=[], pconv=[], 
                 bn=True,
                 step=2, 
                 fusion=add_fusion, 
                 activate=nn.ReLU()
                ):
        super().__init__()

        self.out_ch = ch
        self.fea_ch = 64
        self.step = step

        self.ini = self.initial_EPhi

        self.head = nn.Sequential(
            nn.Conv2d(self.out_ch, self.fea_ch, kernel_size=3, stride=1, padding=1),
            activate
        )

        self.tail = nn.Sequential(
            nn.Conv2d(self.fea_ch, self.out_ch, kernel_size=1),
            nn.Tanh()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.fea_ch*2**len(updown), self.fea_ch, kernel_size=1),
            activate,
            nn.Conv2d(self.fea_ch, self.fea_ch, kernel_size=3, stride=1, padding=1, groups=self.fea_ch if len(dwconv) > 0 else 1),
            activate,
            nn.Conv2d(self.fea_ch, self.fea_ch*2**len(updown), kernel_size=1),
            activate
        )

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.fusions = nn.ModuleList([])
        
        for i in range(layer_num):
            if i in updown:
                if i in dwconv:
                    conv1 = DSConv(ch_in=self.fea_ch, ch_out=self.fea_ch*2, kernel_size=3, stride=2, padding=1)
                    conv2 = DSConv(ch_in=self.fea_ch*2, ch_out=self.fea_ch*2)
                    conv3 = DSConv(ch_in=self.fea_ch*2, ch_out=self.fea_ch*2)
                    conv4 = nn.ConvTranspose2d(in_channels=self.fea_ch*2, out_channels=self.fea_ch, kernel_size=2, stride=2, padding=0)
                elif i in pconv:
                    conv1 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch*2, kernel_size=3, stride=2, padding=1)
                    conv2 = PartialConv(ch=self.fea_ch*2)
                    conv3 = PartialConv(ch=self.fea_ch*2)
                    conv4 = nn.ConvTranspose2d(in_channels=self.fea_ch*2, out_channels=self.fea_ch, kernel_size=2, stride=2, padding=0)
                else:
                    conv1 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch*2, kernel_size=3, stride=2, padding=1)
                    conv2 = nn.Conv2d(in_channels=self.fea_ch*2, out_channels=self.fea_ch*2, kernel_size=3, stride=1, padding=1)
                    conv3 = nn.Conv2d(in_channels=self.fea_ch*2, out_channels=self.fea_ch*2, kernel_size=3, stride=1, padding=1)
                    conv4 = nn.ConvTranspose2d(in_channels=self.fea_ch*2, out_channels=self.fea_ch, kernel_size=2, stride=2, padding=0)
                convblock1 = ConvBlock(ch=self.fea_ch*2, conv=conv1, bn=bn)
                convblock2 = ConvBlock(ch=self.fea_ch*2, conv=conv2, bn=bn)
                convblock3 = ConvBlock(ch=self.fea_ch*2, conv=conv3, bn=bn)
                convblock4 = ConvBlock(ch=self.fea_ch, conv=conv4, bn=bn)
                self.fea_ch = self.fea_ch * 2
            else:
                if i in dwconv:
                    conv1 = DSConv(ch_in=self.fea_ch, ch_out=self.fea_ch)
                    conv2 = DSConv(ch_in=self.fea_ch, ch_out=self.fea_ch)
                    conv3 = DSConv(ch_in=self.fea_ch, ch_out=self.fea_ch)
                    conv4 = DSConv(ch_in=self.fea_ch, ch_out=self.fea_ch)
                elif i in pconv:
                    conv1 = PartialConv(ch=self.fea_ch)
                    conv2 = PartialConv(ch=self.fea_ch)
                    conv3 = PartialConv(ch=self.fea_ch)
                    conv4 = PartialConv(ch=self.fea_ch)
                else:
                    conv1 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch, kernel_size=3, stride=1, padding=1)
                    conv2 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch, kernel_size=3, stride=1, padding=1)
                    conv3 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch, kernel_size=3, stride=1, padding=1)
                    conv4 = nn.Conv2d(in_channels=self.fea_ch, out_channels=self.fea_ch, kernel_size=3, stride=1, padding=1)
                convblock1 = ConvBlock(ch=self.fea_ch, conv=conv1, bn=bn)
                convblock2 = ConvBlock(ch=self.fea_ch, conv=conv2, bn=bn)
                convblock3 = ConvBlock(ch=self.fea_ch, conv=conv3, bn=bn)
                convblock4 = ConvBlock(ch=self.fea_ch, conv=conv4, bn=bn)
                    
            convblocks_en = (convblock1, convblock2)
            encoder_layer = ResBlock(convblocks=convblocks_en, activate=activate)
            self.encoder.append(encoder_layer)

            convblocks_de = (convblock3, convblock4)
            decoder_layer = ResBlock(convblocks=convblocks_de, activate=activate)
            self.decoder.insert(0, decoder_layer)

            fusion_layer = fusion(ch=self.fea_ch)
            self.fusions.insert(0, fusion_layer)

    def initial_H(self, y, mask=None):
        bs, row, col = y.shape
        x = torch.zeros(bs, self.out_ch, row, col-(self.out_ch-1)*self.step).to(y.device)
        for i in range(self.out_ch):
            x[:, i, :, :] = y[:, :, self.step * i:self.step * i + col - (self.out_ch - 1) * self.step]

        return x
    
    def initial_HM(self, y, mask):
        bs, row, col = y.shape
        x = torch.zeros(bs, self.out_ch, row, col-(self.out_ch-1)*self.step).to(y.device)
        for i in range(self.out_ch):
            x[:, i, :, :] = y[:, :, self.step * i:self.step * i + col - (self.out_ch - 1) * self.step]

        return torch.mul(x, mask)
    
    def initial_E(self, meas, shift_mask):
        # A method proposed in BIRNAT to get normalized measurement
        B, H, W = meas.shape
        C = self.out_ch
        step = self.step
        mask_s = torch.sum(shift_mask, 1) / C
        nor_meas = torch.div(meas, mask_s)
        x = torch.unsqueeze(nor_meas, dim=1).expand([B, C, H, W])

        output = torch.zeros(B, C, H, W - (C - 1) * self.step).to(meas.device)
        for i in range(C):
            output[:, i, :, :] = x[:, i, :, step * i:step * i + W - (C - 1) * step]

        return output
    
    def initial_EPhi(self, meas, shift_mask):
        # A method proposed in BIRNAT to get normalized measurement
        B, H, W = meas.shape
        C = self.out_ch
        step = self.step
        mask_s = torch.sum(shift_mask, 1) / C
        nor_meas = torch.div(meas, mask_s)
        x = torch.mul(torch.unsqueeze(nor_meas, dim=1).expand([B, C, H, W]), shift_mask)

        output = torch.zeros(B, C, H, W - (C - 1) * self.step).to(meas.device)
        for i in range(C):
            output[:, i, :, :] = x[:, i, :, step * i:step * i + W - (C - 1) * step]

        return output


    def forward(self, x, mask=None):
        x = self.ini(x, mask)
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