import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.scale = 0.2

    def forward(self, x):
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        out = residual + self.scale * out
        return out


class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // 8),
            nn.PReLU(),
            nn.Linear(ch // 8, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Upsampling(nn.Module):
    def __init__(self, in_channel):
        super(Upsampling, self).__init__()
        self.upsam = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.upsam(x)


class FluorescenceSRUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        self.head = nn.Conv2d(1, base_ch, 3, padding=1)

        # Encoder
        self.enc1 = ResBlock(base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.enc2 = ResBlock(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResBlock(base_ch * 4)

        # Decoder
        self.up2 = Upsampling(base_ch * 4)
        self.dec2 = ResBlock(base_ch * 2)

        self.up1 = Upsampling(base_ch * 2)
        self.dec1 = ResBlock(base_ch)

        self.se = SEBlock(base_ch)
        self.tail = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x):
        inp = x

        x = self.head(x)

        e1 = self.enc1(x)  
        e1_down = self.down1(e1)  
        e2 = self.enc2(e1_down)  
        e2_down = self.down2(e2)  

        b = self.bottleneck(e2_down)  

        up2_out = self.up2(b) 
        d2_input = up2_out + e2 
        d2 = self.dec2(d2_input)

        up1_out = self.up1(d2) 
        d1_input = up1_out + e1 
        d1 = self.dec1(d1_input) 

        # attention mechanism and output
        d1 = self.se(d1)
        residual = self.tail(d1)

        return inp + residual


# discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    gen_net =  FluorescenceSRUNet()
    input_tensor = torch.ones([1, 1, 128, 128])
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        gen_net.to('cuda')
    output = gen_net(input_tensor)
    print(f"Output shape: {output.shape}")
