import torch
import torch.nn as nn
# Pre-Activation 사용
class convolution_block(nn.Module):
    def __init__(self, input, out):
        super(convolution_block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input),
            nn.Mish(),
            nn.Conv2d(input, out, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(out),
            nn.Mish(),
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Convolution Block과 대칭되어야 해서 Post-Activation 사용
class upsampling(nn.Module):
    def __init__(self, input, out):
        super(upsampling, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input, out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out),
            nn.Mish()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, global_channels, local_channels, channels):
        super(AttentionBlock, self).__init__()

        # Global Attention에 대한 가중치 행렬 W_g
        self.global_conv = nn.Sequential(
            nn.Conv2d(global_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels)
        )

        # Local Attention에 대한 가중치 행렬 W_x
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels)
        )

        # 어텐션을 제어하는 파라미터 alpha
        self.alpha = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU 활성화 함수
        self.relu = nn.ReLU(inplace=True)

    def forward(self, global_feat_map, local_feat_map):
        # Global Attention과 Local Attention에 대한 각각의 가중치 행렬 계산
        global_out = self.global_conv(global_feat_map)
        local_out = self.local_conv(local_feat_map)

        # 두 결과를 더하고 ReLU 함수를 적용하여 어텐션을 제어하는 파라미터 alpha 계산
        alpha = self.relu(global_out + local_out)
        # alpha를 이용하여 Local Attention에 가중치를 적용하여 출력
        alpha = self.alpha(alpha)
        return local_feat_map * alpha


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = convolution_block(3, 64)
        self.Conv2 = convolution_block(64, 128)
        self.Conv3 = convolution_block(128, 256)
        self.Conv4 = convolution_block(256, 512)
        self.Conv5 = convolution_block(512, 1024)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        return x5, x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Upsampling1 = upsampling(1024, 512)
        self.Att1 = AttentionBlock(512, 512, 256)
        self.Conv1 = convolution_block(1024, 512)

        self.Upsampling2 = upsampling(512, 256)
        self.Att2 = AttentionBlock(256, 256, 128)
        self.Conv2 = convolution_block(512, 256)

        self.Upsampling3 = upsampling(256, 128)
        self.Att3 = AttentionBlock(128, 128, 64)
        self.Conv3 = convolution_block(256, 128)

        self.Upsampling4 = upsampling(128, 64)
        self.Att4 = AttentionBlock(64, 64, 32)
        self.Conv4 = convolution_block(128, 64)

        self.Conv = nn.Conv2d(64, 19+1, kernel_size=1, stride=1, padding=0)

    def forward(self, x5, x4, x3, x2, x1):
        d5 = self.Upsampling1(x5)
        x4 = self.Att1(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Conv1(d5)

        d4 = self.Upsampling2(d5)
        x3 = self.Att2(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Conv2(d4)

        d3 = self.Upsampling3(d4)
        x2 = self.Att3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Conv3(d3)

        d2 = self.Upsampling4(d3)
        x1 = self.Att4(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Conv4(d2)

        d1 = self.Conv(d2)

        return d1


class UNet_with_Attention(nn.Module):
    def __init__(self, input=3, classes=19+1):
        super(UNet_with_Attention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.encoder(x)
        output = self.decoder(x5, x4, x3, x2, x1)
        return output