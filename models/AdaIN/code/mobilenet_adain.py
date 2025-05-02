import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def adaptive_instance_normalization(content_feat, style_feat):
    B, C, H, W = content_feat.size()
    content_mean = content_feat.view(B, C, -1).mean(2).view(B, C, 1, 1)
    content_std = content_feat.view(B, C, -1).std(2).view(B, C, 1, 1)

    style_mean = style_feat.view(B, C, -1).mean(2).view(B, C, 1, 1)
    style_std = style_feat.view(B, C, -1).std(2).view(B, C, 1, 1)

    normalized = (content_feat - content_mean) / (content_std + 1e-5)
    return normalized * style_std + style_mean


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        self.encoder = nn.Sequential(*mobilenet[:14])  # Include up to Layer 13

    def forward(self, x):
        return self.encoder(x)  # Output: [B, 96, 14, 14]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2),  # [B, 96, 14, 14] → [B, 96, 28, 28]
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # [B, 64, 28, 28] → [B, 64, 56, 56]
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # [B, 32, 56, 56] → [B, 32, 112, 112]
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # [B, 16, 112, 112] → [B, 16, 224, 224]
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x):
        return self.decode(x)


class MobileNetAdaINNet(nn.Module):
    def __init__(self):
        super(MobileNetAdaINNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, content_img, style_img):
        content_feat = self.encoder(content_img)
        style_feat = self.encoder(style_img)
        t = adaptive_instance_normalization(content_feat, style_feat)
        stylized = self.decoder(t)
        return stylized
