import torch.nn.functional as F
from torch import nn


class EfficientRCTCDecoder(nn.Module):
    """
    Uses 1x1 Channel Fusion and Height-wise Average Pooling.
    """
    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 bottleneck_channels=128,
                 return_feats=False,
                 **kwargs):
        super().__init__()

        # 1x1 Channel Fusion (Bottleneck) to reduce parameters in Classifier
        self.fusion = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=True
        )

        self.fc = nn.Linear(
            bottleneck_channels,
            out_channels,
            bias=True,
        )
        self.out_channels = out_channels
        self.return_feats = return_feats

    def forward(self, x, data=None):
        # x shape: [B, C, H, W] (e.g., [B, 256, 4, 32])

        # 1. 1x1 Channel Fusion
        x = self.fusion(x) # [B, 128, H, W]

        # 2. Height-wise Average Pooling (Efficient way to squash height for ALPR)
        # Instead of heavy Attention, we average features vertically.
        feats = x.mean(dim=2) # [B, C, W]
        feats = feats.permute(0, 2, 1) # [B, W, C]

        # 3. Final Classification
        predicts = self.fc(feats) # [B, W, out_channels]

        if self.return_feats:
            result = (feats, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
