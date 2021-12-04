import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn
from .classifier import ClassificationHead
from .registry import register

__all__ = [
    'MlpBlockBase', 'ChannelMlpBlock', 'SpatialMlpBlock', 'MixerBlock',
    'MlpMixer', 'mlpmixer_s16', 'mlpmixer_s32', 'mlpmixer_vs16', 'mlpmixer_vs32'
]


class MlpBlockBase(tnn.CondSeq):

    def __init__(self, outer_features, inner_features, block):
        super().__init__()
        self.inner_features = inner_features
        self.outer_features = outer_features
        self.linear1 = tu.kaiming(block(outer_features, inner_features))
        self.gelu = nn.GELU()
        self.linear2 = tu.normal_init(block(inner_features, outer_features))


class ChannelMlpBlock(MlpBlockBase):

    def __init__(self, outer_features, inner_features):
        super().__init__(outer_features, inner_features, nn.Linear)
        # (B, L, Cin) -> (B, L, Cout)


class SpatialMlpBlock(MlpBlockBase):

    def __init__(self, outer_features, inner_features):
        super().__init__(outer_features, inner_features,
                         lambda i, o: nn.Conv1d(i, o, 1))
        # (B, Lin, C) -> (B, Lout, C)


class MixerBlock(nn.Module):

    def __init__(self, seq_len, in_features, hidden_token_mix,
                 hidden_channel_mix):
        super().__init__()
        self.norm1 = nn.LayerNorm((seq_len, in_features))
        self.tokens_mlp = SpatialMlpBlock(seq_len, hidden_token_mix)
        self.norm2 = nn.LayerNorm((seq_len, in_features))
        self.channels_mlp = ChannelMlpBlock(in_features, hidden_channel_mix)

    def forward(self, x):
        x = x + self.tokens_mlp(self.norm1(x))
        x = x + self.channels_mlp(self.norm2(x))
        return x


class MlpMixer(tnn.CondSeq):

    def __init__(self, im_size, num_classes, patch_size, num_blocks, hidden_dim,
                 tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.patch_size = patch_size
        self.im_size = im_size
        features = tnn.CondSeq()
        features.input = tu.kaiming(
            nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size))
        seq_len = im_size // patch_size
        seq_len2 = seq_len * seq_len
        features.im_size = im_size
        features.reshape = tnn.Reshape(hidden_dim, seq_len2)
        features.permute = tnn.Lambda(lambda x: x.permute(0, 2, 1))

        for i in range(num_blocks):
            mlp = MixerBlock(seq_len2, hidden_dim, tokens_mlp_dim,
                             channels_mlp_dim)
            setattr(features, f'mlp_{i}', mlp)
        setattr(features, f'norm_{i}', nn.LayerNorm(hidden_dim))
        features.unpermute = tnn.Lambda(lambda x: x.permute(0, 2, 1))
        features.unshape = tnn.Reshape(hidden_dim, seq_len, seq_len)
        self.features = features
        self.classifier = ClassificationHead(hidden_dim, num_classes)

    def forward(self, x):
        assert (x.shape[2] == self.im_size and x.shape[3] == self.im_size), (
            f"input image of {self.__class__.__name__} must be of size "
            f"{self.im_size}x{self.im_size}")
        return super().forward(x)


@register
def mlpmixer_vs32(num_classes, im_size=224):
    return MlpMixer(im_size, num_classes, 16, 4, 512, 64, 512)


@register
def mlpmixer_vs16(num_classes, im_size=224):
    return MlpMixer(im_size, num_classes, 32, 4, 512, 64, 512)


@register
def mlpmixer_s32(num_classes, im_size=224):
    return MlpMixer(im_size, num_classes, 32, 8, 512, 256, 2048)


@register
def mlpmixer_s16(num_classes, im_size=224):
    return MlpMixer(im_size, num_classes, 16, 8, 512, 256, 2048)
