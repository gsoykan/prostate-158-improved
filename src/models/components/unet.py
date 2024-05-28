from typing import Optional

import torch
from torch import nn, Tensor


class UNet(nn.Module):
    def __init__(self,
                 depth: int = 4,
                 img_channels: int = 1,
                 initial_feature_channels: int = 16,
                 output_channels: int = 1,
                 conv_padding: int = 1,  # when 0 we have exact same ops with the original paper
                 dropout_p: Optional[float] = None,
                 ):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv_padding = conv_padding
        self.depth = depth
        self.img_channels = img_channels
        self.initial_feature_channels = initial_feature_channels
        self.output_channels = output_channels

        self.first_conv_block = UNet.create_conv_block(img_channels,
                                                       initial_feature_channels,
                                                       padding=self.conv_padding)
        self.output_conv = nn.Conv2d(initial_feature_channels,
                                     output_channels,
                                     kernel_size=1)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p is not None else nn.Identity()

        # create down & up -sampling blocks...
        self.down_blocks = nn.ModuleDict()
        self.up_blocks = nn.ModuleDict()
        prev_channels = self.initial_feature_channels
        for i in range(self.depth):
            self.down_blocks[str(i)] = UNet.create_conv_block(in_channels=prev_channels,
                                                              out_channels=prev_channels * 2,
                                                              start_with_maxpool=True,
                                                              padding=self.conv_padding)
            prev_channels = prev_channels * 2
            self.up_blocks[str(i)] = UNetUp(in_channels=prev_channels,
                                            kernel_size=2,
                                            stride=2,
                                            conv_block_padding=self.conv_padding)

    def forward(self, x: Tensor):
        down_ready = self.first_conv_block(x)
        down_outputs = {-1: down_ready}

        prev_output = down_ready
        for i in range(self.depth):
            down_outputs[i] = self.down_blocks[str(i)](prev_output)
            prev_output = down_outputs[i]

        up_input = down_outputs[self.depth - 1]
        # Drop-out layers at the end of the contracting path perform further implicit data augmentation.
        up_input = self.dropout(up_input) if hasattr(self, 'dropout') else up_input
        for i in reversed(range(self.depth)):
            long_skip_input = down_outputs[i - 1]
            up_input = self.up_blocks[str(i)](up_input, long_skip_input)

        output = self.output_conv(up_input)
        return output

    @staticmethod
    def create_conv_block(in_channels: int,
                          out_channels: int,
                          kernel_size: int = 3,
                          padding: int = 1,
                          start_with_maxpool: bool = False) -> nn.Sequential:
        first_conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding, )
        second_conv = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding)
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) if start_with_maxpool else nn.Identity(),
            first_conv,
            nn.ReLU(inplace=True),
            second_conv,
            nn.ReLU(inplace=True))


class UNetUp(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 conv_block_padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.up_sampler = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=in_channels // 2,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             )
        self.conv_block = UNet.create_conv_block(in_channels=in_channels,
                                                 out_channels=in_channels // 2,
                                                 kernel_size=3,
                                                 padding=conv_block_padding)

    def forward(self,
                x: Tensor,
                long_skip: Tensor):
        up_sampled = self.up_sampler(x)

        # copy and crop
        _, _, h, w = up_sampled.size()
        h_diff = (long_skip.size(2) - h) // 2
        w_diff = (long_skip.size(3) - w) // 2
        cropped_long_skip = long_skip[:, :, h_diff:h + h_diff, w_diff:w + w_diff]
        # cat
        conv_input = torch.cat((cropped_long_skip, up_sampled), dim=1)
        output = self.conv_block(conv_input)

        return output


if __name__ == '__main__':
    unet = UNet(depth=4,
                img_channels=1,
                initial_feature_channels=16,
                output_channels=2,
                conv_padding=1)
    unet = unet.to(device='cuda')
    input = torch.randn((4, 1, 512, 512), device='cuda', )
    output = unet(input)
    print(output)
