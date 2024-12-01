import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

HEIGHT_REDUCTION = 16
WIDTH_REDUCTION = 8


class DepthSepConv2D(nn.Module):
    """
    Depthwise Separable Convolutional 2D layer. It consists of a Depthwise Convolutional layer followed by a Pointwise Convolutional layer.
    The activation function can be applied after the Depthwise Convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int]): Kernel size of the Depthwise Convolutional layer.
        activation (Optional[nn.Module], optional): Activation function. Defaults to None.
        padding (bool, optional): If True, padding is applied. If padding is True, padding is calculated automatically. Defaults to True.
        stride (Union[int, Tuple[int, int]], optional): Stride of the Depthwise Convolutional layer. Defaults to (1, 1).
        dilation (Union[int, Tuple[int, int]], optional): Dilation of the Depthwise Convolutional layer. Defaults to (1, 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        activation: Optional[nn.Module] = None,
        padding: bool = True,
        stride: Union[int, Tuple[int, int]] = (1, 1),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
    ):
        super(DepthSepConv2D, self).__init__()
        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [
                        padding_h // 2,
                        padding_h - padding_h // 2,
                        padding_w // 2,
                        padding_w - padding_w // 2,
                    ]
                    padding = (0, 0)
        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=(1, 1),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)

        if self.padding:
            x = F.pad(x, self.padding)

        if self.activation:
            x = self.activation(x)

        x = self.point_conv(x)

        return x


class MixDropout(nn.Module):
    """
    MixDropout module that applies either 2D or 1D dropout with a probability of 0.5.

    Args:
        dropout_prob (float): Dropout probability for 1D dropout.
        dropout_2d_prob (float): Dropout probability for 2D dropout.
    """

    def __init__(self, dropout_prob: float = 0.4, dropout_2d_prob: float = 0.2):
        super(MixDropout, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2D(x)


class ConvBlock(nn.Module):
    """
    Convolutional Block that consists of 3 Conv2D layers.
    MixedDropout can be applied after Conv2D layers. It is randomly chosen after which layer the dropout is applied.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        stride (Union[int, Tuple[int, int]], optional): Stride of the last Conv2D layer. Defaults to (1, 1).
        kernel (Union[int, Tuple[int, int]], optional): Kernel size of the Conv2D layers. Defaults to 3. Exception: Last Conv2D layer always has kernel size (3, 3).
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU(inplace=True).
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: Union[int, Tuple[int, int]] = (1, 1),
        kernel: Union[int, Tuple[int, int]] = 3,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout: float = 0.5,
    ):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel,
            padding=kernel // 2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=kernel,
            padding=kernel // 2,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
        )
        self.normLayer = nn.InstanceNorm2d(
            num_features=out_c,
            eps=0.001,
            momentum=0.99,
            track_running_stats=False,
        )
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = random.randint(1, 3)

        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)

        return x


class DSCBlock(nn.Module):
    """
    Depthwise Separable Convolution Block (DSCBlock) that consists of 3 DepthSepConv2D layers (kernel size 3).
    MixedDropout can be applied after DepthSepConv2D layers. It is randomly chosen after which layer the dropout is applied.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        stride (Union[int, Tuple[int, int]], optional): Stride of the last DepthSepConv2D layer. Defaults to (2, 1).
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU(inplace=True).
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: Union[int, Tuple[int, int]] = (2, 1),
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout: float = 0.5,
    ):
        super(DSCBlock, self).__init__()
        self.activation = activation
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(
            out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )
        self.norm_layer = nn.InstanceNorm2d(
            out_c,
            eps=0.001,
            momentum=0.99,
            track_running_stats=False,
        )
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)

        return x


class Encoder(nn.Module):
    """
    Convolutional Encoder for the Transformer model.
    It consists of 5 ConvBlocks and 4 DSCBlocks.

    Args:
        in_channels (int): Number of input channels.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int, dropout: float = 0.5):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_c=in_channels, out_c=16, stride=(1, 1), dropout=dropout),
                ConvBlock(in_c=16, out_c=32, stride=(2, 2), dropout=dropout),
                ConvBlock(in_c=32, out_c=64, stride=(2, 2), dropout=dropout),
                ConvBlock(in_c=64, out_c=128, stride=(2, 2), dropout=dropout),
                ConvBlock(in_c=128, out_c=128, stride=(2, 1), dropout=dropout),
            ]
        )
        self.dscblocks = nn.ModuleList(
            [
                DSCBlock(in_c=128, out_c=128, stride=(1, 1), dropout=dropout),
                DSCBlock(in_c=128, out_c=128, stride=(1, 1), dropout=dropout),
                DSCBlock(in_c=128, out_c=128, stride=(1, 1), dropout=dropout),
                DSCBlock(in_c=128, out_c=256, stride=(1, 1), dropout=dropout),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        First, the input tensor is passed through the ConvBlocks.
        Then, the tensor is passed through the DSCBlocks. DSCBlocks output is added to the input tensor if the sizes match.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, out_channels, height, width).
        """

        for layer in self.conv_blocks:
            x = layer(x)

        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x
