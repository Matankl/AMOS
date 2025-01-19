import torch
from torch import nn
import torch.nn.functional as torchFuncs
import numpy as np


class Block(nn.Module):
    '''
    One block of U-Net.
    Contains two repeated 3x3 unpadded convolutions, each followed by a ReLU activation.
    '''

    def __init__(self, in_channel, out_channel, kernel_size):
        '''
        Initializes the Block class.
        Args:
            in_channel: Number of input channels.
            out_channel: Number of output channels.
            kernel_size: Size of the convolutional kernel.
        '''
        super().__init__()
        # First convolutional layer
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size)
        # Second convolutional layer
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size)
        # ReLU activation function
        self.relu = nn.ReLU()

        # Initialize weights for convolutional layers (Gaussian distribution)
        nn.init.normal_(self.conv_1.weight, mean=0.0, std=self.init_std(in_channel, kernel_size))
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=self.init_std(out_channel, kernel_size))

    @staticmethod
    def init_std(channels, kernel_size):
        '''
        Computes the standard deviation for weight initialization in the convolutional layers.
        Args:
            channels: Number of channels.
            kernel_size: Size of the convolutional kernel.
        Returns:
            Standard deviation for weight initialization.
        '''
        return 2.0 / np.sqrt(channels * kernel_size ** 2)

    def forward(self, x):
        '''
        Forward pass of the Block.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after two convolutional layers with ReLU activations.
        '''
        # print(x.shape, "before first conv")
        x = self.conv_1(x)
        x = self.relu(x)
        # print(x.shape,"after first conv")
        x = self.conv_2(x)
        x = self.relu(x)
        # print(x.shape,"after second conv (last)","\n")
        return x


class Encoder(nn.Module):
    '''
    Contractile part of U-Net, responsible for down sampling the input.
    '''

    def __init__(self, channels):
        '''
        Initializes the Encoder class.
        Args:
            channels: List of channel sizes for each layer in the encoder.
        '''
        super().__init__()
        # List to store blocks
        modules = []
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            # print(in_channel, out_channel)
            block = Block(in_channel=in_channel, out_channel=out_channel, kernel_size=3)
            modules.append(block)
        self.blocks = nn.ModuleList(modules)  # Store blocks in a ModuleList
        self.max_pol = nn.MaxPool2d(kernel_size=2, stride=None)  # Max pooling layer for down sampling
        self.feat_maps = []  # Store feature maps for concatenation with the decoder

    def forward(self, x):
        '''
        Forward pass of the Encoder.
        Args:
            x: Input tensor.
        Returns:
            Downsampled tensor after passing through the encoder.
        '''
        for layer_no, layer in enumerate(self.blocks):
            x = layer(x)  # Pass input through the block
            if not self.is_final_layer(layer_no):
                self.feat_maps.append(x)  # Store feature map
                x = self.max_pol(x)  # Apply max pooling
                # print("x after pull", x.shape)
        return x

    def is_final_layer(self, layer_no):
        '''
        Checks if the current layer is the final layer in the encoder.
        Args:
            layer_no: Index of the current layer.
        Returns:
            True if final layer, else False.
        '''
        return layer_no == len(self.blocks) - 1


class Decoder(nn.Module):
    '''
    Expansive part of U-Net, responsible for upsampling the input.
    '''

    def __init__(self, channels):
        '''
        Initializes the Decoder class.
        Args:
            channels: List of channel sizes for each layer in the decoder.
        '''
        super().__init__()
        up_convs = []
        blocks = []
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            # 2x2 up-convolution
            upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
            up_convs.append(upconv)
            # Block with two convolutions and ReLU activations
            block = Block(in_channel, out_channel, kernel_size=3)
            blocks.append(block)
        self.upconvs = nn.ModuleList(up_convs)  # Store up-convolutions in a ModuleList
        self.blocks = nn.ModuleList(blocks)  # Store blocks in a ModuleList

    def forward(self, x, encoded_feat_maps):
        '''
        Forward pass of the Decoder.
        Args:
            x: Input tensor.
            encoded_feat_maps: Feature maps from the encoder for concatenation.
        Returns:
            Upsampled tensor after passing through the decoder.
        '''
        for upconv, block in zip(self.upconvs, self.blocks):
            # print(x.shape, " this is before upconv")
            x = upconv(x)  # Apply up-convolution
            fts = encoded_feat_maps.pop()  # Get corresponding feature map from the encoder
            # print(fts.shape, x.shape)
            fts = self.crop(fts, x.shape[2], x.shape[3])  # Crop feature map to match size
            # print(fts.shape, x.shape, "after crop")
            x = torch.cat([x, fts], dim=1)  # Concatenate feature map with input
            x = block(x)  # Pass through block
            # print(x.shape, "this is after block")
        return x

    @staticmethod
    def crop(tnsr, new_H, new_W):
        '''
        Center crop an input tensor to a specified size.
        Args:
            tnsr: Input tensor.
            new_H: New height.
            new_W: New width.
        Returns:
            Cropped tensor.
        '''
        _, _, H, W = tnsr.size()  # Get original size
        x1 = int(round((H - new_H) / 2.))  # Compute top-left corner
        y1 = int(round((W - new_W) / 2.))
        x2 = x1 + new_H  # Compute bottom-right corner
        y2 = y1 + new_W
        return tnsr[:, :, x1:x2, y1:y2]


class Unet(nn.Module):
    '''
    U-Net model for image segmentation as described in
    "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    '''

    def __init__(self, channels, no_classes, output_size=None):
        '''
        Initializes the U-Net class.
        Args:
            channels: List of channel sizes for the encoder.
            no_classes: Number of output classes for segmentation.
            output_size: (Optional) Desired output size for the final segmentation map.
        '''
        super().__init__()
        self.output_size = output_size
        self.encoder = Encoder(channels)                                  # Initialize encoder
        dec_channels = list(reversed(channels[1:]))                       # Reverse channels for decoder
        self.decoder = Decoder(dec_channels)                              # Initialize decoder
        self.head = nn.Conv2d(in_channels=channels[1], out_channels=no_classes,
                              kernel_size=1)                              # Final layer for segmentation

    def forward(self, x):
        '''
        Forward pass of the U-Net.
        Args:
            x: Input tensor.
        Returns:
            Segmentation map.
        '''
        # print("befoer encoding")
        x = self.encoder(x)  # Pass input through encoder
        # print("before decoding")
        x = self.decoder(x, self.encoder.feat_maps)  # Pass through decoder
        # print("after decoding")
        x = self.head(x)  # Final layer
        # print("after segmentation")
        if self.output_size is not None:  # Retain dimensions if output size is specified
            x = torchFuncs.interpolate(x, self.output_size)
        return x
