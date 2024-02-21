import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
from einops.layers.torch import Rearrange
import numpy as np
from typing import Optional

### ------ Blocks and Layers ------ ###

class FeedForward(nn.Module):
    
    """ 
    Generic FeedForward inverse bottleneck for vision models, seen in
    MobileNets, ConvNeXt, Vision Transformers, Mixer Architectures
    and more. Performs a non-linear mixing of information across
    channels.
    
    Args:
        channels (int): Number of input and output channels.
        expansion_factor (int, optional): Expansion factor for the intermediate hidden layer. Default is 4.
        act (torch.nn.Module, optional): Activation function to be used. Default is torch.nn.GELU().
        init_value (float, optional): Initial value for gamma parameter. Default is 0.1.
    """
    def __init__(self, channels, expansion_factor=4, act=nn.GELU(), init_value=0.1):
        
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(channels)
        self.norm_2 = nn.LayerNorm(channels)
        
        # no bias in second linear layer because it is followed by layernorm
        self.linear_1 = nn.Linear(channels, channels * expansion_factor)
        self.linear_2 = nn.Linear(channels * expansion_factor, channels, bias=False)
        
        self.act = act
        self.gamma = nn.Parameter(torch.ones((channels)) * init_value)

    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape [Batch Size, N, Channels]

        Returns:
            torch.Tensor: Output tensor. Shape [Batch Size, N, Channels]
        """
        skip = x
        x = self.norm_1(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.norm_2(x)
        x = skip + (self.gamma * x)
    
        return x
        
class SpatialMix(nn.Module):
    
    """
    Module that mixes spatial features via an MLP as done in MLP-Mixer, ResMLP, and More.
    The module supports multiple heads similar to Multi-Head Attention (MHA).
    
    Args:
        in_dim (int): The input dimensionality.
        channels (int): The number of channels in the input tensor.
        heads (int): The number of heads for multi-head mixing. Default is 1.
        out_dim (int, optional): The output dimensionality. If not provided, defaults to in_dim.
        init_value (float): The initial value for scaling parameter gamma. Default is 0.1.
    """
    
    def __init__(self, in_dim, channels, heads=1, out_dim=None, init_value=0.1):
        super().__init__()
        
        assert channels%heads==0, "channels must be divisble by the number of heads"
        
        self.heads = heads
        
        self.norm_1 = nn.LayerNorm(channels)
        self.norm_2 = nn.LayerNorm(channels)
        
        if out_dim is None:
            out_dim = in_dim
        
        self.weights = nn.Parameter(torch.rand(heads, in_dim, out_dim), True)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, out_dim))
        
        self.gamma = nn.Parameter(torch.ones((channels)) * init_value)
    
    def forward(self, x):
        """ 
        Forward pass of the SpatialMix module.
        
        Args:
            x (torch.Tensor): Input tensor. Shape [Batch Size, N, Channels].
        
        Returns:
            torch.Tensor: Output tensor after spatial mixing. Shape [Batch Size, N, Channels].
        """
        
        skip = x
        x = self.norm_1(x)
        x = einops.rearrange(x, 'batch n (heads channels) -> batch heads channels n', heads=self.heads)
        x = torch.einsum('bhcn,hnm->bhcm', x, self.weights)
        x = x + self.bias
        x = einops.rearrange(x, 'batch heads channels n -> batch n (heads channels)')
        x = self.norm_2(x)
        x = skip + (self.gamma * x)
        
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention module. 
    Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

    Args:
        dim (int): Dimensionality of the input feature vectors.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
        dropout (float): Dropout probability.
        init_value (float): Layer scale init value.
    
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., init_value=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.gamma = nn.Parameter(torch.ones((dim)) * init_value)

    def forward(self, x):
        skip = x
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = skip + (out * self.gamma)
        return out

### ------ Model Constructors ------ ###

class ViT(nn.Module):
    
    """
    Module implementing a ViT style architecture for image classification. Not strictly
    following the original ViT architecture. Instead following ResMLP more closely but
    with self-attention instead of MLPs.
    
    Args:
        spatial_dim (int): The spatial dimensionality of the input images.
        width (int): The width of the network.
        depth (int): The depth of the network.
        n_classes (int): Number of classes for classification.
        init_value (float): Layer scale init value.
        tokenizer (nn.Module): Tokenizer module for input data. Default is nn.Identity().
    """
    
    def __init__(self, width, depth, n_heads, n_classes, init_value, tokenizer=nn.Identity()):
        super().__init__()
        
        self.tokenizer = tokenizer
        
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(Attention(width, n_heads, dim_head=width//n_heads, init_value=init_value),
                                  FeedForward(width))
            blocks.append(block)
        
        self.blocks = nn.Sequential(*blocks)
        
        self.fc = nn.Linear(width, n_classes)
    
    def forward(self, x):
        
        """
        Forward pass of the ResMLP module.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Class Predictions
        """
        
        x = self.tokenizer(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class ResMLP(nn.Module):
    
    """
    Module implementing the ResMLP architecture for image classification.
    
    Args:
        spatial_dim (int): The spatial dimensionality of the input images.
        width (int): The width of the network.
        depth (int): The depth of the network.
        n_classes (int): Number of classes for classification.
        init_value (float): Layer scale init value.
        tokenizer (nn.Module): Tokenizer module for input data. Default is nn.Identity().
    """
    
    def __init__(self, spatial_dim, width, depth, n_classes, init_value, tokenizer=nn.Identity()):
        super().__init__()
        
        self.tokenizer = tokenizer
        
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(SpatialMix(spatial_dim, width, init_value=init_value),
                                  FeedForward(width))
            blocks.append(block)
        
        self.blocks = nn.Sequential(*blocks)
        
        self.fc = nn.Linear(width, n_classes)
    
    def forward(self, x):
        
        """
        Forward pass of the ResMLP module.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Class Predictions
        """
        
        x = self.tokenizer(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

### ------ Localizer Constructors ------ ###

class Localizer(nn.Module):

    """
    Module that takes an input image, resizes it
    and outputs an (x, y) coordinate corresponding
    to a position in the image.
	
	Args:
		resize (tuple(int, int)): dimensions to resize the input image to before processing
		temperature (float): temperature of softmax applied to saliency map. Default 1.0
		pretrained (bool): Use pytorch imagenet pretrained weights for resnet18. Default False
    """

    def __init__(self, resize, temperature=1.0, pretrained=False, **kwargs):
        super().__init__()
        
        self.resize = resize
        
        assert self.resize[0] > 16 and self.resize[1] > 16, "resize dimensions must be greater than 16" 
        
        base = resnet18(pretrained)
        
        self.net = nn.Sequential(base.conv1,
                                 base.bn1,
                                 base.relu,
                                 base.maxpool,
                                 base.layer1,
                                 base.layer2,
                                 base.layer3)
        
        # temperature of the softmax
        self.temperature = temperature
        self.conv1x1 = nn.Conv2d(256, 1, 1)
        
        # make a grid of xy coordinates corresponding to pixel
        # coordinates, normalized between (-1, 1)
        
        out_H, out_W = self.compute_out_size()
        
        x = torch.linspace(-1, 1, out_W)
        y = torch.linspace(-1, 1, out_H)
        
        grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=0)
        
        self.register_buffer('grid', grid)
            
    def compute_out_size(self):
        """
        Function to compute the spatial dimensions
        of the output featuremaps for a given
        input size.
        """
        
        dummy_img = torch.zeros((1, 3) + self.resize)
        out_shape = self.net(dummy_img).shape[2:]
        
        return out_shape
    
    def forward(self, x):
        """
        Computes a saliency map from an input image using
        resnet18. A softargmax produces an (x,y) coordinate
        
        Args:
            x (torch.tensor): Input tensor of images of shape [batch, 3, height, width]
        
        Returns:
            (torch.tensor): Tensor of (x,y) coordinates of shape [batch, 2]
            
        """
        # Compute saliency map from input image
        x = F.interpolate(x, self.resize, mode='bilinear', antialias=True)
        x = self.net(x)
        x = self.conv1x1(x)
        
        # Computes softargmax to produce an (x,y) coordinate
        B, C, out_H, out_W = x.shape
        
        # as temperature decreases, softmax trends towards argmax
        x = F.softmax(x.view(B, C, -1) / self.temperature).view(B, 1, out_H, out_W)
        x = torch.sum(x * self.grid, dim=(2, 3))
        
        return x

### ------ Predefined Models ------ ###

def resmlp_s12(n_classes, **kwargs):
    return ResMLP(196, 384, 12, n_classes, 0.1, **kwargs)

def vit_s12(n_classes, **kwargs):
    """
    ViT model that is approximately equivalent to
    ResMLP-S12 differing in self-attention layer only.
    """
    return ViT(384, 12, 8, n_classes, 0.1, **kwargs)