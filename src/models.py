import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
from einops.layers.torch import Rearrange
import numpy as np
from typing import Optional
from torchvision.models import resnet18

### ------ Blocks and Layers ------ ###

class FeedForward(nn.Module):
    
    ''' 
    Generic FeedForward inverse bottleneck for vision models, seen in
    MobileNets, ConvNeXt, Vision Transformers, Mixer Architectures
    and more. Performs a non-linear mixing of information across
    channels.
    
    Args:
        channels (int): Number of input and output channels.
        expansion_factor (int, optional): Expansion factor for the intermediate hidden layer. Default is 4.
        act (torch.nn.Module, optional): Activation function to be used. Default is torch.nn.GELU().
    '''
    def __init__(self, channels, expansion_factor=4, act=nn.GELU()):
        
        super().__init__()
                
        self.norm = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, channels * expansion_factor)
        self.linear_2 = nn.Linear(channels * expansion_factor, channels)
        self.act = act
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape [Batch Size, N, Channels]

        Returns:
            torch.Tensor: Output tensor. Shape [Batch Size, N, Channels]
        """
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        
        return x
        
class SpatialMix(nn.Module):
    
    """
    Module that mixes spatial features via a linear layer as done in MLP-Mixer, ResMLP, and More.
    The module supports multiple heads similar to Multi-Head Attention (MHA).
    
    Args:
        n_patches (int): The input dimensionality.
        dim (int): The number of channels in the input tensor.
    """
    
    def __init__(self, n_patches, dim):
        super().__init__()
                        
        self.norm = nn.LayerNorm(dim)
        
        self.linear = nn.Linear(n_patches, n_patches)
            
    def forward(self, x):
        """ 
        Forward pass of the SpatialMix module.
        
        Args:
            x (torch.Tensor): Input tensor. Shape [Batch Size, N, Channels].
        
        Returns:
            torch.Tensor: Output tensor after spatial mixing. Shape [Batch Size, N, Channels].
        """

        x = self.norm(x)
        x = einops.rearrange(x, 'batch n channels -> batch channels n')
        x = self.linear(x)
        x = einops.rearrange(x, 'batch channels n -> batch n channels')
        
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
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
        
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Block(nn.Module):
    def __init__(self, n_patches, dim, init_value=0.1):
        super().__init__()

        self.layerscale_1 = nn.Parameter(init_value * torch.ones((dim)))
        self.layerscale_2 = nn.Parameter(init_value * torch.ones((dim)))

        self.spatial_mix = SpatialMix(n_patches, dim)
        self.channel_mix = FeedForward(dim)

    def forward(self, x):
        x = x + (self.layerscale_1 * self.spatial_mix(x))
        x = x + (self.layerscale_2 * self.channel_mix(x))
        return x

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
        n_heads (int): Number of heads in the self attention layer
        n_classes (int): Number of classes for classification.
        init_value (float): Layer scale init value.
    """
    
    def __init__(self, spatial_dim, width, depth, n_heads, n_classes, init_value):

        super().__init__()
                
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(Attention(width, n_heads, dim_head=width//n_heads, init_value=init_value),
                                  FeedForward(width))
            blocks.append(block)
        
        self.blocks = nn.Sequential(*blocks)
        
        self.fc = nn.Linear(width, n_classes)

        self.pos_embedding = nn.Parameter(torch.zeros((1, spatial_dim+1, width)))
        self.cls_token = nn.Parameter(torch.rand(width))
    
    def forward(self, x):
        
        """
        Forward pass of the ResMLP module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch,  spatial_dim, width]
        
        Returns:
            torch.Tensor: Class Predictions
        """

        # Token preparation
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_token), dim=1)
        x = x + self.pos_embedding

        # NN
        x = self.blocks(x)

        # get class token and predict
        x = x[:,-1,:]
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
    """
    
    def __init__(self, spatial_dim, width, depth, n_classes, init_value=1.0):

        super().__init__()
                
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(Block(spatial_dim, width, init_value))
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
        
        base = resnet18(weights='default') if pretrained else resnet18()
        
        self.net = nn.Sequential(base.conv1,
                                 base.bn1,
                                 base.relu,
                                 base.maxpool,
                                 base.layer1,
                                 base.layer2,
                                 base.layer3)
        
        # temperature of the softmax
        self.temperature = temperature
        self.saliency = nn.Conv2d(256, 1, 1)
        
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
        
        with torch.no_grad():
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
        x = self.saliency(x)
        
        # Computes softargmax to produce an (x,y) coordinate
        B, C, out_H, out_W = x.shape
        
        # as temperature decreases, softmax trends towards argmax
        x = F.softmax(x.view(B, C, -1) / self.temperature, dim=-1).view(B, 1, out_H, out_W)
        x = torch.sum(x * self.grid, dim=(2, 3))
        
        return x

### ------ Spatial Transformer ------ ###

class STN(nn.Module):
    """
    Wrapper module that takes a localizer, tokenizer and classifier.
    The module uses the localizer to produce an (x,y) coordinate from
    the input image, that is used to centre the tokenizer's FoV on the
    most salient region in the image. The tokenizer tokenizes the 
    image and passes it to the classifier which makes a class prediction.

    Args:
        localizer (nn.Module): localization network
        tokenizer (nn.Module): tokenizer where forward(image, (x,y))
        classifier (nn.Module): classifier network that operates on tokenized images

    """
    def __init__(self, localizer, tokenizer, classifier):
        super().__init__()

        self.localizer = localizer
        self.tokenizer = tokenizer
        self.classifier = classifier

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.tensor): Input image

        Returns:
            torch.tensor: class predictions
        """

        fixation = self.localizer(x)
        x = self.tokenizer(x, fixation)
        x = self.classifier(x)
        return x

### ------ Predefined Models ------ ###

def resmlp_s12(n_classes, **kwargs):
    return ResMLP(196, 384, 12, n_classes, **kwargs)

def vit_s12(n_classes, **kwargs):
    """
    ViT model that is approximately equivalent to
    ResMLP-S12 differing in self-attention layer only.
    """
    return ViT(196, 384, 12, 8, n_classes, 0.1, **kwargs)
