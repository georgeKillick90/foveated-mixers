import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.spatial import KDTree
import warnings
### ------ Image Tokenizers ------ ###

class PatchSensor(nn.Module):
    """
    Module that samples patches from an input image where patch centres are defined by the user
    permitting arbitrary spatial distributions of patches. Patches are scaled according to local
    sampling density and a patch scaling parameter. The expectation is that the local distributions of patches
    are approximately uniform. e.g. a blue noise distribution.

    Args:
        tessellation (torch.tensor): the spatial centres of the patches of shape [N_patches, 2]
        patch_size (int): the spatial dimensions of a patch i.e. [int x int] grid of sampling kernels
        in_channels (int): the number of channels in the input image.
        out_channels (int): the number of output channels / dimensionality of the token embeddings.
        patch_scaling (float): spatial scaling factor applied to all patches to control the amount of overlap.
        pos_embedding (bool): optionally apply learned positional embeddings to each patch. Default False

    """
    def __init__(self, tessellation, patch_size, in_channels, out_channels, patch_scaling=0.5, pos_embedding=False):
        super().__init__()

        dists = KDTree(tessellation).query(tessellation, k=3)[0][:,1:3]
        dists = torch.tensor(dists).mean(dim=-1)

        x = torch.linspace(-1, 1, patch_size)
        y = torch.linspace(-1, 1, patch_size)
        grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)

        scaled_grids = dists[:,None,None,None] * grid[None,:] * patch_scaling

        shifted_grids = scaled_grids + tessellation[:,None,None,:]

        shifted_grids = rearrange(shifted_grids, 'patches patch_h patch_w coordinate -> (patch_h patch_w) patches coordinate')

        self.sampling_kernels = nn.Parameter(shifted_grids.float(), False)
        
        self.linear = nn.Linear(in_channels * patch_size**2, out_channels)
        
        self.norm = nn.LayerNorm(out_channels)

        if pos_embedding:
            self.pos_embedding = nn.Parameter(torch.rand(1, tessellation.shape[0], out_channels))
        else:
            self.pos_embedding = None

    def forward(self, x, fixations=None):
        """
        Samples the input image with the use provided patchs to tokenize the input image.
        Additionally allows for all patches to be translated across the image, mimicking
        eye movements.

        Args:
            x (torch.tensor): Input image to be tokenized, shape [batch, channels, height, width]
            fixations (torch.tensor): Optional (x,y) coordinates to translate all patches. Default None
        Returns:
            (torch.tensor): Tokenized representation of input image, shape [batch, num_patches, out_channels]
        """
        sampling_kernels = self.sampling_kernels.repeat(x.shape[0], 1, 1, 1)
        if fixations is not None:
            sampling_kernels = sampling_kernels + fixations[:,None,None,:]

        x = F.grid_sample(x, sampling_kernels)
        x = rearrange(x, 'batch channels patch_dim patches -> batch patches (channels patch_dim)')
        
        x = self.linear(x)
        x = self.norm(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
            
        return x 

class Patchify(nn.Module):
    """
    Module to transform images into patch tokens similar to Vision Transformers (ViT).
    Optionally applies learned positional embeddings to the tokens if pos_embedding=True.
    
    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels for each patch.
        pos_embedding (bool): Whether to apply learned positional embeddings to the tokens. Default is False.
    """
    def __init__(self, image_size, patch_size, in_channels, out_channels, pos_embedding=False):
        super().__init__()
        
        assert image_size % patch_size == 0, 'image_size should be divisible by patch_size'
        
        self.patch_conv = nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_channels)
        self.reshape = Rearrange('batch channels h w -> batch (h w) channels')
        
        self.num_patches = (image_size//patch_size)**2
        
        if pos_embedding:
            self.pos_embedding = nn.Parameter(torch.rand(1, self.num_patches, out_channels))
        else:
            self.pos_embedding = None
    
    def forward(self, x):
        """
        Forward pass of the Patchify module.
        
        Args:
            x (torch.Tensor): Input tensor. Shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor after patchification. Shape [batch, n_patches, out_channels].
        """
        x = self.patch_conv(x)
        x = self.reshape(x)
        x = self.norm(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        
        return x

### ------ Foveated Sampling Distributions ------ ###

def log_fibonacci(n_nodes, fovea_radius, verbose=False):
    
    """
    Function that generates a foveated sampling layout using the method described i
    "Foveation in The Era of Deep Learning" - BMVC2023. 

    Args:
        n_nodes (int): number of points/pixels in the layout.
        fovea_radius (float): radius of the fovea. accepts values between (0.0, 1.0)
        fovea_density (float): The percentage of n_nodes that lay within the fovea radius.
        auto_density (bool): Automatically finds a reasonable fovea_density by matching fovea
        sampling resolution to the sampling resolution immediately outside the fovea.
        verbose (bool): Prints the resolution of a uniform image with equivalent resolution of
        the fovea and same field of view.

    Returns:
        torch.tensor: a tensor of shape [n_nodes, 2], corresponding to the xy coordinate of each node.

    """
    ### Checks ###
    
    assert fovea_radius > 0.0 and fovea_radius < 1.0, 'fovea radius should be between (0.0 - 1.0) not inclusive'
    assert n_nodes > 5, 'n_nodes must be greater than 5'
    
    if n_nodes < 49:
        
        message ="""
        While 'n_nodes' accepts values greater than 5, we recommend using values greater than 49,
        as the sampling layout becomes ill defined below this threshold.
        """
        warnings.warn(message.strip(), UserWarning)
            
    ##############
    
    fovea_density = 3/n_nodes
    f_res = 1.0
    p_res = 0.0
    
    # automatically finding a good foveal sampling resolution for the given fovea radius
    while f_res > p_res:
        
        n_f_nodes = int(n_nodes * fovea_density)
        n_p_nodes = n_nodes - n_f_nodes

        f_nodes = np.sqrt(np.linspace(1, n_f_nodes, n_f_nodes) -0.5) * np.sqrt(n_f_nodes) 
        p_nodes = np.geomspace(n_f_nodes, n_f_nodes//fovea_radius, n_p_nodes) 
        
        fovea_density = fovea_density + 0.01
        f_res = f_nodes[-1] - f_nodes[-2]
        p_res = p_nodes[1] - p_nodes[0]

    ### prints some statistics / info about the sampling layout
    if verbose:
        fovea_area = len(f_nodes)
        full_res_area = (1/fovea_radius) ** 2 * fovea_area
        square_area = int(np.sqrt(np.pi) * full_res_area)
        print("Fovea Resolution equivalent to {res}x{res} pixel image".format(res=int(np.sqrt(square_area))))

    # concatenate fovea and periphery radial coordinates and normalize by max radius
    r = np.concatenate((f_nodes, p_nodes))
    r = r / r.max()
    
    # generate angular coordinates
    g_ratio = (np.sqrt(5)+1) / 2
    theta = np.pi * 2 * g_ratio * np.arange(1, n_nodes+1)

    # sorting for visualization and debugging but unecessary, keeping for posterity
    sort_idx = np.argsort(theta)
    theta = theta[sort_idx]
    r = r[sort_idx]

    # convert from polar to cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta) 
    out = np.stack((x, y), axis=-1)
    return torch.tensor(out).float().contiguous()

### ------ Predefined Tokenizers ------ ###

def foveated_tokenizer(n_patches, fovea_radius, patch_size, out_channels, **kwargs):
    tessellation = log_fibonacci(n_patches, fovea_radius)
    return PatchSensor(tessellation, patch_size, 3, out_channels,**kwargs)