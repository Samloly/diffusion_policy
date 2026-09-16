import math
import torch
import torch.nn as nn
from typing import Union

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Transform input of shape (B, dim, T) to output of shape (B, dim, T/2)
        """
        Conv1d has dim kernels of shape (dim, 3), each kernel is applied to a window
        of shape (dim, 3).  The result of every kernel is a single value, and each convolution 
        the scalar result of each kernel is stacked to form the output of shape (dim, T/2).  
        The stride of 2 means that the window is shifted by 2 for each convolution,
        so the output has half the length of the input.
        So one layer like this has the number of parameters equal to dim * dim * 3 + dim (for bias)
        It is like Conv2d but without the channel spatial dimension.
        """
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1) # kernel_size=3, stride=2, padding=1

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Transforms input of shape (B, dim, T) to output of shape (B, dim, 2*T)
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1) # kernel_size=4, stride=2, padding=1

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # GroupNorm: Splits channels into 8 groups and normalizes them per-sample 
            # with mean and variance computed over each group. This is more stable 
            # than BatchNorm for small batch sizes.
            nn.GroupNorm(n_groups, out_channels), 
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        # Turns input of shape (B, in_channels, T) to output of shape (B, out_channels, T)
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias 
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), # (B, cond_dim) -> (B, cond_dim)
            nn.Linear(cond_dim, cond_channels), # (B, cond_channels) -> (B, out_channels * 2)
            nn.Unflatten(-1, (-1, 1)) # (B, out_channels * 2) -> (B, out_channels * 2, 1) allows broadcasting over time dimension
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ] -> noisy actions
            cond : [ batch_size x cond_dim] -> global conditioning, usually (obs_horizon * obs_dim + diffusion_step_embed_dim)

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) # [B, in_channels, T] -> [B, out_channels, T]
        embed = self.cond_encoder(cond) # [B, cond_dim] -> [B, out_channels * 2, 1]

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1) # [B, out_channels * 2, 1] -> [B, 2, out_channels, 1] 
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        # Create a list of tuples containig in-out channel sizes for each level of the UNet
        # in_out = [(input_dim, down_dims[0]), (down_dims[0], down_dims[1]), (down_dims[1], down_dims[2])]
        # in_out = [(input_dim, 256), (256, 512), (512, 1024)]
        in_out = list(zip(all_dims[:-1], all_dims[1:])) 
        # The middle dimension is the last element of all_dims, 
        # which is the last down_dim (before the upsampling starts)
        mid_dim = all_dims[-1] # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D( # (B, mid_dim, T) -> (B, mid_dim, T)
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D( # (B, mid_dim, T) -> (B, mid_dim, T)
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out): # len(in_out) = 3
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D( # (B, dim_in, T) -> (B, dim_out, T)
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D( # (B, dim_out, T) -> (B, dim_out, T)
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity() # (B, dim_out, T) -> (B, dim_out, T/2) if not last else Identity
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D( # (B, dim_out*2, T) -> (B, dim_in, T) dim_out*2 because of concatenation with skip connection
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D( # (B, dim_in, T) -> (B, dim_in, T)
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity() # (B, dim_in, T) -> (B, dim_in, 2*T) if not last else Identity
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size), # (B, start_dim, T) -> (B, start_dim, T)
            nn.Conv1d(start_dim, input_dim, 1), # (B, start_dim, T) -> (B, input_dim, T)
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim) -> noisy actions
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim) -> global conditioning, usually obs_horizon * obs_dim
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature) # (B, dim_in, T) -> (B, dim_out, T)
            x = resnet2(x, global_feature) # (B, dim_out, T) -> (B, dim_out, T)
            h.append(x)
            x = downsample(x) # (B, dim_out, T) -> (B, dim_out, T/2) if not last else Identity

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature) # (B, mid_dim, T) -> (B, mid_dim, T)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1) # Concat along channel dimension, (B, dim_out*2, T) 
            x = resnet(x, global_feature) # (B, dim_out*2, T) -> (B, dim_in, T)
            x = resnet2(x, global_feature) # (B, dim_in, T) -> (B, dim_in, T)
            x = upsample(x) # (B, dim_in, T) -> (B, dim_in, 2*T) if not last else Identity

        x = self.final_conv(x) # (B, start_dim, T) -> (B, input_dim, T)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x