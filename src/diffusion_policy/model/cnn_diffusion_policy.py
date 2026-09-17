from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .unet import ConditionalUnet1D


def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class DiffusionPolicy:
    def __init__(self,   
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 16,
        action_dim: int = 2,
        lowdim_obs_dim: int = 2,
        vision_feature_dim: int = 512,
        num_diffusion_iters: int = 100,
        device: Optional[torch.device] = None,
        mode: str = 'train'
        ):

        vision_encoder = get_resnet('resnet18')
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # ResNet18 has output dim of 512
        self.vision_feature_dim = vision_feature_dim
        # agent_pos is 2 dimensional
        self.lowdim_obs_dim = lowdim_obs_dim
        # observation feature has 514 dims in total per step
        self.obs_dim = self.vision_feature_dim + self.lowdim_obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.device = device

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )

        # Create nn.ModuleDict to hold the networks
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })
        if mode == 'train':
            # for this demo, we use DDPMScheduler with 100 diffusion iterations
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule='squaredcos_cap_v2',
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type='epsilon'
            )
        elif mode == 'eval':
            # for evaluation, we use DDIMScheduler to speed up inference
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon',
                set_alpha_to_one=True,      # DDIM-specific, affects final step
                steps_offset=0,
            )
    
    def add_noise(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the action according to the diffusion process.
        action: (B, pred_horizon, action_dim)
        return: noisy_action, noise, timesteps
        """
        # sample noise
        noise = torch.randn(action.shape, device=self.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (action.shape[0],), device=self.device
        ).long()

        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # this is the forward diffusion process
        noisy_actions = self.noise_scheduler.add_noise(
            action, noise, timesteps)
        return noisy_actions, noise, timesteps

    def extract_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from the image using the vision encoder.
        image: (B, obs_horizon, C, H, W)
        return: (B, obs_horizon, vision_feature_dim)
        """
        B = image.shape[0]
        # flatten the first two dims to feed into the vision encoder
        image = image.flatten(end_dim=1) # (B*obs_horizon, C, H, W)
        features = self.nets['vision_encoder'](image) # (B*obs_horizon, vision_feature_dim)
        features = features.reshape(B, self.obs_horizon, -1) # (B, obs_horizon, vision_feature_dim)
        return features

    def get_global_cond(self, image: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """
        Concatenate vision features and low-dim obs to get the observation features.
        Then flattens the obs_horizon dimension to get the global condition for the diffusion model.
        image: (B, obs_horizon, C, H, W)
        agent_pos: (B, obs_horizon, lowdim_obs_dim)
        return: (B, obs_horizon*obs_dim)
        """
        image_features = self.extract_image_features(image) # (B, obs_horizon, vision_feature_dim)
        obs_features = torch.cat([image_features, agent_pos], dim=-1) # (B, obs_horizon, obs_dim)
        global_cond = obs_features.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
        return global_cond

    def predict_noise(self, noisy_action: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise residual given the noisy action, timestep, and global condition.
        noisy_action: (B, pred_horizon, action_dim)
        timestep: (B,)
        global_cond: (B, obs_horizon*obs_dim)
        return: (B, pred_horizon, action_dim)
        """
        noise_pred = self.nets['noise_pred_net'](
            noisy_action, timestep, global_cond=global_cond)
        return noise_pred