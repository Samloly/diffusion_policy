import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from diffusion_policy.model.cnn_diffusion_policy import DiffusionPolicy
from diffusion_policy.train.mujoco_dataloader import normalize, unnormalize,load_stats,compute_norm_stats

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Yields dataset frame indices belonging to a single episode, in order.
 
    Different lerobot versions expose episode boundaries differently:
      - older (v2.x): dataset.episode_data_index["from"/"to"]
      - newer (v3.x): dataset.meta.episodes["dataset_from_index"/"dataset_to_index"]
    This tries both so the script works regardless of installed version.
    """
    def __init__(self, dataset, episode_index):
        if hasattr(dataset, "episode_data_index"):
            from_idx = dataset.episode_data_index["from"][episode_index].item()
            to_idx = dataset.episode_data_index["to"][episode_index].item()
        elif hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
            episodes = dataset.meta.episodes
            from_idx = int(episodes["dataset_from_index"][episode_index])
            to_idx = int(episodes["dataset_to_index"][episode_index])
        else:
            raise AttributeError(
                "Could not find episode boundary metadata on this LeRobotDataset "
                "(checked dataset.episode_data_index and dataset.meta.episodes). "
                "Run `print(dir(dataset))` / `print(dir(dataset.meta))` and let me know "
                "what episode-boundary attribute is available in your installed lerobot version."
            )
        self.frame_ids = list(range(from_idx, to_idx))

    def __len__(self):
        return len(self.frame_ids)
    
    def __iter__(self):
        return iter(self.frame_ids)
    
def predict_action_chunk(model, image, agent_pos, stats, pred_horizon,action_dim,device):
    """
    Runs the full reverse-diffusion sampling loop and returns an UNNORMALIZED
    (B, pred_horizon, action_dim) action chunk.

    image: (B, obs_horizon, C, H, W), pixel values in [0, 1]
    agent_pos: (B, obs_horizon, D), RAW (not yet normalized) — normalized inside
    """

    image =image.to(device)
    agent_pos = normalize(agent_pos.to(device), stats["agent_pos"]["min"],stats["agent_pos"]["max"])
    obs_cond = model.get_global_cond(image, agent_pos)

    B = image.shape[0]
    action = torch.randn((B,pred_horizon,action_dim),device=device)

    # iteratively denoise, feeding each step's output into the next
    # (this loop must match the fixed version used in deploy_mujoco.py)
    for k in model.noise_scheduler.timesteps:
        noise_pred = model.nets['noise_pred_net'](sample=action, timestep=k,global_cond=obs_cond)
        action = model.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=action).prev_sample

    return unnormalize(action, stats["action"]["min"], stats["action"]["max"])

def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on Mujoco")
    parser.add_argument("--model-file", type=str, default="best_mujoco_diffusion.pt", 
                        help="name of the model file to load placed under /results folder")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="index of the episode to evaluate (0-indexed)") 
    parser.add_argument("--data", type=str, default="omy_pnp_language",
                        help="folder name of the dataset under /data folder")
    # parser.add_argument("--norm-data", type=str, default="norm_stats.pt",
    #                     help="normalization data file name under /data folder")
    parser.add_argument("--repo-id", type=str, default="Jeongeun/omy_pnp_language",
                        help="repository ID of the dataset")
    parser.add_argument("--image-key", type=str, default="observation.image",
                        help="key for image observations in the dataset")
    parser.add_argument("--pred-horizon", type=int, default=16,
                        help="prediction horizon for the model")
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="observation horizon for the model")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="action horizon for the model")
    args = parser.parse_args() 

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_path = Path(__file__).parent.resolve()

    # Model horizons
    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon

    # --- Dataset Setup (for both stats and training dataloader) ---
    image_key = args.image_key
    repo_id = args.repo_id
    dataset_root = os.path.join(current_path, "../data", args.data)
    temp_dataset = LeRobotDataset(repo_id, root=dataset_root)  # Load the dataset to inspect metadata
    fps = temp_dataset.fps

    stats = compute_norm_stats(temp_dataset)
    # Convert integer frame indices into time offsets in seconds
    delta_timestamps = {
        image_key: [(i - obs_horizon + 1) / fps for i in range(obs_horizon)],
        "observation.state": [(i - obs_horizon + 1) / fps for i in range(obs_horizon)],
        "action": [i / fps for i in range(pred_horizon)]
    }

    print(f"Dataset FPS: {fps}. Loading dataset with delta_timestamps: {delta_timestamps}")

    # Re-initialize the dataset WITH the correct delta_timestamps argument
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )

    action_dim = dataset.features["action"]["shape"][0]
    lowdim_obs_dim = dataset.features["observation.state"]["shape"][0]
    vision_feature_dim = 512
    num_diffusion_iters = 100
    
    # --- Model ---
    model = DiffusionPolicy(
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        pred_horizon=pred_horizon,
        vision_feature_dim=vision_feature_dim,
        lowdim_obs_dim=lowdim_obs_dim,
        action_dim=action_dim,
        num_diffusion_iters=num_diffusion_iters,
        device=device,
        mode="eval",
    )

    save_path = os.path.join(current_path, "../results", args.model_file)
    model.nets.load_state_dict(torch.load(save_path, map_location=device))
    inference_steps = num_diffusion_iters//10
    model.noise_scheduler.set_timesteps(inference_steps)
    model.nets.to(device)
    model.nets.eval()
    print("Model loaded successfully!")

    def load_image(raw_image):
        image = raw_image.float()
        if image.max()>1.0:
            image = image/255.0
        return image
    
    sampler =EpisodeSampler(dataset, args.episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=sampler,
    )

    preds, gts =[], []
    with torch.no_grad():
        for batch in test_dataloader:
            image= load_image(batch[image_key]) # (1, obs_horizon, C, H, W)
            agent_pos = batch["observation.state"].float() #(1,obs_horizon,D)
            gt_action = batch["action"][:, 0, :].to(device)

            pred_chunk = predict_action_chunk(
                model,image,agent_pos,stats, pred_horizon,action_dim,device
            )
            pred_action = pred_chunk[:,0,:]
            preds.append(pred_action)
            gts.append(gt_action)
    pred_actions = torch.cat(preds, dim=0)
    gt_actions = torch.cat(gts,dim=0)

    mean_error = torch.mean(torch.abs(pred_actions- gt_actions)).item()
    print(f"[episode mode] Episode {args.episode_index}, {len(sampler)} steps. Mean action error: {mean_error:.4f}")

    pred_np = pred_actions.detach().cpu().numpy()
    gt_np = gt_actions.detach().cpu().numpy()
    title = f"Episode {args.episode_index} — pred vs gt action over time"
    xlabel = "timestep"

    # --- plot ---
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
    if action_dim == 1:
        axs = [axs]
    for i in range(action_dim):
        axs[i].plot(pred_np[:, i], label="pred")
        axs[i].plot(gt_np[:, i], label="gt")
        axs[i].set_ylabel(f"dim {i}")
        axs[i].legend(loc="upper right")
    axs[0].set_title(title)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()

    out_path = os.path.join(current_path, f"../results/validation_ep{args.episode_index}_plot.png")
    plt.savefig(out_path)
    print(f"Saved validation plot to {out_path}")


if __name__ == "__main__":
    main()