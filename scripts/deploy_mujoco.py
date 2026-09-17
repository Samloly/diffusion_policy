import os
import argparse
import collections
import torch
from pathlib import Path
from PIL import Image
import torchvision

# LeRobot imports
from lerobot.datasets import LeRobotDataset

# Diffusion policy imports
from diffusion_policy.model.cnn_diffusion_policy import DiffusionPolicy
from diffusion_policy.envs.mujoco_env.y_env import SimpleEnv
from diffusion_policy.envs.mujoco_env.y_env2 import SimpleEnv2
from diffusion_policy.train.mujoco_dataloader import normalize, unnormalize, load_stats, compute_norm_stats

def main():
    # Load arguments
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on Mujoco")
    parser.add_argument("--model-file", type=str, default="best_mujoco_diffusion.pt", 
                        help="name of the model file to load placed under /results folder")
    parser.add_argument("--image-key", type=str, default="observation.image",
                        help="key for the image observations in the dataset")
    parser.add_argument("--xml-path", type=str, default="src/diffusion_policy/envs/asset/example_scene_y2.xml", 
                        help="path to the XML file for the Mujoco environment from the root of the repo")
    parser.add_argument("--mode", type=str, default="eval", choices=["train", "eval"],
                        help="mode of the model, 'train' or 'eval'") 
    parser.add_argument("--data", type=str, default="omy_pnp_language",
                        help="folder name of the dataset under /data folder")
    parser.add_argument("--norm-data", type=str, default="norm_stats.pt",
                        help="normalization data file name under /data folder")
    parser.add_argument("--repo-id", type=str, default="Jeongeun/omy_pnp_language",
                        help="repository ID of the dataset")
    parser.add_argument("--pred-horizon", type=int, default=16,
                        help="prediction horizon for the model")
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="observation horizon for the model")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="action horizon for the model")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="number maximum steps to run before resetting the environment")
    args = parser.parse_args() 

    print("=== Starting Diffusion Policy Evaluation on Mujoco ===")
    print("\nConfiguration:")
    for arg in vars(args):
        print(f"\t-{arg}: {getattr(args, arg)}")

    # torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_path = Path(__file__).parent.resolve()

    max_steps = args.max_steps
    mode = args.mode
    # Model horizons
    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon
    # Normalization stats
    # stats = load_stats(os.path.join(current_path, "../data", args.norm_data))

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

    dataset = LeRobotDataset(
        repo_id, 
        root=dataset_root, 
        delta_timestamps=delta_timestamps, 
    )

    # Extract dimensions
    action_dim = dataset.features["action"]["shape"][0] 
    lowdim_obs_dim = dataset.features["observation.state"]["shape"][0] 
    vision_feature_dim = 512 
    num_diffusion_iters = 100 

    # Create model
    model = DiffusionPolicy(
        obs_horizon=obs_horizon, 
        action_horizon=action_horizon, 
        pred_horizon=pred_horizon, 
        vision_feature_dim=vision_feature_dim, 
        lowdim_obs_dim=lowdim_obs_dim, 
        action_dim=action_dim, 
        num_diffusion_iters=num_diffusion_iters, 
        device=device, 
        mode=mode 
    )

    save_path = os.path.join(current_path, "../results", args.model_file) 

    # Load weights
    model.nets.load_state_dict(torch.load(save_path, map_location=device)) 
    
    # DDIMScheduler is used for inference, which is faster than DDPMScheduler used for training
    inference_steps = num_diffusion_iters // 10 
    model.noise_scheduler.set_timesteps(inference_steps) 

    model.nets.to(device) 
    model.nets.eval() 
    print("Model loaded successfully!") 

    # Initialize your Mujoco Environment 
    xml_path = os.path.join(current_path, "../", args.xml_path)
    print(f"Initializing Mujoco Environment with XML: {xml_path}")
    PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

    B = 1  # Batch size for inference
    step = 0  
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(maxlen=obs_horizon) 
    img_transform = torchvision.transforms.ToTensor()

    def get_normalized_obs():
        """Grab current state/image from env, building a normalized agent_pos."""
        state = PnPEnv.get_joint_state()[:6]  # Get the first 6 joint angles
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = img_transform(image)
 
        agent_pos = torch.tensor(state, dtype=torch.float32, device=device)
        agent_pos = normalize(agent_pos, stats["agent_pos"]["min"], stats["agent_pos"]["max"])
        # print(f" state={state}, norm_state={agent_pos.cpu().numpy()}")
        return {"image": image, "agent_pos": agent_pos.cpu()}

    while PnPEnv.env.is_viewer_alive():
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=20):
            # Check if the task is completed
            success = PnPEnv.check_success()
            if success:
                print('Success')
                PnPEnv.reset(seed=0)
                step = 0
                save_image = False
            if step >= max_steps:
                print(f"Reached max steps ({max_steps}). Resetting environment.")
                PnPEnv.reset(seed=0)
                step = 0
                save_image = False
            if step == 0:
                # Initialize the observation deque with the first obs_horizon observations
                for _ in range(obs_horizon):
                    obs_deque.append(get_normalized_obs())

            # retrieve the last obs_horizon number of observations
            images = torch.stack([x['image'] for x in obs_deque]).to(device)
            agent_poses = torch.stack([x['agent_pos'] for x in obs_deque]).to(device)

            # get global_cond (image features + low-dim obs)
            obs_cond = model.get_global_cond(images.unsqueeze(0), agent_poses.unsqueeze(0)) 

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, pred_horizon, action_dim), device=device) 
            action = noisy_action 

            for k in model.noise_scheduler.timesteps: 
                # predict noise
                noise_pred = model.nets['noise_pred_net'](
                    sample=action, 
                    timestep=k, 
                    global_cond=obs_cond 
                )
                # inverse diffusion step (remove noise)
                action = model.noise_scheduler.step(
                    model_output=noise_pred, 
                    timestep=k, 
                    sample=action 
                ).prev_sample 

            # action is in normalized [-1, 1] space; map back to real units
            # before executing in the environment
            pred_action = unnormalize(action, stats["action"]["min"], stats["action"]["max"])
            pred_action_np = pred_action.detach().to('cpu').numpy()[0]


            # only take action_horizon number of actions
            start = obs_horizon - 1 
            end = start + action_horizon 
            exec_action = pred_action_np[start:end, :]

            # execute action_horizon number of steps without replanning
            for act in exec_action:
                _ = PnPEnv.step(act)  # Execute the action in the environment
                PnPEnv.render()
                step += 1

                # Get the current state of the environment
                obs_deque.append(get_normalized_obs())

                # Check if the task is completed
                success = PnPEnv.check_success()
                if success:
                    print('Success')
                    break

if __name__ == "__main__":
    main() 