import argparse
from pathlib import Path

import torch
import yaml
from lerobot.datasets import LeRobotDataset
from torch.utils.data import DataLoader

from diffusion_policy.model.cnn_diffusion_policy import DiffusionPolicy
from diffusion_policy.train.diffusion_trainer import DiffusionPolicyTrainer
from diffusion_policy.train.mujoco_dataloader import (
    DiffusionLeRobotDatasetWrapper,
    load_stats,
    compute_norm_stats,
)

def resolve_path(project_root: Path, path_value:str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

def load_yaml_config(config_path:Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def select_device(device_cfg:str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)

def build_delta_timestamps(image_key:str, obs_horizon:int,pred_horizon:int,fps:int)->dict:
    return {
        image_key:[(i-obs_horizon+1)/fps for i in range(obs_horizon)],
        "observation.state": [(i-obs_horizon+1)/fps for i in range(obs_horizon)],
        "action": [i/fps for i in range(pred_horizon)],
    }

def create_dataloader(policy_type: str, dataset, stats:dict, dataset_cfg:dict,dataloader_cfg:dict)->DataLoader:
    image_key = dataset_cfg["image_key"]
    prompt_key = dataset_cfg.get("prompt_key", "task")

    if policy_type =="diffusion":
        wrapped_dataset = DiffusionLeRobotDatasetWrapper(dataset, stats, image_key=image_key)
    
    num_workers = int(dataloader_cfg.get("num_workers",0))
    persistent_workers = bool(dataloader_cfg.get("presistent_workers", False)) and num_workers>0

    return DataLoader(
        wrapped_dataset,
        batch_size=int(dataloader_cfg.get("batch_size", 64)),
        num_workers=num_workers,
        shuffle=bool(dataloader_cfg.get("shuffle", True)),
        pin_memory=bool(dataloader_cfg.get("pin_memory",True)),
        persistent_workers=persistent_workers,
    )

def print_debug_info(dataloader: DataLoader, dataset) -> None:
    print("\n\n========== Dataloader Batch Info ==========")
    batch = next(iter(dataloader))
    for key, value in batch.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            print(f"{key:15s} shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key:15s} type={type(value)}")

    print("\n\n========== Dataset Info ==========")
    raw_item = dataset[0]
    for key, value in raw_item.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            print(f"{key:25s} shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key:25s} value={value}")

def train_diffusion(
    dataloader:DataLoader,
    dataset,
    policy_cfg:dict,
    device:torch.device,
    save_path: Path,
)->None:
    horizons = policy_cfg["horizons"]
    model_cfg = policy_cfg["model"]
    trainer_cfg = policy_cfg["trainer"]

    action_dim = dataset.features["action"]["shape"][0]
    lowdim_obs_dim = dataset.features["observation.state"]["shape"][0]

    model = DiffusionPolicy(
        obs_horizon=int(horizons["obs"]),
        action_horizon=int(horizons["action"]),
        pred_horizon=int(horizons["pred"]),
        vision_feature_dim=int(model_cfg.get("vision_feature_dim",512)),
        lowdim_obs_dim=lowdim_obs_dim,
        action_dim=action_dim,
        num_diffusion_iters=int(model_cfg.get("num_diffusion_iters",100)),
        device=device,
        mode="train",
    )

    trainer = DiffusionPolicyTrainer(
        model=model,
        lr=float(trainer_cfg.get("lr",1e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay",1e-6)),
        ema_power=float(trainer_cfg.get("ema_power",0.75)),
        lr_scheduler_name=trainer_cfg.get("lr_scheduler_name", "cosine"),
        lr_num_warmup_steps=int(trainer_cfg.get("lr_num_warmup_steps", 500)),
        num_epochs=int(trainer_cfg["num_epochs"]),
        dataloader=dataloader,
        obs_horizon=int(horizons["obs"]),
        device=device,
        save_path=str(save_path),
    )
    trainer.train()

def main()->None:
    parser = argparse.ArgumentParser(description="Unified Mujoco training entrypoint")
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=["diffusion", "vla"],
        help="policy to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mujoco_train.yaml",
        help="path to yaml config file",
    )
    args = parser.parse_args()

    current_path = Path(__file__).parent.resolve()
    project_root = current_path.parent

    config_path = resolve_path(project_root,args.config)
    config = load_yaml_config(config_path)

    global_cfg = config.get("global", {})
    dataset_cfg = config["dataset"]
    policy_cfg = config["policies"][args.policy_type]
    dataloader_cfg = config.get("dataloader", {})

    torch.manual_seed(int(global_cfg.get("seed", 42)))
    device = select_device(global_cfg.get("device", "auto"))
    debug = bool(global_cfg.get("debug", False))

    # stats_path = resolve_path(project_root, dataset_cfg["norm_data_path"])
    dataset_root = resolve_path(project_root, dataset_cfg["dataset_root"])
    save_path = resolve_path(project_root, policy_cfg["trainer"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # stats = load_stats(str(stats_path))

    temp_dataset = LeRobotDataset(dataset_cfg["repo_id"], root=str(dataset_root))
    fps = temp_dataset.fps
    stats = compute_norm_stats(temp_dataset)

    horizons = policy_cfg["horizons"]
    delta_timestamps = build_delta_timestamps(
        image_key=dataset_cfg["image_key"],
        obs_horizon=int(horizons["obs"]),
        pred_horizon=int(horizons["pred"]),
        fps=fps,
    )

    print(f"Dataset FPS: {fps}. Loading dataset with delta_timestamps: {delta_timestamps}")

    dataset = LeRobotDataset(
        dataset_cfg["repo_id"],
        root=str(dataset_root),
        delta_timestamps=delta_timestamps,
    )

    dataloader = create_dataloader(
        policy_type=args.policy_type,
        dataset=dataset,
        stats=stats,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
    )

    if debug:
        print_debug_info(dataloader, dataset)

    print(f"\nStarting training for policy type: {args.policy_type}")
    print(f"Using config: {config_path}")

    if args.policy_type == "diffusion":
        train_diffusion(
            dataloader=dataloader,
            dataset=dataset,
            policy_cfg=policy_cfg,
            device=device,
            save_path=save_path,
        )
    


if __name__ == "__main__":
    main()