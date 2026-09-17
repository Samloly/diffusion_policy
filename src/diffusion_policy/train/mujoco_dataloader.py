import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor

def compute_norm_stats(lerobot_dataset):
    dataset_stats = lerobot_dataset.meta.stats
    state_stats = dataset_stats["observation.state"]
    action_stats = dataset_stats["action"]

    stats = {
        "agent_pos": {
            "min": torch.as_tensor(
                state_stats["min"],
                dtype=torch.float32,
            ),
            "max": torch.as_tensor(
                state_stats["max"],
                dtype=torch.float32,
            ),
        },
        "action": {
            "min": torch.as_tensor(
                action_stats["min"],
                dtype=torch.float32,
            ),
            "max": torch.as_tensor(
                action_stats["max"],
                dtype=torch.float32,
            ),
        },
    }

    return stats

def load_stats(path, device=None):
    stats = torch.load(path, map_loaction=device)
    if device is not None:
        for key in stats:
            stats[key]["min"] = stats[key]["min"].to(device)
            stats[key]["max"] = stats[key]["max"].to(device)
    return stats

def normalize(x, lo, hi, eps=1e-8):
    """
    Min-max normalize x from [lo, hi] to [-1, 1].
    lo/hi are per-dimension tensors broadcastable against the last dim of x.
    """
    lo = lo.to(device=x.device, dtype=x.dtype)
    hi = hi.to(device=x.device, dtype=x.dtype)
    x = (x - lo) / (hi - lo + eps)   # -> [0, 1]
    return x * 2.0 - 1.0             # -> [-1, 1]

def unnormalize(x, lo, hi):
    """
    Inverse of normalize(): maps x from [-1, 1] back to [lo, hi].
    """
    lo = lo.to(device=x.device, dtype=x.dtype)
    hi = hi.to(device=x.device, dtype=x.dtype)
    x = (x + 1.0) / 2.0              # -> [0, 1]
    return x * (hi - lo) + lo        # -> [lo, hi]

class DiffusionLeRobotDatasetWrapper(Dataset):
    """
    Wraps a LeRobot dataset to:
      1. Output the plain dict format expected by DiffusionPolicy / the trainer.
      2. Normalize agent_pos (observation.state) and action to [-1, 1] using
         precomputed stats. Images are separately scaled to [0, 1].

    NOTE: `stats` must be the same object/values used at deploy time to
    un-normalize predicted actions, or the policy's outputs will be
    interpreted on the wrong scale.
    """
    def __init__(self, lerobot_dataset, stats, image_key="observation.image"):
        self.dataset = lerobot_dataset
        self.image_key = image_key
        self.stats = stats

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # LeRobot returns shape (T, C, H, W) for images and (T, D) for 1D arrays
        image = item[self.image_key].float()
        if image.max()>1.0:
            image = image/255.0

        agent_pos = item["observation.state"].float()
        action = item["action"].float()

        agent_pos = normalize(agent_pos,self.stats["agent_pos"]["min"],self.stats["agent_pos"]["max"])
        action = normalize(action, self.stats["action"]["min"], self.stats["action"]["max"])

        return {
            "image":image,
            "agent_pos":agent_pos,
            "action": action,
        }
    
class VLALeRobotDatasetWrapper(Dataset):
    """
    Wraps a LeRobotDataset for Pi0 VLA model training:
      1. Slices single current observation frame (T_obs = 1).
      2. Tokenizes prompt with SmolLM2 tokenizer.
      3. Processes images for SigLIP (224x224, normalized).
      4. Normalizes robot state and action chunk to [-1, 1].
    """
    def __init__(
        self,
        lerobot_dataset,
        stats: dict,
        smollm_name: str = "HuggingFaceTB/SmolLM2-135M",
        siglip_name: str = "google/siglip-base-patch16-224",
        prompt_key: str = "tasks", # or fixed prompt string
        image_key: str = "observation.image",
        max_prompt_length: int = 32,
    ):
        self.dataset = lerobot_dataset
        self.stats = stats
        self.prompt_key = prompt_key
        self.image_key = image_key

        # Load Tokenizer & Processor
        self.tokenizer = AutoTokenizer.from_pretrained(smollm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.image_processor = AutoProcessor.from_pretrained(siglip_name)
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # 1. Process Text Prompt
        # Handle either string column in dataset or fallback fixed prompt
        if isinstance(self.prompt_key, str) and self.prompt_key in item:
            prompt_text = item[self.prompt_key]
        elif isinstance(self.prompt_key, str) and self.prompt_key not in item:
            prompt_text = self.prompt_key  # Direct text string provided
            print(f"Warning: prompt_key '{self.prompt_key}' not found in dataset. Using fixed prompt string instead.")
        else:
            raise ValueError(f"Invalid prompt_key: {self.prompt_key}. Must be a string column in dataset or a fixed prompt string.")

        text_inputs = self.tokenizer(
            prompt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)          # (S,)
        attention_mask = text_inputs["attention_mask"].squeeze(0) # (S,)

        # 2. Extract Observations (T_obs = 1)
        # LeRobot observation shapes can be (T_obs, C, H, W) or (C, H, W)
        raw_image = item[self.image_key]
        if raw_image.ndim == 4:
            raw_image = raw_image[-1]  # Take the current frame (latest time step)

        # Apply SigLIP Image Processing (resize to 224x224 & normalize)
        if isinstance(raw_image, torch.Tensor):
            raw_image = raw_image.cpu().numpy()
        
        pixel_values = self.image_processor(
            images=raw_image, 
            return_tensors="pt"
        )["pixel_values"].squeeze(0)  # (3, 224, 224)

        # 3. Robot State (Current frame)
        raw_state = item["observation.state"].float()
        if raw_state.ndim == 2:
            raw_state = raw_state[-1]  # (7,)
            
        norm_state = normalize(
            raw_state, 
            self.stats["agent_pos"]["min"], 
            self.stats["agent_pos"]["max"]
        )

        # 4. Action Chunk (Target trajectory sequence H)
        # item["action"] shape is (H, action_dim)
        raw_action_chunk = item["action"].float()
        norm_action_chunk = normalize(
            raw_action_chunk, 
            self.stats["action"]["min"], 
            self.stats["action"]["max"]
        )

        return {
            "pixel_values": pixel_values,       # (3, 224, 224)
            "input_ids": input_ids,             # (S,)
            "attention_mask": attention_mask,   # (S,)
            "robot_state": norm_state,          # (7,)
            "action_chunk": norm_action_chunk,  # (H, 7)
        }