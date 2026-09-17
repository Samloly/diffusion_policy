from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import tqdm
from torch.utils.data import DataLoader

class DiffusionPolicyTrainer:
    def __init__(self, 
        model, 
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        ema_power: float = 0.75,
        lr_scheduler_name: str = "cosine",
        lr_num_warmup_steps: int = 500,
        num_epochs: int = 100,
        dataloader: DataLoader | None = None,
        obs_horizon: int = 10,
        device: Optional[torch.device] = None,
        save_path: str = "best_model.pt"
        ):

        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.nets.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_model = EMAModel(self.model.nets.parameters(), power=ema_power)
        self.lr_scheduler = get_scheduler(
            lr_scheduler_name,
            self.optimizer,
            num_warmup_steps=lr_num_warmup_steps,
            num_training_steps=num_epochs * len(dataloader)
        )
        self.dataloader = dataloader
        self.obs_horizon = obs_horizon
        self.num_epochs = num_epochs
        self.device = device 
        self.save_path = Path(save_path)

    def train_step(self, batch: Dict[str, torch.Tensor], device: torch.device) -> float:
        self.optimizer.zero_grad()

        # data normalized in dataset
        # device transfer
        nimage = batch['image'][:,:self.obs_horizon].to(self.device) # (B, obs_horizon, C, H, W)
        nagent_pos = batch['agent_pos'][:,:self.obs_horizon].to(self.device) # (B, obs_horizon, 2)
        naction = batch['action'].to(self.device) # (B, pred_horizon, 2)
        B = nagent_pos.shape[0]

        obs_cond = self.model.get_global_cond(nimage, nagent_pos) # compute global condition for the batch

        # get noise and noisy actions
        noisy_actions, noise, timesteps = self.model.add_noise(naction)

        # predict the noise residual
        noise_pred = self.model.predict_noise(noisy_actions, timesteps, obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        self.ema_model.step(self.model.nets.parameters())

        return loss.item()

    def train(self):
        self.model.nets.to(self.device)
        self.model.nets.train()
        self.ema_model.to(self.device)
        best_loss = float('inf')
        acum_epoch_loss = 0
        epoch_loss = list()
        for epoch in tqdm.tqdm(range(self.num_epochs), desc='Epoch', leave=True):
            acum_epoch_loss = 0
            num_batches = len(self.dataloader)

            for nbatch in tqdm.tqdm(self.dataloader, desc='Batch', leave=False):
                loss = self.train_step(nbatch, self.device)
                acum_epoch_loss += loss # accumulate loss for the epoch
            avg_epoch_loss = acum_epoch_loss / num_batches
            epoch_loss.append(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # 1. Store the current training weights and copy EMA weights to the model
                self.ema_model.store(self.model.nets.parameters())
                self.ema_model.copy_to(self.model.nets.parameters())
                # 2. Save the model state dict (which now contains EMA weights)
                torch.save(self.model.nets.state_dict(), self.save_path)
                print(f" -> Saved new best EMA model checkpoint to {self.save_path}")
                # 3. Restore original training weights so AdamW can keep optimizing
                self.ema_model.restore(self.model.nets.parameters())

            if epoch % 50 == 0:
                # save the final model after training
                self.ema_model.store(self.model.nets.parameters())
                self.ema_model.copy_to(self.model.nets.parameters())
                checkpoint_model_path =  self.save_path.with_name(f"model_{epoch}.pt")
                torch.save(self.model.nets.state_dict(), checkpoint_model_path)
                torch.save(self.model.nets.state_dict(), self.save_path)
                
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {acum_epoch_loss:.4f}")

        # save the final model after training
        self.ema_model.store(self.model.nets.parameters())
        self.ema_model.copy_to(self.model.nets.parameters())
        final_model_path =  self.save_path.with_name("final_model.pt")
        torch.save(self.model.nets.state_dict(), final_model_path)
        torch.save(self.model.nets.state_dict(), self.save_path)
        # plot the loss history
        self.save_plot_loss(epoch_loss, self.save_path.with_name("loss_plot.png"))


    def save_plot_loss(self, loss_history: list, save_path: str):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        plt.figure(figsize=(10,5))
        plt.plot(loss_history)
        plt.title("Training Loss Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # save the plot
        plt.savefig(save_path)