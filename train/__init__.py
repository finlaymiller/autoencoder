""""""

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import os
from datetime import datetime
from tqdm import tqdm, trange

from utils.image import format_image, plot_images

from typing import List


class Trainer:
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        loader: DataLoader,
        device: torch.device,
        params,
    ) -> None:
        print(f"initializing model: '{model_name}'")

        self.model_name = model_name
        self.model = model
        self.train_dl = loader
        self.params = params
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

        if os.path.exists(self.params.log_dir):
            print("Logging directory already exists")
        else:
            print(f"Creating new log folder as {self.params.log_dir}")
            os.makedirs(self.params.log_dir)

        self.log_dir = os.path.join(
            self.params.log_dir,
            f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}{'_' if self.params.log_tag else ''}{self.params.log_tag}",
        )

        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"loaded model '{self.model_name}' with {total_parameters} parameters")

    def train(self) -> None:
        """Train the model"""

        print(f"training for {self.params.epochs} epochs")
        writer = SummaryWriter(self.log_dir)

        self.train_loss = []
        for epoch in trange(self.params.epochs, desc="Total"):
            running_loss = 0.0
            with tqdm(self.train_dl, unit="batch") as tepoch:
                for i, (names, images) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1:02d}")
                    # train step
                    images = images.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, images)
                    # backpropagation
                    loss.backward()
                    # update the parameters
                    self.optimizer.step()
                    # update loss tracking
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=f"{loss.item():03f}")
                    # log to tensorboard
                    global_step = epoch * len(self.train_dl) + i
                    writer.add_scalar("training/loss", loss.item(), global_step)
                    for p_name, param in self.model.named_parameters():
                        writer.add_histogram(
                            f"weights/{p_name}", param.data, global_step
                        )
                        if param.requires_grad:
                            writer.add_histogram(
                                f"gradients/{p_name}.grad", param.grad, global_step
                            )
                # track loss
                loss = running_loss / len(self.train_dl)
                self.train_loss.append(loss)

    # def test_reconstruction(self, num_tests: int = 1):
    #     for names, data in self.train_dl:
    #         img_noisy = data + self.params.noise_factor * torch.randn(data.shape)
    #         # img_noisy = np.clip(img_noisy, 0.0, 1.0)
    #         img_noisy = img_noisy.to(self.device)

    #         outputs = self.model(img_noisy)
    #         og_filename = names[0][: names[0].rfind("_")] + ".mid"

    #         images = [
    #             format_image(clean_dataset[og_filename]),
    #             img_noisy[0].cpu().data,
    #             outputs[0].cpu().data,
    #         ]
    #         titles = [
    #             f"{names[0]} (epochs={self.params.epochs})",
    #             f"noisy ({self.params.noise_factor}% noise)",
    #             f"reconstructed (loss={loss:.03f})",
    #         ]

    #         plot_images(images, titles)
    #         break
