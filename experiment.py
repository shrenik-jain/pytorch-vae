import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    # def training_step(self, batch, batch_idx, optimizer_idx = 0):
    #     real_img, labels = batch
    #     self.curr_device = real_img.device

    #     results = self.forward(real_img, labels = labels)
    #     train_loss = self.model.loss_function(*results,
    #                                           M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
    #                                           optimizer_idx=optimizer_idx,
    #                                           batch_idx = batch_idx)

    #     self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

    #     return train_loss['loss']

    # def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    #     real_img, labels = batch
    #     self.curr_device = real_img.device

    #     results = self.forward(real_img, labels = labels)
    #     val_loss = self.model.loss_function(*results,
    #                                         M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
    #                                         optimizer_idx = optimizer_idx,
    #                                         batch_idx = batch_idx)

    #     self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def training_step(self, batch, batch_idx):  
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'],
                                            batch_idx = batch_idx)  

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx): 
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                        M_N = 1.0,
                                        batch_idx = batch_idx)  # Remove optimizer_idx here

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        self.evaluate_metrics()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    # def configure_optimizers(self):

    #     optims = []
    #     scheds = []

    #     optimizer = optim.Adam(self.model.parameters(),
    #                            lr=self.params['LR'],
    #                            weight_decay=self.params['weight_decay'])
    #     optims.append(optimizer)
    #     # Check if more than 1 optimizer is required (Used for adversarial training)
    #     try:
    #         if self.params['LR_2'] is not None:
    #             optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
    #                                     lr=self.params['LR_2'])
    #             optims.append(optimizer2)
    #     except:
    #         pass

    #     try:
    #         if self.params['scheduler_gamma'] is not None:
    #             scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
    #                                                          gamma = self.params['scheduler_gamma'])
    #             scheds.append(scheduler)

    #             # Check if another scheduler is required for the second optimizer
    #             try:
    #                 if self.params['scheduler_gamma_2'] is not None:
    #                     scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
    #                                                                   gamma = self.params['scheduler_gamma_2'])
    #                     scheds.append(scheduler2)
    #             except:
    #                 pass
    #             return optims, scheds
    #     except:
    #         return optims

    def configure_optimizers(self):
        patience = self.params.get('patience', 10)  
        factor = self.params.get('factor', 0.5)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.params['LR'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            patience=patience,
                                                            factor=factor)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss"
        }
        }

    def evaluate_metrics(self):
        """Calculate Inception Score and FID for generated images"""
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        
        # Initialize metrics
        fid = FrechetInceptionDistance(feature=2048)
        inception_score = InceptionScore(feature=2048)
        
        # Move to the same device as the model
        fid = fid.to(self.curr_device)
        inception_score = inception_score.to(self.curr_device)
        
        # Get real samples from the test set
        batch_size = 50
        real_samples = []
        labels = []
        
        # Collect real samples
        for i, (images, lbls) in enumerate(self.trainer.datamodule.test_dataloader()):
            real_samples.append(images)
            labels.append(lbls)
            if i >= 5:  # Limit number of batches for efficiency
                break
        
        real_images = torch.cat(real_samples, dim=0)[:500].to(self.curr_device)  # Limit to 500 images
        real_labels = torch.cat(labels, dim=0)[:500].to(self.curr_device)
        
        # Generate fake samples
        num_samples = 500  # Same number as real samples
        with torch.no_grad():
            fake_images = self.model.sample(num_samples, self.curr_device, labels=real_labels)
        
        # Preprocess images for the metrics
        # FID and IS expect RGB images in range [0, 255]
        real_images = (real_images * 255).to(torch.uint8)
        fake_images = (fake_images * 255).to(torch.uint8)
        
        # If images are not RGB (3 channels), convert them
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Calculate FID
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        fid_score = fid.compute()
        
        # Calculate IS
        inception_score.update(fake_images)
        is_mean, is_std = inception_score.compute()
        
        # Log directly to the tensorboard logger instead of using self.log()
        if self.logger:
            self.logger.experiment.add_scalar("metrics/fid", fid_score, self.current_epoch)
            self.logger.experiment.add_scalar("metrics/inception_score", is_mean, self.current_epoch)
        
        # Print results
        print(f"\nMetrics For Epoch {self.current_epoch}")
        print(f"FID Score: {fid_score:.4f}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print("==========================")
        
        return {"fid": fid_score, "is_mean": is_mean, "is_std": is_std}