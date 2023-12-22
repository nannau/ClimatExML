import lightning as pl
import torch
from ClimatExML.models import Generator, Generator_hr_cov, Critic
from ClimatExML.mlclasses import (
    ClimatExMlFlow,
    ClimateExMLTraining,
    HyperParameters,
    InvariantData,
)
import torch.nn as nn
from ClimatExML.mlflow_tools.mlflow_tools import gen_grid_images

from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    # multiscale_structural_similarity_index_measure,
)
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

import mlflow
import matplotlib.pyplot as plt


class SuperResolutionWGANGP(pl.LightningModule):
    tracking: ClimatExMlFlow
    hardware: ClimateExMLTraining
    hyperparameters: HyperParameters
    invariant: InvariantData
    is_noise: bool

    def __init__(
        self,
        tracking,
        hardware,
        hyperparameters,
        invariant,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = hardware.num_workers
        self.log_every_n_steps = tracking.log_every_n_steps
        self.validation_log_every_n_steps = tracking.validation_log_every_n_steps

        self.learning_rate = hyperparameters.learning_rate
        self.b1 = hyperparameters.b1
        self.b2 = hyperparameters.b2
        self.gp_lambda = hyperparameters.gp_lambda
        self.n_critic = hyperparameters.n_critic
        self.alpha = hyperparameters.alpha
        self.is_noise = hyperparameters.noise_injection

        self.lr_shape = invariant.lr_shape
        self.hr_shape = invariant.hr_shape
        self.hr_invariant_shape = invariant.hr_invariant_shape

        # networks
        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape

        n_hr_covariates = self.hr_invariant_shape[0]
        self.G = Generator_hr_cov(
            self.is_noise, lr_dim, hr_dim, n_covariates, n_hr_covariates, n_predictands
        )

        self.C = Critic(lr_dim, hr_dim, n_predictands)
        self.cross_entropy = nn.CrossEntropyLoss()

        mlflow.pytorch.autolog()

        self.automatic_optimization = False


        self.delta_precip_hr = - 0.0769779160618782 / 0.3726992905139923
        self.delta_precip_lr = - 0.0795731395483017 / 0.2713610827922821

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        current_batch_size = real_samples.size(0)
        # Calculate interpolation

        # gradient penalty
        gp_alpha = (
            torch.rand(current_batch_size, 1, 1, 1, requires_grad=True)
            .expand_as(real_samples)
            .to(real_samples)
        )

        interpolated = gp_alpha * real_samples.data + (1 - gp_alpha) * fake_samples.data

        # Calculate probability of interpolated examples
        critic_interpolated = self.C(interpolated)

        # self.register_buffer("gp_ones", torch.ones(critic_interpolated.size(), requires_grad=True))

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), requires_grad=True).to(
                real_samples
            ),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(current_batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def kurtoses(self, x):
        mean = x.mean()
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        return torch.mean(torch.pow(zscores, 4.0)) - 3.0



    def training_step(self, batch, batch_idx):
        # train generator
        lr, hr, hr_cov = batch[0]
        
        hr_pr_mask = 1.0*(hr[:, 0, ...] > self.delta_precip_hr)
        lr_pr_mask = 1.0*(lr[:, 0, ...] > self.delta_precip_lr)

        g_opt, c_opt = self.optimizers()
        self.toggle_optimizer(c_opt)

        lr_w_mask = torch.cat([lr, lr_pr_mask.unsqueeze(1)], dim=1)
        # hr_w_mask = torch.stack([hr, hr_pr_mask.unsqueeze(1)], dim=1)

        sr = self.G(lr_w_mask, hr_cov).detach()
        sr_pr_mask = 1.0*(sr[:, 0, ...] > self.delta_precip_hr)

        gradient_penalty = self.compute_gradient_penalty(hr, sr)
        mean_sr = torch.mean(self.C(sr))
        mean_hr = torch.mean(self.C(hr))
        loss_c = mean_sr - mean_hr + self.gp_lambda * gradient_penalty

        # precip_cross_entropy = torch.BoolTensor([False, False, False, False, False, True])

        # Consider changing this to weights instead of a mask
        self.go_downhill(loss_c, c_opt)

        if (batch_idx + 1) % self.n_critic == 0: 
            weight_mse = torch.Tensor([0.0001, 1, 1, 1, 1]).type_as(hr)
            self.toggle_optimizer(g_opt)
            sr = self.G(lr_w_mask, hr_cov)
            sr_pr_mask = sr[:, 0, ...] > self.delta_precip_hr
            loss_g = -torch.mean(
                self.C(sr).detach()
            ) + self.alpha * mean_squared_error(
                torch.einsum("bchw,c->bchw", sr, weight_mse),
                torch.einsum("bchw,c->bchw", hr, weight_mse),
            ) + self.cross_entropy(
                1.0*sr_pr_mask, hr_pr_mask
            ) + (self.kurtoses(sr[:, 0, ...]) - self.kurtoses(hr[:, 0, ...]))**2  

            self.go_downhill(loss_g, g_opt)

        self.log_dict(
            {
                "MAE": mean_absolute_error(sr.detach(), hr),
                "MSE": mean_squared_error(sr.detach(), hr),
                "MSSIM": multiscale_structural_similarity_index_measure(sr.detach(), hr),
                "Wasserstein Distance": mean_hr.detach() - mean_sr.detach(),
                "Cross Entropy on Precip Mask": self.cross_entropy(1.0*sr_pr_mask.detach(), hr_pr_mask).detach(),
                "kurtoses": (self.kurtoses(sr[:, 0, ...]) - self.kurtoses(hr[:, 0, ...]))**2,
            },
            sync_dist=True,
        )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            for var in range(hr.shape[1]):
                fig = plt.figure(figsize=(30, 10))
                fig = gen_grid_images(
                    var,
                    fig,
                    self.G,
                    lr_w_mask,
                    hr,
                    hr_cov,
                    use_hr_cov=self.hr_invariant_shape is not None,
                    n_examples=3,
                    cmap="viridis",
                )
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"train_images_{var}.png",
                )
                plt.close(fig)

    def validation_step(self, batch, batch_idx):
        # train generator
        lr, hr, hr_cov = batch
        
        hr_pr_mask = 1.0*(hr[:, 0, ...] > self.delta_precip_hr)
        lr_pr_mask = 1.0*(lr[:, 0, ...] > self.delta_precip_lr)
        lr_w_mask = torch.cat([lr, lr_pr_mask.unsqueeze(1)], dim=1)

        sr = self.G(lr_w_mask, hr_cov).detach()
        sr_pr_mask = 1.0*(sr[:, 0, ...] > self.delta_precip_hr)
        mean_sr = torch.mean(self.C(sr).detach())
        mean_hr = torch.mean(self.C(hr).detach())
        self.log_dict(
            {
                "Validation MAE": mean_absolute_error(sr, hr),
                "Validation MSE": mean_squared_error(sr, hr),
                "Validation MSSIM": multiscale_structural_similarity_index_measure(
                    sr, hr
                ),
                "Validation Wasserstein Distance": mean_hr - mean_sr,
                "Validation Cross Entropy on Precip Mask": self.cross_entropy(sr_pr_mask.detach(), hr_pr_mask).detach(),
                "Validation kurtoses": (self.kurtoses(sr[:, 0, ...]) - self.kurtoses(hr[:, 0, ...]))**2,

            },
            sync_dist=True,
        )

        if (batch_idx + 1) % self.validation_log_every_n_steps == 0:
            for var in range(hr.shape[1]):
                fig = plt.figure(figsize=(30, 10))
                fig = gen_grid_images(
                    var,
                    fig,
                    self.G,
                    lr_w_mask,
                    hr,
                    hr_cov,
                    use_hr_cov=self.hr_invariant_shape is not None,
                    n_examples=3,
                    cmap="viridis",
                )
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"validation_images_{var}.png",
                )
                plt.close(fig)

    # def on_train_epoch_end(self):
    #     self.logger._log_model(self.G, "G")
    #     self.logger._log_model(self.C, "C")
        # artifact_path = f"models/"
        # torch.save(self.G.state_dict(), artifact_path + "G.pt")
        # torch.save(self.C.state_dict(), artifact_path + "C.pt")
        # self.log_artifacts(artifact_path)


    def go_downhill(self, loss, opt):
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        return opt_g, opt_d
