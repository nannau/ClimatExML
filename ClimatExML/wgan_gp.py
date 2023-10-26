import lightning as pl
import torch.nn.functional as F
import torch
from ClimatExML.models import Generator, Generator_hr_cov, Critic
from ClimatExML.mlclasses import (
    ClimatExMlFlow,
    ClimateExMLTraining,
    HyperParameters,
    InvariantData,
)
from ClimatExML.mlflow_tools.mlflow_tools import (
    gen_grid_images,
    log_metrics_every_n_steps,
    log_pytorch_model,
)
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    multiscale_structural_similarity_index_measure,
)

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
            lr_dim, hr_dim, n_covariates, n_hr_covariates, n_predictands
        )

        self.C = Critic(lr_dim, hr_dim, n_predictands)

        self.automatic_optimization = False

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

    def training_step(self, batch, batch_idx):
        print(batch[0])
        # train generator
        lr, hr, hr_cov = batch[0]
        lr = lr.squeeze(0)
        hr = hr.squeeze(0)
        hr_cov = hr_cov.squeeze(0)
        hr_cov = hr_cov * torch.ones((hr.size(0), 1, hr.size(2), hr.size(3))).to(hr)

        print("SIZES", lr.size(), hr.size(), hr_cov.size())

        sr = self.G(lr, hr_cov).detach()

        g_opt, c_opt = self.optimizers()
        self.toggle_optimizer(c_opt)
        sr = self.G(lr, hr_cov).detach()
        gradient_penalty = self.compute_gradient_penalty(hr, sr)
        mean_sr = torch.mean(self.C(sr))
        mean_hr = torch.mean(self.C(hr))
        loss_c = mean_sr - mean_hr + self.gp_lambda * gradient_penalty

        self.manual_backward(loss_c)
        c_opt.step()
        c_opt.zero_grad()
        self.untoggle_optimizer(c_opt)

        if (batch_idx + 1) % self.n_critic == 0:
            self.toggle_optimizer(g_opt)
            sr = self.G(lr, hr_cov)
            loss_g = -torch.mean(
                self.C(sr).detach()
            ) + self.alpha * mean_absolute_error(sr, hr)
            self.manual_backward(loss_g)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)

        self.log_dict(
            {
                "MAE": mean_absolute_error(sr, hr),
                "MSE": mean_squared_error(sr, hr),
                "MSSIM": multiscale_structural_similarity_index_measure(sr, hr),
                "Wasserstein Distance": mean_hr - mean_sr,
            },
            sync_dist=True,
        )

        # if (batch_idx + 1) % self.log_every_n_steps == 0:
        #     fig = plt.figure(figsize=(30, 10))
        #     for var in range(lr.shape[1] - 1):
        #         self.logger.experiment.log_figure(
        #             mlflow.active_run().info.run_id,
        #             gen_grid_images(
        #                 var,
        #                 fig,
        #                 self.G,
        #                 lr,
        #                 hr,
        #                 hr_cov,
        #                 lr.size(0),
        #                 use_hr_cov=self.hr_cov_shape is not None,
        #                 n_examples=3,
        #                 cmap="viridis",
        #             ),
        #             f"train_images_{var}.png",
        #         )
        #         plt.close()

    def go_downhill(self, opt, loss):
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        return opt_g, opt_d
