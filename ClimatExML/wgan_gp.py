import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from ClimatExML.models import Generator, Critic
from ClimatExML.mlflow_tools.plot_images import gen_grid_images
from ClimatExML.loader import ClimatExMLLoader
from torch.utils.data import DataLoader


class SuperResolutionWGANGP(pl.LightningModule):
    def __init__(
            self,
            data_glob: dict,
            batch_size: int = 16,
            num_workers: int = 24,
            learning_rate: float = 0.00025,
            b1: float = 0.9,
            b2: float = 0.999,
            gp_lambda: float = 10,
            alpha: float = 1e-3,
            lr_shape: tuple = (3, 64, 64),
            hr_shape: tuple = (2, 512, 512),
            n_critic: int = 5,
            log_every_n_steps: int = 100,
            **kwargs
    ):
        super().__init__()
        # self.save_hyperparameters()

        # data
        self.data_glob = data_glob
        self.num_workers = num_workers

        # training
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic
        self.alpha = alpha
        self.log_every_n_steps = log_every_n_steps

        # networks
        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape
        self.G = Generator(lr_dim, n_covariates, n_predictands)
        self.C = Critic(lr_dim, hr_dim, n_predictands)

        self.automatic_optimization = False

    def setup(self, stage: str):
        self.lr_test = ClimatExMLLoader(self.data_glob['lr_test'])
        self.hr_test = ClimatExMLLoader(self.data_glob['hr_test'])

        self.lr_train = ClimatExMLLoader(self.data_glob['lr_train'])
        self.hr_train = ClimatExMLLoader(self.data_glob['hr_train'])

    def train_dataloader(self):
        return {
            "lr": DataLoader(self.lr_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
            "hr": DataLoader(self.hr_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        }

    def test_dataloader(self):
        return {
            "lr": DataLoader(self.lr_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
            "hr": DataLoader(self.hr_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        }

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        current_batch_size = real_samples.size(0)
        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=self.device)
        alpha = alpha.expand_as(real_samples)

        interpolated = alpha * real_samples.data + (1 - alpha) * fake_samples.data

        # Calculate probability of interpolated examples
        critic_interpolated = self.C(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(self.batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_lambda * ((gradients_norm - 1) ** 2).mean()

    def training_step(self, batch, batch_idx):
        # train generator
        lr, hr = batch["lr"], batch["hr"]
        sr = self.G(lr)
        g_opt, c_opt = self.optimizers()

        # update discriminator every other step
        gradient_penalty = self.compute_gradient_penalty(hr, sr)
        loss_c = -torch.mean(self.C(hr)) + torch.mean(self.C(sr)) + self.gp_lambda * gradient_penalty
        self.go_downhill(c_opt, loss_c)

        if (batch_idx + 1) % self.n_critic == 0:
            sr = self.G(lr)
            loss_g = -torch.mean(self.C(sr)) + self.alpha*F.l1_loss(sr, hr)
            self.go_downhill(g_opt, loss_g)

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            self.log_dict(
                {
                    "MAE loss": F.l1_loss(sr, hr),
                    "MSE loss": F.mse_loss(sr, hr),
                    "wasserstein": torch.mean(self.C(hr)) - torch.mean(self.C(sr)),
                },
            )

    def go_downhill(self, opt, loss):
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def test_step(self, batch, batch_idx):
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            lr, hr = batch["lr"], batch["hr"]
            sr = self.G(lr)
            self.log_dict(
                {
                    "Test_MAE loss": F.l1_loss(sr, hr),
                    "Test_MSE loss": F.mse_loss(sr, hr),
                    "Test_wasserstein": torch.mean(self.C(hr)) - torch.mean(self.C(sr)),
                },
            )
            self.logger.experiment.log_figure(
                gen_grid_images(
                    self.G,
                    lr,
                    hr,
                    self.batch_size,
                    n_examples=3,
                    cmap="viridis"
                ),
                "test_images.png",
                on_step=True,
            )


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        return opt_g, opt_d
