---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Customize

```{note}
These instructions may evolve as the project evolves, but the core functionality should remain the same.
```

Configuration and customization typically occurs at a high level in the `ClimatExML/conf/config.yaml` file, or in lower levels in the `.py` files. As a quick outline of what each file does:

## config.yaml

Here is where paths, hyperparameters, covariates, tracking, and other information is read in by the project. Users can choose to hard-code in data paths, or use environment variables (e.g. `${oc.env:PROJECT_DIR}` in the yaml file). Importantly, this is also where the project name and experiment name is supplied to CometML. Please always use a unique `experiment_name` unless you plan on overwriting trained models.

## train.py

This is the launch point of the program, and so high-level changes can be made here. Usually changes to this file are major, so if you find yourself modifying this file, perhaps consider your problem elsewhere or consider introducing functionality in a new module or file.

## wgan-gp.py

This is where the low-level training methods occur. Here you can add losses to log, change the training algorithm, and so on. Specifically, the function `training_step` is looped while training so research and experimentation can go here. Below is an example of this loop with inline comments to explain:

```python
 def training_step(self, batch, batch_idx):

        # train generator
        lr, hr, hr_cov = batch[0] # load the data from the dataloader
        lr = lr.float() # change to float
        hr = hr.float()
        hr_cov = hr_cov.float()

        g_opt, c_opt = self.optimizers() # get optimizers
        self.toggle_optimizer(c_opt) # turn ON critic optimizer for first n_critic steps

        sr = self.G(lr, hr_cov).detach() # detach generated field from Generator network so graph is not used (leads to memory leaks)

        gradient_penalty = self.compute_gradient_penalty(hr, sr) # compute WGAN-GP grad penalty
        mean_sr = torch.mean(self.C(sr)) # compute mean of vectors for wasserstein distance
        mean_hr = torch.mean(self.C(hr))
        loss_c = mean_sr - mean_hr + self.gp_lambda * gradient_penalty # assemble final critic loss function

        self.go_downhill(loss_c, c_opt) # go downhill performs the gradient descent step

        if (batch_idx + 1) % self.n_critic == 0: # ever n_critic steps, update the generator network!
            self.toggle_optimizer(g_opt) # turn ON the generator optimizer
            sr = self.G(lr, hr_cov) # generate example field
            loss_g = -torch.mean(self.C(sr).detach()) + self.alpha * mean_squared_error(
                sr, hr
            ) # use the critic loss (adversarial loss) and MSE to assemble generator loss

            self.go_downhill(loss_g, g_opt) # gradient step downhill

        self.log_dict(
            self.losses("Train", hr, sr.detach(), mean_sr.detach(), mean_hr.detach()),
            sync_dist=True,
        ) # log training losses

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            self.configure_figure(
                "Train",
                lr,
                hr,
                hr_cov,
                n_examples=3,
                cmap="viridis",
            ) # create figure and log
```