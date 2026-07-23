from __future__ import annotations

import math

import torch


class CosmosRFlowScheduler:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, num_train_timesteps=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = self._log_spaced_sigmas()

    def _log_spaced_sigmas(self):
        steps = self.num_train_timesteps + 1
        log_sigmas = torch.linspace(
            math.log(self.sigma_min), math.log(self.sigma_max), steps
        )
        return torch.exp(log_sigmas)

    def timestep_to_sigma(self, timestep):
        idx = int(timestep * (self.num_train_timesteps - 1))
        return self.sigmas[idx]

    def sigma_to_t(self, sigma):
        return sigma / (sigma + 1.0)

    def calculate_input(self, latent, sigma):
        t = self.sigma_to_t(sigma)
        return latent * (1.0 - t)

    def calculate_denoised(self, model_output, model_input, sigma):
        t = self.sigma_to_t(sigma)
        return model_input * (1.0 - t) - model_output * t


def get_sigmas(scheduler, num_steps):
    indices = torch.linspace(scheduler.num_train_timesteps - 1, 0, num_steps, dtype=torch.long)
    sigmas = scheduler.sigmas[indices]
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(x, x0_pred, sigma_curr, sigma_next):
    noise = (x - x0_pred) / sigma_curr
    return x0_pred + sigma_next * noise


def euler_ancestral_step(x, x0_pred, sigma_curr, sigma_next, generator=None):
    noise = (x - x0_pred) / sigma_curr
    sigma_hat = (sigma_curr ** 2 - sigma_next ** 2).clamp(min=0).sqrt()
    extra_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    return x0_pred + sigma_next * noise + sigma_hat * extra_noise
