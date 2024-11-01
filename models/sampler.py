import numpy as np
import torch


class DiffusionProcess():
    def __init__(self, args, diffusion_fn, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.args = args
        self.device = args.device
        self.shape = shape
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.deterministic = args.deterministic
        # self.karras = 'karras' in args.model
        # self.a_dim = args.a_dim
        # self.model = args.model
        self.net = diffusion_fn.to(device=self.device)
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

    def sample(self, latents, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def impute(self, x, latents, mask, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        noise_impute = (latents.to(torch.float64) * t_steps[0]) * (1 - mask)
        x_image_clear = x * mask
        x_next = x_image_clear + noise_impute
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)

            t_ = (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            noise_impute = t_ * (1 - mask)
            x_to_impute = x_cur * (1 - mask) + noise_impute
            x_cur = x_cur * mask

            x_hat = x_cur + x_to_impute

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            imputed_x_part = (x_hat + (t_next - t_hat) * d_cur) * (1 - mask)
            x_next = x_image_clear + imputed_x_part

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)) * (1 - mask) + x_image_clear

        return x_next


    def forecast(self, past, latents, pad, f_shape, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        s, e = past.shape[-1], f_shape

        # Main sampling loop.
        x_next = (latents.to(torch.float64) * t_steps[0])[..., :e]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            t_ = (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            x_hat = pad(torch.cat([past, x_cur + t_], dim=-1))

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = (x_hat + (t_next - t_hat) * d_cur)[..., s:(s+e)]

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                x_next = pad(torch.cat([past, x_next], dim=-1))
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime))[..., s:(s+e)]

        return x_next



    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return self.sample(xT)


    @torch.no_grad()
    def interpolate(self, x, mask, xT=None):
        if xT is None:
            xT = torch.randn([x.shape[0], *self.shape]).to(device=self.device)

        return self.impute(x, xT, mask)


    @torch.no_grad()
    def forecasting(self, x, pad, f_shape, xT=None):
        if xT is None:
            xT = torch.randn([x.shape[0], *self.shape]).to(device=self.device)

        return self.forecast(x, xT, pad, f_shape)