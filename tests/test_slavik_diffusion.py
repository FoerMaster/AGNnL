import unittest

import torch

from slavik_activations import SlavikA, SlavikR2
from slavik_diffusion import SlavikAutoencoder, SlavikUNet, diffusion_schedule


class SlavikDiffusionTests(unittest.TestCase):
    def test_sla_activation_supports_last_and_channel_dimensions(self):
        last = SlavikA(3)
        conv = SlavikA(3, channel_dim=1)
        x = torch.randn(2, 4, 5, 3, requires_grad=True)
        y = last(x)
        self.assertEqual(y.shape, x.shape)
        y.sum().backward()
        self.assertIsNotNone(last.alpha.grad)

        x_conv = torch.randn(2, 3, 7, 7, requires_grad=True)
        y_conv = conv(x_conv)
        self.assertEqual(y_conv.shape, x_conv.shape)
        y_conv.sum().backward()
        self.assertIsNotNone(conv.alpha.grad)

    def test_r2_activation_shape(self):
        activation = SlavikR2(5)
        x = torch.randn(2, 4, 5)
        self.assertEqual(activation(x).shape, x.shape)

    def test_autoencoder_round_trip_shapes(self):
        model = SlavikAutoencoder(latent_channels=2, base_channels=16)
        x = torch.randn(2, 1, 28, 28)
        reconstruction, mu, logvar = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (2, 2, 7, 7))
        self.assertEqual(logvar.shape, mu.shape)

    def test_unet_and_diffusion_schedule_shapes(self):
        model = SlavikUNet(latent_channels=2, base_channels=16)
        x = torch.randn(2, 2, 7, 7)
        timesteps = torch.tensor([0, 3])
        labels = torch.tensor([1, 10])
        self.assertEqual(model(x, timesteps, labels).shape, x.shape)
        betas, alphas, alpha_bars = diffusion_schedule(8, torch.device("cpu"))
        self.assertEqual(betas.shape, (8,))
        self.assertTrue(torch.all(alphas > 0))
        self.assertTrue(torch.all(alpha_bars[1:] < alpha_bars[:-1]))
        self.assertLess(float(alpha_bars[-1]), 0.01)

    def test_cosine_schedule_ends_near_pure_noise(self):
        _, _, alpha_bars = diffusion_schedule(100, torch.device("cpu"))
        self.assertLess(float(alpha_bars[-1]), 0.001)
        self.assertGreater(float(alpha_bars[-1]), 0)


if __name__ == "__main__":
    unittest.main()
