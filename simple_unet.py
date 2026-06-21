"""
Architektura RestorationUNet dla restauracji obrazów portretowych.

U-Net (Ronneberger et al.) z modyfikacjami dostosowanymi do image restoration:
- Residual learning: model przewiduje korektę pikseli, nie obraz bezpośrednio.
- tanh × 0.5 ogranicza korektę do [-0.5, 0.5] — model może zarówno rozjaśniać,
  jak i przyciemniać piksele.
- AvgPool2d przy downsamplingu zachowuje więcej informacji niż MaxPool2d.
- Inicjalizacja Kaiming dla stabilnego startu; warstwa residual startuje od zera
  (na początku treningu model jest bliski funkcji tożsamości).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Dwa bloki Conv → BN → LeakyReLU z inicjalizacją Kaiming."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RestorationUNet(nn.Module):
    """
    U-Net z residual learning dla restauracji obrazów.

    output = clamp(input + tanh(residual_head(decoder_features)) × 0.5,  0, 1)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_features: int = 64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,     f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        # AvgPool zamiast MaxPool — lepsze zachowanie średniej przestrzennej
        # przy downsamplingu potrzebnym do rekonstrukcji
        self.pool = nn.AvgPool2d(2)

        # Decoder (kanały podwojone przez skip connections z torch.cat)
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3  = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8,  f * 4)

        self.up2  = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4,  f * 2)

        self.up1  = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2,  f)

        # Warstwa residual bez aktywacji; tanh stosowany w forward()
        self.residual_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        self._init_output_layers()

    def _init_output_layers(self):
        """Residual startuje od zera; sieć zachowuje się jak tożsamość na początku treningu."""
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)

        for m in [self.up4, self.up3, self.up2, self.up1]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = torch.tanh(self.residual_conv(d1)) * 0.5
        return torch.clamp(identity + residual, 0.0, 1.0)


# Alias dla kompatybilności wstecznej
SimpleUNet = RestorationUNet
