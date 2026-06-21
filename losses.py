"""
Moduł funkcji strat dla restauracji obrazów portretowych.

Trzy strategie dostosowane do różnych problemów:
1. PERCEPTUAL - ogólna poprawa jakości (LPIPS + MS-SSIM + L1)
2. FREQUENCY - deblurring, szczegóły (FFT Loss + Gradient + L1)  
3. IDENTITY - zachowanie tożsamości twarzy (ArcFace + LPIPS + L1)

Biologiczne uzasadnienie:
- Ludzki wzrok jest wrażliwy na strukturę, nie na pojedyncze piksele
- Funkcja wrażliwości kontrastowej (CSF) - max czułość 4-8 cykli/stopień
- Luminancja ważniejsza niż chrominancja (kanał Y vs Cb/Cr)
- Korelacje lokalne - sąsiednie piksele są podobne w naturalnych obrazach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Tuple
import math


# ============================================================================
# KOMPONENTY BAZOWE
# ============================================================================

class L1CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1) - bardziej odporna na outliers niż L1.
    
    L_char = sqrt((x - y)^2 + eps^2)
    
    Biologicznie: odpowiada na błędy percepcyjne bardziej płynnie,
    nie karze ekstremalnie za pojedyncze duże błędy.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class GradientLoss(nn.Module):
    """
    Strata gradientowa - mierzy różnice w krawędziach i teksturach.
    
    Biologicznie: ludzki wzrok jest bardzo wrażliwy na krawędzie
    (komórki ganglionowe typu edge detector w siatkówce).
    Gradient aproksymuje odpowiedź filtrów kierunkowych w V1.
    
    ZOPTYMALIZOWANE: Proste różnice zamiast konwolucji Sobela
    (szybsze i bardziej kompatybilne z CUDA).
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Konwertuj do luminancji (Y) - wzrok bardziej wrażliwy na jasność
        # Współczynniki ITU-R BT.601
        pred_y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_y = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Gradienty przez proste różnice (szybsze i bardziej stabilne niż Sobel conv)
        # Gradient X (poziomy)
        pred_gx = pred_y[:, :, :, 1:] - pred_y[:, :, :, :-1]
        target_gx = target_y[:, :, :, 1:] - target_y[:, :, :, :-1]
        
        # Gradient Y (pionowy)
        pred_gy = pred_y[:, :, 1:, :] - pred_y[:, :, :-1, :]
        target_gy = target_y[:, :, 1:, :] - target_y[:, :, :-1, :]
        
        # L1 między gradientami
        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index - mierzy podobieństwo strukturalne.
    
    SSIM porównuje:
    - Luminancję (średnia jasność)
    - Kontrast (odchylenie standardowe)
    - Strukturę (korelacja znormalizowana)
    
    NAPRAWIONE: Dodano stabilność numeryczną (clamp na sigma_sq)
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))
        
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Tworzy Gaussowskie okno."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.outer(g)
        window = window.view(1, 1, window_size, window_size)
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        
        # Rozszerz okno na wszystkie kanały
        window = self.window.repeat(channel, 1, 1, 1)
        
        # Średnie
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Wariancje - CLAMP dla stabilności numerycznej!
        # E[X^2] - E[X]^2 może być ujemne przez błędy zaokrągleń
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # KLUCZOWE: clamp wariancji do wartości nieujemnych
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
        
        # Stałe stabilizujące (dla dynamic range [0, 1])
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)  # +eps dla pewności
        
        # Clamp SSIM do [0, 1] (może wyjść poza przez numeryczne błędy)
        ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        
        return 1.0 - ssim_map.mean()


class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM - SSIM na różnych skalach.
    
    Biologicznie: wzrok przetwarza obraz na wielu skalach przestrzennych
    (różne wielkości pól recepcyjnych, od małych w fovea do dużych na peryferiach).
    
    Różne skale odpowiadają różnym odległościom oglądania.
    """
    def __init__(self, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.ssim = SSIMLoss()
        # Wagi dla różnych skal (empirycznie dobrane)
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363][:levels]
        self.weights = [w / sum(self.weights) for w in self.weights]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        
        for i in range(self.levels):
            loss = self.ssim(pred, target)
            losses.append(loss * self.weights[i])
            
            if i < self.levels - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
        
        return sum(losses)


# ============================================================================
# STRATY PERCEPCYJNE (VGG-based)
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss używając cech VGG.
    
    Biologicznie: VGG trenowany na ImageNet nauczył się reprezentacji
    podobnych do tych w korze wzrokowej:
    - Wczesne warstwy: krawędzie, tekstury (V1)
    - Środkowe warstwy: części obiektów (V2, V4)
    - Późne warstwy: obiekty, semantyka (IT)
    
    Używamy wczesnych i środkowych warstw dla tekstur i struktury.
    """
    def __init__(self, layers: Tuple[int, ...] = (3, 8, 15, 22)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.ModuleList()
        
        prev_idx = 0
        for idx in layers:
            self.features.append(nn.Sequential(*list(vgg.features.children())[prev_idx:idx]))
            prev_idx = idx
        
        # Zamroź wagi
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalizacja ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Wagi dla różnych warstw (wczesne ważniejsze dla tekstur)
        self.layer_weights = [1.0, 0.75, 0.5, 0.25]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalizacja
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            loss += self.layer_weights[i] * F.l1_loss(pred, target)
        
        return loss


# ============================================================================
# STRATY CZĘSTOTLIWOŚCIOWE (FFT-based)
# ============================================================================

class FFTLoss(nn.Module):
    """
    Strata w dziedzinie częstotliwości (Fourier).
    
    Biologicznie motywowana - funkcja wrażliwości kontrastowej (CSF).
    
    ZOPTYMALIZOWANE:
    - Log-amplitude (standardowe w analizie spektralnej)
    - Znormalizowane do sensownego zakresu wartości
    - Fokus na wysokie częstotliwości (szczegóły, krawędzie)
    """
    def __init__(self, focus_on_high_freq: bool = True):
        super().__init__()
        self.focus_on_high_freq = focus_on_high_freq
        self._weights_cache = {}
    
    def _get_frequency_weights(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Tworzy macierz wag częstotliwościowych - CSF-inspired."""
        cache_key = (h, w, str(device))
        if cache_key in self._weights_cache:
            return self._weights_cache[cache_key]
        
        # Siatka częstotliwości
        freq_y = torch.fft.fftfreq(h, device=device).view(-1, 1)
        freq_x = torch.fft.fftfreq(w, device=device).view(1, -1)
        freq_r = torch.sqrt(freq_y ** 2 + freq_x ** 2)
        
        if self.focus_on_high_freq:
            # High-pass: tłumi DC i niskie, przepuszcza wysokie
            # Użyj prostej funkcji sigmoidalnej
            weights = torch.sigmoid((freq_r - 0.05) * 50)  # Przejście przy ~5% Nyquista
        else:
            weights = torch.ones_like(freq_r)
        
        self._weights_cache[cache_key] = weights
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Konwertuj do luminancji (Y channel)
        pred_y = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_y = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # FFT 2D
        pred_fft = torch.fft.fft2(pred_y)
        target_fft = torch.fft.fft2(target_y)
        
        # Log-amplitude (standardowe w spektralnej analizie)
        # +1e-8 dla stabilności numerycznej
        pred_log_amp = torch.log(torch.abs(pred_fft) + 1e-8)
        target_log_amp = torch.log(torch.abs(target_fft) + 1e-8)
        
        # Wagi częstotliwościowe
        h, w = pred_y.shape[-2:]
        weights = self._get_frequency_weights(h, w, pred.device)
        
        # Ważona różnica L1 w przestrzeni log-amplitude
        diff = torch.abs(pred_log_amp - target_log_amp) * weights
        
        # Średnia daje wartości ~0.01-0.1 (sensowny zakres)
        return diff.mean()


class LaplacianPyramidLoss(nn.Module):
    """
    Strata na piramidzie Laplace'a - wieloskalowa analiza częstotliwości.
    
    Każdy poziom piramidy zawiera różne pasma częstotliwości.
    Biologicznie odpowiada hierarchii przetwarzania w korze wzrokowej.
    """
    def __init__(self, levels: int = 4):
        super().__init__()
        self.levels = levels
        # Wyższe poziomy (niższe częstotliwości) mają mniejszą wagę
        self.weights = [2 ** i for i in range(levels)]
        self.weights = [w / sum(self.weights) for w in self.weights]
    
    def _laplacian_pyramid(self, img: torch.Tensor) -> list:
        """Buduje piramidę Laplace'a."""
        pyramid = []
        current = img
        
        for _ in range(self.levels - 1):
            down = F.avg_pool2d(current, 2)
            up = F.interpolate(down, size=current.shape[-2:], mode='bilinear', align_corners=False)
            laplacian = current - up
            pyramid.append(laplacian)
            current = down
        
        pyramid.append(current)  # Ostatni poziom to residual
        return pyramid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_pyramid = self._laplacian_pyramid(pred)
        target_pyramid = self._laplacian_pyramid(target)
        
        loss = 0.0
        for i, (p, t) in enumerate(zip(pred_pyramid, target_pyramid)):
            loss += self.weights[i] * F.l1_loss(p, t)
        
        return loss


# ============================================================================
# STRATA TOŻSAMOŚCI (dla twarzy)
# ============================================================================

class FaceIdentityLoss(nn.Module):
    """
    Strata zachowania tożsamości twarzy.
    
    Używa sieci rozpoznawania twarzy (np. ResNet pretrenowany)
    do wyciągnięcia embeddingów twarzy i porównania ich.
    
    Biologicznie: odpowiada obszarowi FFA (Fusiform Face Area)
    specjalizującemu się w rozpoznawaniu twarzy.
    """
    def __init__(self):
        super().__init__()
        # Używamy ResNet18 jako feature extractor
        # W produkcji lepiej użyć ArcFace/CosFace
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Zamroź wagi
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalizacja
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalizacja
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Upewnij się że obrazy mają min 224x224
        if pred.shape[-1] < 224:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Wyciągnij embeddingi
        pred_feat = self.features(pred).flatten(1)
        target_feat = self.features(target).flatten(1)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_feat, target_feat)
        return 1 - cos_sim.mean()


# ============================================================================
# SZYBKIE STRATEGIE (bez ciężkich sieci jak VGG/ResNet)
# ============================================================================

class FastPerceptualStrategy(nn.Module):
    """
    SZYBKA strategia percepcyjna - bez VGG.
    
    Zoptymalizowana dla PSNR + SSIM:
    - L1 Charbonnier (główny driver PSNR)
    - SSIM (główny driver SSIM) - WYSOKA WAGA
    - Gradient Loss (krawędzie, szczegóły)
    
    Wagi dobrane empirycznie dla balansu PSNR/SSIM.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.5,      # Zwiększone dla lepszego SSIM
        gradient_weight: float = 0.3,   # Zmniejszone - mniej agresywne
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.gradient_weight * gradient
        )
        
        return {
            'total': total,
            'l1': l1,
            'ssim': ssim,
            'gradient': gradient,
        }


class FastFrequencyStrategy(nn.Module):
    """
    SZYBKA strategia częstotliwościowa - dla deblurringu.
    
    Zoptymalizowana - fokus na wysokie częstotliwości:
    - L1 (rekonstrukcja bazowa)
    - FFT Loss (szczegóły częstotliwościowe)
    - SSIM (struktura)
    - Gradient (krawędzie)
    
    WAGI ZBALANSOWANE:
    - FFT loss ~ 0.5-0.7 -> waga 0.2 -> wkład ~0.1-0.14
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        fft_weight: float = 0.2,        # Zbalansowane (było 100 - za dużo!)
        ssim_weight: float = 0.8,       # SSIM ważne dla jakości
        gradient_weight: float = 0.5,   # Gradient dla krawędzi
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.fft_loss = FFTLoss(focus_on_high_freq=True)
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        
        self.l1_weight = l1_weight
        self.fft_weight = fft_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        fft = self.fft_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.fft_weight * fft +
            self.ssim_weight * ssim +
            self.gradient_weight * gradient
        )
        
        return {
            'total': total,
            'l1': l1,
            'fft': fft,
            'ssim': ssim,
            'gradient': gradient,
        }


class FastIdentityStrategy(nn.Module):
    """
    SZYBKA strategia dla zachowania struktury twarzy.
    
    Zoptymalizowana - SSIM dominuje dla zachowania struktury:
    - SSIM (bardzo wysoka waga - główny cel)
    - L1 (rekonstrukcja pikseli)
    - Gradient (detale twarzy)
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 2.0,       # Bardzo wysoka - struktura najważniejsza
        gradient_weight: float = 0.2,
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.gradient_weight * gradient
        )
        
        return {
            'total': total,
            'l1': l1,
            'ssim': ssim,
            'gradient': gradient,
        }


class FastCombinedStrategy(nn.Module):
    """
    NOWA strategia łącząca najlepsze elementy wszystkich.
    
    Balans między PSNR (L1) i SSIM (struktura):
    - L1 Charbonnier dla PSNR (główny driver)
    - SSIM dla jakości strukturalnej (wysoki priorytet)
    - FFT dla szczegółów (pomocniczy)
    - Gradient dla krawędzi (pomocniczy)
    
    WAGI ZBALANSOWANE tak, by każdy komponent miał podobny wpływ:
    - L1 loss ~ 0.05-0.08 -> waga 1.0 -> wkład ~0.05-0.08
    - SSIM loss ~ 0.3-0.5 -> waga 1.0 -> wkład ~0.3-0.5
    - FFT loss ~ 0.5-0.7 -> waga 0.1 -> wkład ~0.05-0.07
    - Gradient ~ 0.02 -> waga 1.0 -> wkład ~0.02
    
    Cel: PSNR > 22dB i SSIM > 0.70
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,       # Główny dla SSIM
        fft_weight: float = 0.1,        # Pomocniczy (było 50 - za dużo!)
        gradient_weight: float = 1.0,   # Pomocniczy dla krawędzi
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.fft_loss = FFTLoss(focus_on_high_freq=True)
        self.gradient_loss = GradientLoss()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        fft = self.fft_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.fft_weight * fft +
            self.gradient_weight * gradient
        )
        
        return {
            'total': total,
            'l1': l1,
            'ssim': ssim,
            'fft': fft,
            'gradient': gradient,
        }


# ============================================================================
# STRATEGIE PEŁNE (z VGG/ResNet - wolniejsze ale dokładniejsze)
# ============================================================================

class PerceptualStrategy(nn.Module):
    """
    Strategia PERCEPTUAL - ogólna poprawa jakości percepcyjnej.
    
    Łączy:
    - L1/Charbonnier: rekonstrukcja pikseli
    - VGG Perceptual: tekstury i struktura
    - MS-SSIM: podobieństwo strukturalne na wielu skalach
    
    Użycie: ogólna poprawa jakości, super-resolution, denoising.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.5,
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.ssim_loss = MultiScaleSSIMLoss()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.ssim_weight * ssim
        )
        
        return {
            'total': total,
            'l1': l1,
            'perceptual': perceptual,
            'ssim': ssim,
        }


class FrequencyStrategy(nn.Module):
    """
    Strategia FREQUENCY - dla deblurringu i przywracania szczegółów.
    
    Łączy:
    - L1: rekonstrukcja podstawowa
    - FFT Loss: przywracanie wysokich częstotliwości (szczegóły)
    - Gradient Loss: zachowanie krawędzi
    - Laplacian Pyramid: wieloskalowa analiza
    
    Biologicznie motywowana:
    - Blur = utrata wysokich częstotliwości
    - FFT Loss celuje dokładnie w to co zostało utracone
    - Gradient Loss odpowiada detekcji krawędzi w V1
    
    Użycie: deblurring, motion blur, defocus.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        fft_weight: float = 0.1,
        gradient_weight: float = 0.5,
        laplacian_weight: float = 0.2,
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.fft_loss = FFTLoss(focus_on_high_freq=True)
        self.gradient_loss = GradientLoss()
        self.laplacian_loss = LaplacianPyramidLoss()
        
        self.l1_weight = l1_weight
        self.fft_weight = fft_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        fft = self.fft_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        laplacian = self.laplacian_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.fft_weight * fft +
            self.gradient_weight * gradient +
            self.laplacian_weight * laplacian
        )
        
        return {
            'total': total,
            'l1': l1,
            'fft': fft,
            'gradient': gradient,
            'laplacian': laplacian,
        }


class IdentityStrategy(nn.Module):
    """
    Strategia IDENTITY - zachowanie tożsamości twarzy.
    
    Łączy:
    - L1: rekonstrukcja pikseli
    - VGG Perceptual: tekstury
    - Face Identity: zachowanie tożsamości
    
    Biologicznie:
    - Face Identity odpowiada FFA (Fusiform Face Area)
    - Zapewnia że restauracja nie zmienia "kto" jest na zdjęciu
    
    Użycie: restauracja portretów, face enhancement, aging/de-aging.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        identity_weight: float = 0.5,
    ):
        super().__init__()
        self.l1_loss = L1CharbonnierLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.identity_loss = FaceIdentityLoss()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        identity = self.identity_loss(pred, target)
        
        total = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.identity_weight * identity
        )
        
        return {
            'total': total,
            'l1': l1,
            'perceptual': perceptual,
            'identity': identity,
        }


# ============================================================================
# FABRYKA STRATEGII
# ============================================================================

def get_loss_strategy(
    strategy_name: str,
    fast: bool = False,
    **kwargs
) -> nn.Module:
    """
    Fabryka strategii strat.
    
    Args:
        strategy_name: "perceptual", "frequency", "identity", lub "combined"
        fast: True = szybkie wersje bez VGG/ResNet (~10x szybciej)
        **kwargs: dodatkowe parametry dla strategii
    
    Returns:
        Moduł strategii strat
    
    Strategie Fast:
        - perceptual: L1 + SSIM (wysoka) + Gradient
        - frequency: L1 + FFT (wysoka) + SSIM + Gradient  
        - identity: L1 + SSIM (bardzo wysoka) + Gradient
        - combined: L1 + SSIM + FFT + Gradient (balans PSNR/SSIM)
    
    Przykład:
        >>> criterion = get_loss_strategy("combined", fast=True)
        >>> criterion = get_loss_strategy("perceptual", fast=False)
    """
    if fast:
        strategies = {
            'perceptual': FastPerceptualStrategy,
            'frequency': FastFrequencyStrategy,
            'identity': FastIdentityStrategy,
            'combined': FastCombinedStrategy,  # NOWA - najlepsza dla balansu PSNR/SSIM
        }
    else:
        strategies = {
            'perceptual': PerceptualStrategy,
            'frequency': FrequencyStrategy,
            'identity': IdentityStrategy,
            # combined nie ma wersji slow - używaj fast
        }
    
    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Nieznana strategia: {strategy_name}. "
            f"Dostępne (fast={fast}): {available}"
        )
    
    return strategies[strategy_name](**kwargs)


# ============================================================================
# METRYKI EWALUACJI
# ============================================================================

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio - podstawowa metryka jakości."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())


_ssim_instances: dict = {}


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM jako metryka (nie loss). Cachuje SSIMLoss per device — bez alokacji przy każdym batchu."""
    device = pred.device
    if device not in _ssim_instances:
        _ssim_instances[device] = SSIMLoss().to(device)
    return 1 - _ssim_instances[device](pred, target).item()


# Test
if __name__ == "__main__":
    # Test strategii
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred = torch.rand(2, 3, 256, 256, device=device)
    target = torch.rand(2, 3, 256, 256, device=device)
    
    for strategy_name in ['perceptual', 'frequency', 'identity']:
        print(f"\n=== Strategia: {strategy_name} ===")
        criterion = get_loss_strategy(strategy_name).to(device)
        losses = criterion(pred, target)
        for name, value in losses.items():
            print(f"  {name}: {value.item():.4f}")
