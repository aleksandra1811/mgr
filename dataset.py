"""
Dataset do restauracji obrazów portretowych CelebA-HQ.
Obsługuje różne typy degradacji: blur, noise, low-light.

Zoptymalizowany dla RTX 3070 (8GB VRAM):
- Efektywne ładowanie z dysku
- Opcjonalne cache w pamięci RAM
- Wielowątkowe ładowanie danych
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Literal, List
import numpy as np
import cv2
import torch
import torch.utils.data as thd
import albumentations as albu
import random


# ============================================================================
# DEGRADACJE - symulują zniszczenia obrazu
# ============================================================================

class ImageDegradation:
    """
    Klasa do aplikowania realistycznych degradacji na obrazy.
    
    Biologiczne uzasadnienie degradacji:
    - Blur: symuluje ruch kamery, niewłaściwe ogniskowanie
    - Noise: szum sensora przy wysokim ISO, słabe oświetlenie
    - Low-light: niedoświetlenie, utrata kontrastu i szczegółów
    - Compression: artefakty JPEG (częste w rzeczywistych zdjęciach)
    - Color cast: przebarwienia od sztucznego oświetlenia
    """
    
    @staticmethod
    def add_gaussian_blur(img: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Dodaje rozmycie Gaussowskie (symulacja defocus)."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7, 9, 11, 13])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def add_motion_blur(img: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Dodaje motion blur (ruch kamery/obiektu)."""
        if kernel_size is None:
            kernel_size = random.choice([5, 9, 15, 21, 25])
        
        # Losowy kierunek ruchu
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / (kernel.sum() + 1e-8)
        return cv2.filter2D(img, -1, kernel)
    
    @staticmethod
    def add_defocus_blur(img: np.ndarray, radius: int = None) -> np.ndarray:
        """Rozmycie typu defocus (okrągłe bokeh)."""
        if radius is None:
            radius = random.choice([3, 5, 7, 9])
        
        # Okrągły kernel (disk blur)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        kernel = kernel.astype(np.float32)
        kernel = kernel / kernel.sum()
        return cv2.filter2D(img, -1, kernel)
    
    @staticmethod
    def add_gaussian_noise(img: np.ndarray, sigma: float = None) -> np.ndarray:
        """
        Dodaje szum Gaussowski (szum sensora).
        Sigma w zakresie [5, 50] - typowe dla wysokiego ISO.
        """
        if sigma is None:
            sigma = random.uniform(10, 50)
        
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_poisson_noise(img: np.ndarray, scale: float = None) -> np.ndarray:
        """
        Szum Poissona - bardziej realistyczny model szumu sensora.
        Szum zależny od intensywności sygnału (shot noise).
        """
        if scale is None:
            scale = random.uniform(0.5, 2.0)
        
        img_float = img.astype(np.float32) / 255.0
        vals = img_float * 255 * scale
        vals = np.clip(vals, 0.001, None)  # Unikaj log(0)
        noisy = np.random.poisson(vals) / (255 * scale)
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_speckle_noise(img: np.ndarray, intensity: float = None) -> np.ndarray:
        """Szum typu speckle (multiplikatywny)."""
        if intensity is None:
            intensity = random.uniform(0.05, 0.2)
        
        img_float = img.astype(np.float32) / 255.0
        noise = np.random.randn(*img.shape) * intensity
        noisy = img_float + img_float * noise
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def simulate_low_light(img: np.ndarray, factor: float = None) -> np.ndarray:
        """
        Symuluje niedoświetlenie.
        
        Biologicznie: przy słabym świetle spada stosunek sygnał/szum,
        tracimy szczegóły w cieniach i kontrast.
        """
        if factor is None:
            factor = random.uniform(0.1, 0.4)
        
        # Redukcja jasności
        darkened = (img.astype(np.float32) * factor)
        
        # Dodaj szum (naturalny przy low-light)
        sigma = (1 - factor) * 30  # Więcej szumu przy ciemniejszym obrazie
        noise = np.random.normal(0, sigma, img.shape)
        
        result = darkened + noise
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_jpeg_compression(img: np.ndarray, quality: int = None) -> np.ndarray:
        """
        Artefakty kompresji JPEG.
        Bardzo częste w rzeczywistych zdjęciach z internetu/telefonów.
        """
        if quality is None:
            quality = random.randint(10, 50)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    @staticmethod
    def add_color_cast(img: np.ndarray, intensity: float = None) -> np.ndarray:
        """
        Przebarwienia od sztucznego oświetlenia.
        Symuluje: żarówki (ciepłe), świetlówki (zielonkawe), LED (zimne).
        """
        if intensity is None:
            intensity = random.uniform(0.05, 0.2)
        
        # Losowy odcień
        cast_type = random.choice(['warm', 'cool', 'green', 'magenta'])
        
        img_float = img.astype(np.float32)
        
        if cast_type == 'warm':  # Żarówki
            img_float[:, :, 0] *= (1 - intensity * 0.3)  # Mniej blue
            img_float[:, :, 2] *= (1 + intensity)        # Więcej red
        elif cast_type == 'cool':  # LED zimne
            img_float[:, :, 0] *= (1 + intensity)        # Więcej blue
            img_float[:, :, 2] *= (1 - intensity * 0.3)  # Mniej red
        elif cast_type == 'green':  # Świetlówki
            img_float[:, :, 1] *= (1 + intensity)        # Więcej green
        else:  # Magenta
            img_float[:, :, 0] *= (1 + intensity * 0.5)  # Więcej blue
            img_float[:, :, 2] *= (1 + intensity * 0.5)  # Więcej red
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_vignette(img: np.ndarray, strength: float = None) -> np.ndarray:
        """Efekt winietowania (ciemniejsze rogi)."""
        if strength is None:
            strength = random.uniform(0.3, 0.7)
        
        h, w = img.shape[:2]
        
        # Utwórz maskę gradientową
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Odległość od środka (znormalizowana)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = dist / max_dist
        
        # Maska winiety
        vignette = 1 - (dist ** 2) * strength
        vignette = np.clip(vignette, 0, 1)
        
        # Aplikuj na każdy kanał
        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_chromatic_aberration(img: np.ndarray, shift: int = None) -> np.ndarray:
        """Aberracja chromatyczna (przesunięcie kanałów RGB)."""
        if shift is None:
            shift = random.randint(1, 3)
        
        h, w = img.shape[:2]
        result = img.copy()
        
        # Przesuń kanały R i B w przeciwnych kierunkach
        # Red channel - przesuń w prawo
        result[:, shift:, 2] = img[:, :-shift, 2]
        # Blue channel - przesuń w lewo  
        result[:, :-shift, 0] = img[:, shift:, 0]
        
        return result
    
    @staticmethod
    def apply_degradation(
        img: np.ndarray, 
        degradation_type: str = "mixed",
        severity: str = "medium"
    ) -> np.ndarray:
        """
        Aplikuje wybrany typ degradacji.
        
        Args:
            img: Obraz wejściowy (RGB, uint8)
            degradation_type: "blur", "noise", "lowlight", "compression", "mixed"
            severity: "light", "medium", "heavy" - intensywność degradacji
        """
        D = ImageDegradation  # Skrót
        
        if degradation_type == "blur":
            blur_type = random.choice(['gaussian', 'motion', 'defocus'])
            if blur_type == 'gaussian':
                return D.add_gaussian_blur(img)
            elif blur_type == 'motion':
                return D.add_motion_blur(img)
            else:
                return D.add_defocus_blur(img)
        
        elif degradation_type == "noise":
            noise_type = random.choice(['gaussian', 'poisson', 'speckle'])
            if noise_type == 'gaussian':
                return D.add_gaussian_noise(img)
            elif noise_type == 'poisson':
                return D.add_poisson_noise(img)
            else:
                return D.add_speckle_noise(img)
        
        elif degradation_type == "lowlight":
            return D.simulate_low_light(img)
        
        elif degradation_type == "compression":
            return D.add_jpeg_compression(img)
        
        elif degradation_type == "mixed":
            # Losowa kombinacja wielu degradacji
            degraded = img.copy()
            
            # Blur (50% szans)
            if random.random() > 0.5:
                blur_type = random.choice(['gaussian', 'motion', 'defocus'])
                if blur_type == 'gaussian':
                    degraded = D.add_gaussian_blur(degraded)
                elif blur_type == 'motion':
                    degraded = D.add_motion_blur(degraded)
                else:
                    degraded = D.add_defocus_blur(degraded)
            
            # Szum (60% szans)
            if random.random() > 0.4:
                noise_type = random.choice(['gaussian', 'poisson', 'speckle'])
                if noise_type == 'gaussian':
                    degraded = D.add_gaussian_noise(degraded)
                elif noise_type == 'poisson':
                    degraded = D.add_poisson_noise(degraded)
                else:
                    degraded = D.add_speckle_noise(degraded)
            
            # Low-light (30% szans)
            if random.random() > 0.7:
                degraded = D.simulate_low_light(degraded)
            
            # Kompresja JPEG (40% szans)
            if random.random() > 0.6:
                degraded = D.add_jpeg_compression(degraded)
            
            # Przebarwienia (20% szans)
            if random.random() > 0.8:
                degraded = D.add_color_cast(degraded)
            
            # Winietowanie (15% szans)
            if random.random() > 0.85:
                degraded = D.add_vignette(degraded)
            
            # Aberracja chromatyczna (10% szans)
            if random.random() > 0.9:
                degraded = D.add_chromatic_aberration(degraded)
            
            return degraded
        
        else:
            raise ValueError(f"Nieznany typ degradacji: {degradation_type}")


# ============================================================================
# AUGMENTACJE - zaawansowane dla lepszej generalizacji
# ============================================================================

def get_train_augmentation(img_size: int, aggressive: bool = True) -> albu.Compose:
    """
    Zaawansowane augmentacje dla zbioru treningowego.
    
    WAŻNE: Wszystkie transformacje są stosowane IDENTYCZNIE 
    do czystego i zdegradowanego obrazu (additional_targets).
    
    Args:
        img_size: Docelowy rozmiar obrazu
        aggressive: True = pełne augmentacje, False = lekkie
    
    Augmentacje geometryczne:
        - HorizontalFlip: Odbicie lustrzane (twarze są symetryczne)
        - ShiftScaleRotate: Przesunięcie, skalowanie, obrót
        - Perspective: Zmiana perspektywy (symulacja kąta kamery)
        - ElasticTransform: Delikatne zniekształcenia elastyczne
        - GridDistortion: Dystorsja siatki (symulacja obiektywu)
        - OpticalDistortion: Dystorsja optyczna (barrel/pincushion)
    
    Augmentacje kolorystyczne (delikatne - nie psują task restauracji):
        - RandomBrightnessContrast: Jasność/kontrast
        - HueSaturationValue: Barwa/nasycenie
        - RandomGamma: Korekta gamma
        - CLAHE: Lokalna normalizacja histogramu
    """
    
    if aggressive:
        # Pełne augmentacje dla maksymalnej generalizacji
        geometric = [
            # Podstawowe
            albu.HorizontalFlip(p=0.5),
            
            # Przesunięcie, skala, obrót (używamy Affine zamiast ShiftScaleRotate)
            albu.Affine(
                scale=(0.85, 1.15),           # ±15% skalowania
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # ±10%
                rotate=(-15, 15),             # ±15° obrotu
                p=0.7
            ),
            
            # Zmiana perspektywy (symuluje różne kąty kamery)
            albu.Perspective(
                scale=(0.02, 0.08),   # Delikatna zmiana perspektywy
                keep_size=True,
                p=0.3
            ),
            
            # Zniekształcenia elastyczne (delikatne)
            albu.ElasticTransform(
                alpha=50,             # Siła zniekształcenia
                sigma=10,             # Gładkość
                p=0.2
            ),
            
            # Dystorsja siatki (symuluje niedoskonałości obiektywu)
            albu.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),
            
            # Dystorsja optyczna (barrel/pincushion)
            albu.OpticalDistortion(
                distort_limit=0.1,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),
        ]
        
        color = [
            # Jasność i kontrast (delikatne)
            albu.RandomBrightnessContrast(
                brightness_limit=0.1,  # ±10%
                contrast_limit=0.1,
                p=0.3
            ),
            
            # Barwa, nasycenie, wartość (HSV)
            albu.HueSaturationValue(
                hue_shift_limit=10,    # ±10° barwy
                sat_shift_limit=15,    # ±15% nasycenia
                val_shift_limit=10,    # ±10% jasności
                p=0.3
            ),
            
            # Korekta gamma
            albu.RandomGamma(
                gamma_limit=(85, 115),  # 0.85-1.15
                p=0.2
            ),
            
            # CLAHE - lokalna normalizacja (pomaga z low-light)
            albu.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.2
            ),
        ]
        
        # Dodatkowe transformacje geometryczne
        extra_geometric = [
            # Random crop + resize (lepsze niż samo resize)
            albu.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),     # Crop 80-100% obrazu
                ratio=(0.9, 1.1),     # Lekka zmiana proporcji
                p=0.5
            ),
            
            # Affine dla dodatkowej różnorodności
            albu.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-5, 5),
                shear=(-3, 3),
                p=0.3
            ),
        ]
        
        transforms = geometric + color + extra_geometric
        
    else:
        # Lekkie augmentacje (dla fine-tuningu lub gdy aggressive=False)
        transforms = [
            albu.HorizontalFlip(p=0.5),
            albu.Affine(
                scale=(0.9, 1.1),              # ±10% skalowania
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # ±5%
                rotate=(-10, 10),              # ±10° obrotu
                p=0.3
            ),
        ]
    
    # Końcowy resize (zawsze)
    transforms.append(albu.Resize(height=img_size, width=img_size))
    
    return albu.Compose(
        transforms, 
        additional_targets={'degraded': 'image'}  # Identyczne dla obu obrazów!
    )


def get_eval_augmentation(img_size: int) -> albu.Compose:
    """Augmentacje dla zbioru walidacyjnego/testowego (tylko resize)."""
    return albu.Compose([
        albu.Resize(height=img_size, width=img_size),
    ], additional_targets={'degraded': 'image'})


# ============================================================================
# DATASET
# ============================================================================

class CelebADataset(thd.Dataset):
    """
    Dataset CelebA-HQ do restauracji obrazów portretowych.
    
    Struktura folderów (po uruchomieniu split_data.py):
    data/
        train/  (24,000 obrazów)
        val/    (3,000 obrazów)
        test/   (3,000 obrazów)
    
    Args:
        data_root: Ścieżka do folderu z danymi
        split: "train", "val", lub "test"
        image_size: Rozmiar obrazu wyjściowego
        degradation_type: Typ degradacji ("blur", "noise", "lowlight", "mixed")
        aggressive_augment: Tylko dla split=="train"; False = lekkie augmentacje
    
    Zwraca:
        (degraded, clean): Para tensorów (C, H, W) z wartościami [0, 1]
    """
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    
    def __init__(
        self,
        data_root: str = "data",
        split: Literal["train", "val", "test"] = "train",
        image_size: int = 256,
        degradation_type: str = "mixed",
        aggressive_augment: bool = True,
    ):
        self.split = split
        self.img_size = image_size
        self.degradation_type = degradation_type
        
        # Ścieżka do folderu z danymi
        data_path = Path(data_root) / split
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Folder {data_path} nie istnieje!\n"
                f"Uruchom najpierw: python split_data.py --source ../archive/celeba_hq_256 --dest ./data"
            )
        
        # Znajdź wszystkie obrazy
        self.images: List[Path] = []
        for ext in self.VALID_EXTENSIONS:
            self.images.extend(data_path.glob(f"*{ext}"))
            self.images.extend(data_path.glob(f"*{ext.upper()}"))
        
        self.images = sorted(self.images)
        
        if len(self.images) == 0:
            raise ValueError(f"Brak obrazów w {data_path}")
        
        # Augmentacje
        if split == "train":
            self.transform = get_train_augmentation(
                image_size, aggressive=aggressive_augment
            )
        else:
            self.transform = get_eval_augmentation(image_size)
        
        print(f"[{split.upper()}] Załadowano {len(self.images)} obrazów z {data_path}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca parę (zdegradowany_obraz, czysty_obraz).
        
        Oba tensory mają format (C, H, W) z wartościami [0, 1].
        """
        # Wczytaj obraz
        img_path = str(self.images[idx])
        img = cv2.imread(img_path)
        
        if img is None:
            # Fallback: użyj losowego innego obrazu
            print(f"Uwaga: Nie można wczytać {img_path}, używam losowego")
            idx = random.randint(0, len(self.images) - 1)
            img_path = str(self.images[idx])
            img = cv2.imread(img_path)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplikuj degradację
        degraded = ImageDegradation.apply_degradation(
            img.copy(), 
            self.degradation_type
        )
        
        # Augmentacje (takie same dla obu obrazów)
        transformed = self.transform(image=img, degraded=degraded)
        clean = transformed['image']
        degraded = transformed['degraded']
        
        # Konwersja do tensorów [0, 1]
        clean_tensor = (torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0).contiguous()
        degraded_tensor = (torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0).contiguous()
        
        return degraded_tensor, clean_tensor


# Alias dla kompatybilności wstecznej
PortraitsDataset = CelebADataset


def create_dataloaders(
    data_root: str = "data",
    batch_size: int = 8,
    image_size: int = 256,
    degradation_type: str = "mixed",
    num_workers: int = 4,
    data_limit: Optional[float] = None,
    aggressive_augment: bool = True,
) -> Tuple[thd.DataLoader, thd.DataLoader, thd.DataLoader]:
    """
    Tworzy dataloaders dla train/val/test.
    
    Args:
        data_root: Ścieżka do folderu z danymi
        batch_size: Rozmiar batcha
        image_size: Rozmiar obrazów
        degradation_type: Typ degradacji
        num_workers: Liczba workerów do ładowania danych
        data_limit: Limit danych (0.1 = 10%) - do szybkiego testowania
        aggressive_augment: Tylko train — True = pełne aug. (Perspective, Elastic, …)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = CelebADataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        degradation_type=degradation_type,
        aggressive_augment=aggressive_augment,
    )
    
    val_dataset = CelebADataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        degradation_type=degradation_type,
    )
    
    test_dataset = CelebADataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
        degradation_type=degradation_type,
    )
    
    # Opcjonalnie ogranicz dane (do szybkiego testowania)
    if data_limit is not None and 0 < data_limit < 1:
        def limit_dataset(dataset, limit):
            n = int(len(dataset) * limit)
            indices = list(range(n))
            return thd.Subset(dataset, indices)
        
        train_dataset = limit_dataset(train_dataset, data_limit)
        val_dataset = limit_dataset(val_dataset, data_limit)
        test_dataset = limit_dataset(test_dataset, data_limit)
        print(f"[DATA LIMIT] Używam {data_limit*100:.0f}% danych: "
              f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    train_loader = thd.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = thd.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    test_loader = thd.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Test CelebADataset")
    print("=" * 50)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root="data",
            batch_size=4,
            degradation_type="mixed"
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test jednego batcha
        degraded, clean = next(iter(train_loader))
        print(f"\nBatch shape: {degraded.shape}")
        print(f"Value range: [{degraded.min():.3f}, {degraded.max():.3f}]")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nAby przygotować dane, uruchom:")
        print("  python split_data.py --source ../archive/celeba_hq_256 --dest ./data")