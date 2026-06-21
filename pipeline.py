#!/usr/bin/env python3
"""
Pipeline sekwencyjnego treningu — wszystkie 4 strategie strat naraz.

Każda strategia trenuje od tych samych losowych wag (rzetelne porównanie).
Użyj --curriculum, aby każda strategia fine-tunowała z wag poprzedniej.

Przykłady:
    python pipeline.py --epochs 50 --batch_size 8
    python pipeline.py --strategies perceptual frequency --epochs 50
    python pipeline.py --epochs 30 --slow --batch_size 4
"""

import argparse
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from cuda_setup import configure_cuda_backends
from dataset import create_dataloaders
from simple_unet import RestorationUNet
from losses import get_loss_strategy, calculate_psnr, calculate_ssim


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class PipelineConfig:
    """Parametry całego pipeline'u treningowego."""

    data_root: str = "data"
    degradation_type: str = "mixed"
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4

    epochs_per_strategy: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    strategies: List[str] = field(
        default_factory=lambda: ["combined", "perceptual", "frequency", "identity"]
    )

    use_amp: bool = True
    fast_mode: bool = True

    output_dir: str = "output"
    save_every: int = 5
    patience: int = 10
    data_limit: Optional[float] = None

    use_cudnn: bool = True

    # False (domyślnie) = każda strategia startuje od tych samych wag → rzetelne porównanie
    # True = każda strategia fine-tunuje z wag poprzedniej (curriculum learning)
    curriculum: bool = False


def parse_args() -> PipelineConfig:
    """Parsowanie argumentów CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline treningowy dla restauracji portretów",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--degradation", type=str, default="mixed",
                        choices=["blur", "noise", "lowlight", "mixed"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epoki na każdą strategię")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (domyślnie 1e-4)")
    parser.add_argument("--strategies", nargs="+", 
                        default=["combined", "perceptual", "frequency", "identity"],
                        choices=["perceptual", "frequency", "identity", "combined"])
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--no_amp", action="store_true",
                        help="Wyłącz mixed precision")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--slow", action="store_true",
                        help="Użyj pełnych funkcji straty z VGG/ResNet (wolniejsze ale dokładniejsze)")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit danych (0.1 = 10%% danych) - do szybkiego testowania")
    parser.add_argument("--no_cudnn", action="store_true",
                        help="Wyłącz cuDNN")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Tryb curriculum: każda strategia fine-tunuje z wag poprzedniej. "
            "Domyślnie każda strategia startuje od tej samej losowej inicjalizacji "
            "(uczciwe porównanie strat)."
        ),
    )

    args = parser.parse_args()

    return PipelineConfig(
        data_root=args.data_root,
        degradation_type=args.degradation,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs_per_strategy=args.epochs,
        learning_rate=args.lr,
        strategies=args.strategies,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        fast_mode=not args.slow,
        data_limit=args.limit,
        use_cudnn=not args.no_cudnn,
        curriculum=args.curriculum,
    )


# ============================================================================
# METRYKI
# ============================================================================

class MetricsTracker:
    """Śledzenie i zapisywanie metryk treningowych do CSV."""

    def __init__(self, output_dir: Path, strategy_name: str):
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.history: List[Dict] = []

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        record = {
            "epoch": epoch,
            "lr": lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        self.history.append(record)
        pd.DataFrame(self.history).to_csv(
            self.output_dir / f"{self.strategy_name}_history.csv", index=False
        )

    def get_best_epoch(self, metric: str = "val_total") -> Tuple[int, float]:
        """Zwraca (numer epoki, wartość metryki) dla najlepszego checkpointu."""
        if not self.history:
            return 0, float("inf")
        
        best_idx = min(range(len(self.history)), 
                       key=lambda i: self.history[i].get(metric, float('inf')))
        return self.history[best_idx]["epoch"], self.history[best_idx][metric]


# ============================================================================
# TRAINER
# ============================================================================

class StrategyTrainer:
    """Trainer jednej strategii strat (część pipeline'u)."""

    def __init__(
        self,
        model: nn.Module,
        strategy_name: str,
        config: PipelineConfig,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.strategy_name = strategy_name
        self.config = config
        self.device = device
        self.output_dir = output_dir

        self.criterion = get_loss_strategy(strategy_name, fast=config.fast_mode).to(device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        self.scaler = GradScaler() if config.use_amp else None
        self.metrics = MetricsTracker(output_dir, strategy_name)

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(
        self, 
        train_loader, 
        epoch: int,
        scheduler=None
    ) -> Dict[str, float]:
        """Trening jednej epoki z detekcją NaN."""
        self.model.train()
        
        running_losses = {}
        running_psnr = 0.0
        num_batches = 0
        nan_count = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"[{self.strategy_name}] Epoch {epoch+1} Train",
            leave=False,
            ncols=120
        )
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device, non_blocking=True).contiguous()
            clean = clean.to(self.device, non_blocking=True).contiguous()

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                with autocast():
                    output = self.model(degraded)
                    losses = self.criterion(output, clean)
                    loss = losses["total"]

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        tqdm.write(f"UWAGA: Zbyt wiele NaN ({nan_count}), przerywam epokę")
                        break
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(degraded)
                losses = self.criterion(output, clean)
                loss = losses["total"]

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        tqdm.write(f"UWAGA: Zbyt wiele NaN ({nan_count}), przerywam epokę")
                        break
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            num_batches += 1
            for name, value in losses.items():
                v = value.item()
                if not (math.isnan(v) or math.isinf(v)):
                    running_losses[name] = running_losses.get(name, 0.0) + v

            with torch.no_grad():
                psnr_val = calculate_psnr(output, clean)
                if not math.isnan(psnr_val):
                    running_psnr += psnr_val

            if num_batches > 0:
                pbar.set_postfix(OrderedDict(
                    loss=f"{running_losses.get('total', 0) / num_batches:.4f}",
                    psnr=f"{running_psnr / num_batches:.2f}",
                ))

        if num_batches > 0:
            metrics = {k: v / num_batches for k, v in running_losses.items()}
            metrics["psnr"] = running_psnr / num_batches
        else:
            metrics = {"total": float("nan"), "psnr": float("nan")}

        if nan_count > 0:
            tqdm.write(f"  [!] Pominięto {nan_count} batchy z NaN")

        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """Walidacja."""
        self.model.eval()
        
        running_losses = {}
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0
        
        pbar = tqdm(
            val_loader,
            desc=f"[{self.strategy_name}] Epoch {epoch+1} Val",
            leave=False,
            ncols=120
        )
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device, non_blocking=True).contiguous()
            clean = clean.to(self.device, non_blocking=True).contiguous()

            if self.config.use_amp:
                with autocast():
                    output = self.model(degraded)
                    losses = self.criterion(output, clean)
            else:
                output = self.model(degraded)
                losses = self.criterion(output, clean)

            num_batches += 1
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + value.item()

            running_psnr += calculate_psnr(output, clean)
            running_ssim += calculate_ssim(output, clean)

            pbar.set_postfix(OrderedDict(
                loss=f"{running_losses['total'] / num_batches:.4f}",
                psnr=f"{running_psnr / num_batches:.2f}",
                ssim=f"{running_ssim / num_batches:.4f}",
            ))

        metrics = {k: v / num_batches for k, v in running_losses.items()}
        metrics["psnr"] = running_psnr / num_batches
        metrics["ssim"] = running_ssim / num_batches
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "strategy": self.strategy_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": vars(self.config),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, self.output_dir / f"{self.strategy_name}_latest.pth")
        if is_best:
            torch.save(checkpoint, self.output_dir / f"{self.strategy_name}_best.pth")

    def train(self, train_loader, val_loader, num_epochs: int) -> Dict[str, float]:
        """Pełny trening strategii z early stopping."""
        print(f"\n{'='*60}\nSTRATEGIA: {self.strategy_name.upper()}\n{'='*60}")

        # CosineAnnealingLR: krok per epoka (stabilniejszy niż OneCycleLR per batch)
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01,
        )

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch, scheduler=None)
            val_metrics = self.validate(val_loader, epoch)
            scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.metrics.log_epoch(epoch, train_metrics, val_metrics, lr)

            val_loss = val_metrics["total"]
            is_best = val_loss < self.best_loss

            if is_best:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            tqdm.write(
                f"[{self.strategy_name}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f}dB | "
                f"SSIM: {val_metrics['ssim']:.4f} | "
                f"LR: {lr:.6f}"
                + (" *BEST*" if is_best else "")
            )
            
            if self.epochs_without_improvement >= self.config.patience:
                tqdm.write(f"Early stopping po {self.config.patience} epokach bez poprawy.")
                break

        print(f"\n[{self.strategy_name}] Najlepsza epoka: {self.best_epoch+1}, Loss: {self.best_loss:.4f}")
        return {"best_epoch": self.best_epoch, "best_loss": self.best_loss}


# ============================================================================
# TESTY KOŃCOWE
# ============================================================================

@torch.no_grad()
def run_final_tests(
    model: nn.Module,
    test_loader,
    strategies: List[str],
    device: torch.device,
    output_dir: Path,
    use_amp: bool = True,
    fast_mode: bool = True,
) -> pd.DataFrame:
    """Wczytuje najlepszy checkpoint każdej strategii i ocenia na zbiorze testowym."""
    print(f"\n{'='*60}\nTESTY KOŃCOWE\n{'='*60}")

    results = []

    for strategy_name in strategies:
        print(f"\nTestowanie strategii: {strategy_name}")

        best_path = output_dir / f"{strategy_name}_best.pth"
        if not best_path.exists():
            print(f"  Brak checkpointu dla {strategy_name}, pomijam")
            continue

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        criterion = get_loss_strategy(strategy_name, fast=fast_mode).to(device)

        running_losses: Dict[str, float] = {}
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0

        for degraded, clean in tqdm(test_loader, desc=f"  Testing {strategy_name}", ncols=100):
            degraded = degraded.to(device, non_blocking=True).contiguous()
            clean = clean.to(device, non_blocking=True).contiguous()

            if use_amp:
                with autocast():
                    output = model(degraded)
                    losses = criterion(output, clean)
            else:
                output = model(degraded)
                losses = criterion(output, clean)

            num_batches += 1
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + value.item()

            running_psnr += calculate_psnr(output, clean)
            running_ssim += calculate_ssim(output, clean)

        result = {
            "strategy": strategy_name,
            "best_epoch": checkpoint["epoch"] + 1,
            **{f"test_{k}": v / num_batches for k, v in running_losses.items()},
            "test_psnr": running_psnr / num_batches,
            "test_ssim": running_ssim / num_batches,
        }
        results.append(result)
        print(f"  Loss: {result['test_total']:.4f} | "
              f"PSNR: {result['test_psnr']:.2f}dB | "
              f"SSIM: {result['test_ssim']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "test_results.csv", index=False)
    print(f"\nWyniki testów zapisane w {output_dir / 'test_results.csv'}")
    return df


def main():
    config = parse_args()

    print("\n" + "=" * 60)
    print("PIPELINE RESTAURACJI OBRAZÓW PORTRETOWYCH")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        configure_cuda_backends(no_cudnn=not config.use_cudnn)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("UWAGA: Trenuję na CPU — będzie bardzo wolno!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nŁadowanie danych z {config.data_root}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        degradation_type=config.degradation_type,
        num_workers=config.num_workers,
        data_limit=config.data_limit,
    )

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

    model = RestorationUNet(in_channels=3, out_channels=3, base_features=64).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parametrów")

    # Każda strategia startuje od tych samych wag → rzetelne porównanie.
    # --curriculum wyłącza reset (każda strategia fine-tunuje z poprzedniej).
    init_weights_path = output_dir / "init_weights.pth"
    torch.save(model.state_dict(), init_weights_path)

    print(f"\nKonfiguracja:")
    print(f"  Strategie: {config.strategies}")
    print(f"  Epoki/strategię: {config.epochs_per_strategy}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  Fast mode: {config.fast_mode} {'(~10x szybciej)' if config.fast_mode else '(VGG/ResNet)'}")
    print(f"  Degradacja: {config.degradation_type}")
    print(
        f"  Tryb treningu: {'curriculum (fine-tuning)' if config.curriculum else 'niezależny (uczciwe porównanie)'}"
    )

    all_results = {}
    for strategy_name in config.strategies:
        if not config.curriculum:
            model.load_state_dict(torch.load(init_weights_path, map_location=device))
            print(f"\n[{strategy_name}] Reset modelu do inicjalnych wag.")

        trainer = StrategyTrainer(
            model=model,
            strategy_name=strategy_name,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        all_results[strategy_name] = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.epochs_per_strategy,
        )

    test_results = run_final_tests(
        model=model,
        test_loader=test_loader,
        strategies=config.strategies,
        device=device,
        output_dir=output_dir,
        use_amp=config.use_amp,
        fast_mode=config.fast_mode,
    )

    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(test_results.to_string(index=False))

    best_idx = test_results["test_psnr"].idxmax()
    best_strategy = test_results.loc[best_idx, "strategy"]
    best_psnr = test_results.loc[best_idx, "test_psnr"]
    print(f"\nNajlepsza strategia: {best_strategy} (PSNR: {best_psnr:.2f} dB)")
    print(f"Wyniki zapisane w: {output_dir}")

    return test_results


if __name__ == "__main__":
    main()
