#!/usr/bin/env python3
"""
Trening pojedynczej strategii strat dla restauracji obrazów portretowych.

Dostępne strategie:
    combined   — balans PSNR/SSIM (L1 + SSIM + FFT + Gradient)
    perceptual — jakość percepcyjna (L1 + SSIM + Gradient)
    frequency  — deblurring (L1 + FFT + SSIM + Gradient)
    identity   — zachowanie struktury twarzy (L1 + SSIM + Gradient)

Przykłady:
    python train.py --strategy combined   --epochs 50
    python train.py --strategy perceptual --epochs 50
    python train.py --strategy frequency  --epochs 50 --degradation blur
    python train.py --strategy identity   --epochs 50

Dla treningu wszystkich strategii naraz: python pipeline.py
"""

import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from cuda_setup import configure_cuda_backends
from dataset import create_dataloaders
from simple_unet import RestorationUNet
from losses import get_loss_strategy, calculate_psnr, calculate_ssim


def parse_args():
    """Parsowanie argumentów CLI."""
    parser = argparse.ArgumentParser(
        description="Trening modelu restauracji portretów",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dane
    parser.add_argument("--data_root", type=str, default="data",
                        help="Ścieżka do folderu z danymi")
    parser.add_argument("--degradation", type=str, default="mixed",
                        choices=["blur", "noise", "lowlight", "mixed"],
                        help="Typ degradacji obrazów")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Rozmiar obrazów")
    parser.add_argument(
        "--light_augment",
        action="store_true",
        help="Lekkie augmentacje treningowe (bez Perspective, Elastic, GridDistortion, …)",
    )
    
    # Trening
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Rozmiar batcha (8 dla RTX 3070 @ 256x256)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Liczba epok")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Liczba workerów do ładowania danych")
    
    # Strategia
    parser.add_argument("--strategy", type=str, default="perceptual",
                        choices=["combined", "perceptual", "frequency", "identity"],
                        help="Strategia funkcji straty")
    
    # Checkpointy
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Folder na wyniki")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ścieżka do checkpointu do wznowienia")
    
    # Inne
    parser.add_argument("--no_amp", action="store_true",
                        help="Wyłącz mixed precision")
    parser.add_argument("--no_cudnn", action="store_true",
                        help="Wyłącz cuDNN")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    
    return parser.parse_args()


class SingleStrategyTrainer:
    """
    Trainer dla pojedynczej strategii strat.
    
    Features:
    - Mixed precision (AMP) dla oszczędności VRAM
    - Gradient clipping dla stabilności
    - tqdm progress bars
    - CSV logging
    - Early stopping
    - Checkpointing
    """
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        self.device = self._setup_device()
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"{args.strategy}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        print(f"\nŁadowanie danych z {args.data_root}...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            degradation_type=args.degradation,
            num_workers=args.num_workers,
            aggressive_augment=not args.light_augment,
        )
        
        # Model
        self.model = RestorationUNet(
            in_channels=3,
            out_channels=3,
            base_features=64
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {num_params:,} parametrów")
        
        # Loss
        self.criterion = get_loss_strategy(args.strategy).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * args.epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        
        # AMP
        self.use_amp = not args.no_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []
        
        # Resume
        if args.resume:
            self._load_checkpoint(args.resume)
        
        # Save config
        self._save_config()
    
    def _setup_device(self) -> torch.device:
        """Setup GPU/CPU."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            configure_cuda_backends(no_cudnn=self.args.no_cudnn)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("UWAGA: Trening na CPU będzie bardzo wolny!")
        return device
    
    def _save_config(self):
        """Save configuration."""
        config = vars(self.args)
        config['num_params'] = sum(p.numel() for p in self.model.parameters())
        config['train_samples'] = len(self.train_loader.dataset)
        config['val_samples'] = len(self.val_loader.dataset)
        config['test_samples'] = len(self.test_loader.dataset)
        
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"KONFIGURACJA TRENINGU")
        print(f"{'='*60}")
        print(f"Strategia: {self.args.strategy}")
        print(f"Degradacja: {self.args.degradation}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Epoki: {self.args.epochs}")
        print(f"Mixed precision: {self.use_amp}")
        print(
            "Augmentacje (train): "
            + ("lekkie (--light_augment)" if self.args.light_augment else "pełne")
        )
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"Wczytuję checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Wznawiam od epoki {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        running_losses = {}
        running_psnr = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]",
            ncols=120,
            leave=True
        )
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device, non_blocking=True).contiguous()
            clean = clean.to(self.device, non_blocking=True).contiguous()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            if self.use_amp:
                with autocast():
                    output = self.model(degraded)
                    losses = self.criterion(output, clean)
                    loss = losses['total']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(degraded)
                losses = self.criterion(output, clean)
                loss = losses['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # OneCycleLR: jeden step na batch (po udanym kroku optimizera)
            self.scheduler.step()
            
            # Accumulate
            num_batches += 1
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + value.item()
            
            with torch.no_grad():
                running_psnr += calculate_psnr(output, clean)
            
            # Update progress
            pbar.set_postfix(OrderedDict(
                loss=f"{running_losses['total']/num_batches:.4f}",
                psnr=f"{running_psnr/num_batches:.2f}dB",
                lr=f"{self.optimizer.param_groups[0]['lr']:.6f}"
            ))
        
        metrics = {k: v / num_batches for k, v in running_losses.items()}
        metrics['psnr'] = running_psnr / num_batches
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int, loader=None, desc="Val") -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        loader = loader or self.val_loader
        
        running_losses = {}
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0
        
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{self.args.epochs} [{desc}]",
            ncols=120,
            leave=True
        )
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device, non_blocking=True).contiguous()
            clean = clean.to(self.device, non_blocking=True).contiguous()
            
            if self.use_amp:
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
                loss=f"{running_losses['total']/num_batches:.4f}",
                psnr=f"{running_psnr/num_batches:.2f}dB",
                ssim=f"{running_ssim/num_batches:.4f}"
            ))
        
        metrics = {k: v / num_batches for k, v in running_losses.items()}
        metrics['psnr'] = running_psnr / num_batches
        metrics['ssim'] = running_ssim / num_batches
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'strategy': self.args.strategy,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pth")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pth")
    
    def train(self):
        """Main training loop."""
        print(f"\nRozpoczynam trening strategii: {self.args.strategy}\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log
            lr = self.optimizer.param_groups[0]['lr']
            record = {
                'epoch': epoch + 1,
                'lr': lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(record)
            
            # Save CSV
            pd.DataFrame(self.history).to_csv(
                self.output_dir / "training_history.csv", index=False
            )
            
            # Check best
            val_loss = val_metrics['total']
            is_best = val_loss < self.best_loss
            
            if is_best:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.epochs} | "
                  f"Train Loss: {train_metrics['total']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"PSNR: {val_metrics['psnr']:.2f}dB | "
                  f"SSIM: {val_metrics['ssim']:.4f}"
                  + (" | *BEST*" if is_best else ""))
            print(f"{'='*80}\n")
            
            # Early stopping
            if self.epochs_without_improvement >= self.args.patience:
                print(f"Early stopping: {self.args.patience} epok bez poprawy")
                break
        
        print(f"\nTrening zakończony!")
        print(f"Najlepsza epoka: {self.best_epoch+1}, Loss: {self.best_loss:.4f}")
        
        # Final test
        self._run_final_test()
    
    def _run_final_test(self):
        """Run final test with best model."""
        print(f"\n{'='*60}")
        print("TESTY KOŃCOWE (najlepszy model)")
        print(f"{'='*60}")
        
        # Load best
        best_path = self.output_dir / "best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test
        test_metrics = self.validate(self.best_epoch, self.test_loader, desc="Test")
        
        # Save results
        results = {
            'strategy': self.args.strategy,
            'best_epoch': self.best_epoch + 1,
            **{f"test_{k}": v for k, v in test_metrics.items()}
        }
        
        pd.DataFrame([results]).to_csv(
            self.output_dir / "test_results.csv", index=False
        )
        
        print(f"\nWyniki testowe:")
        print(f"  Loss: {test_metrics['total']:.4f}")
        print(f"  PSNR: {test_metrics['psnr']:.2f}dB")
        print(f"  SSIM: {test_metrics['ssim']:.4f}")
        print(f"\nWyniki zapisane w: {self.output_dir}")


def main():
    """Main function."""
    args = parse_args()
    trainer = SingleStrategyTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
