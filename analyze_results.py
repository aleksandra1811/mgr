"""
Analiza wyników treningu — generowanie wykresów publikacyjnych.

Skrypt wczytuje wszystkie pliki historii (*_history.csv) oraz wyniki
końcowe (test_results.csv) z podanego katalogu eksperymentu i tworzy
zestaw wykresów odpowiednich do pracy naukowej.

Generowane wykresy:
    1. Zbieżność funkcji straty (walidacja) — wszystkie strategie
    2. Ewolucja PSNR w trakcie treningu — wszystkie strategie
    3. Ewolucja SSIM w trakcie treningu — wszystkie strategie
    4. Składowe funkcji straty per strategia
    5. Porównanie końcowe na zbiorze testowym (wykresy słupkowe)
    6. Harmonogram uczenia (learning rate schedule)

Użycie:
    python analyze_results.py output/20260206_140046
    python analyze_results.py output/20260206_140046 --output plots/ --format pdf
    python analyze_results.py output/20260206_140046 --dpi 300 --no_show
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        100,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
    "lines.markersize":  5,
})

# Paleta kolorów — spójna w całym dokumencie
STRATEGY_COLORS = {
    "combined":   "#1f77b4",   # niebieski
    "perceptual": "#d62728",   # czerwony
    "frequency":  "#2ca02c",   # zielony
    "identity":   "#9467bd",   # fioletowy
}

STRATEGY_LABELS = {
    "combined":   "Combined strategy",
    "perceptual": "Perceptual strategy",
    "frequency":  "Frequency strategy",
    "identity":   "Identity strategy",
}

LOSS_COMPONENT_LABELS = {
    "l1":       "L1 Loss (Charbonnier)",
    "ssim":     "SSIM Loss",
    "fft":      "FFT Loss",
    "gradient": "Gradient Loss",
    "total":    "Total Loss",
}

LOSS_COMPONENT_STYLES = {
    "total":    ("-",  None),
    "l1":       ("--", "o"),
    "ssim":     (":",  "s"),
    "fft":      ("-.", "^"),
    "gradient": ("--", "D"),
}


# ---------------------------------------------------------------------------
# Wczytywanie danych
# ---------------------------------------------------------------------------

def load_experiment(exp_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Wczytuje wszystkie pliki historii i wyniki testów z katalogu eksperymentu."""
    histories: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(exp_dir.glob("*_history.csv")):
        strategy = csv_path.stem.replace("_history", "")
        df = pd.read_csv(csv_path)
        # Epoch w pliku zaczyna się od 0 → dla czytelności wykresu przesuwamy na 1-based
        if "epoch" in df.columns and df["epoch"].min() == 0:
            df["epoch"] = df["epoch"] + 1
        histories[strategy] = df
        print(f"  Wczytano historię: {strategy} ({len(df)} epok)")

    test_csv = exp_dir / "test_results.csv"
    test_results: Optional[pd.DataFrame] = None
    if test_csv.exists():
        test_results = pd.read_csv(test_csv)
        print(f"  Wczytano wyniki testów: {len(test_results)} strategie")

    return histories, test_results


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name.capitalize())


def strategy_color(name: str) -> str:
    return STRATEGY_COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Wykres 1: Zbieżność strat walidacyjnych
# ---------------------------------------------------------------------------

def plot_validation_loss(histories: dict, ax: plt.Axes) -> None:
    """Wszystkie krzywe val_total na jednym wykresie."""
    for name, df in histories.items():
        if "val_total" not in df.columns:
            continue
        ax.plot(
            df["epoch"], df["val_total"],
            color=strategy_color(name),
            label=strategy_label(name),
        )

    ax.set_title("Validation Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Wykres 2: PSNR
# ---------------------------------------------------------------------------

def plot_psnr(histories: dict, ax: plt.Axes) -> None:
    for name, df in histories.items():
        if "val_psnr" not in df.columns:
            continue
        ax.plot(
            df["epoch"], df["val_psnr"],
            color=strategy_color(name),
            label=strategy_label(name),
        )

    ax.set_title("Validation PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR [dB]")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Wykres 3: SSIM
# ---------------------------------------------------------------------------

def plot_ssim(histories: Dict, ax: plt.Axes) -> None:
    """val_ssim = metryka SSIM (0–1), wyższa wartość = lepszy wynik."""
    any_ssim = any("val_ssim" in df.columns for df in histories.values())
    if not any_ssim:
        ax.text(0.5, 0.5, "No SSIM data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Validation SSIM")
        return

    for name, df in histories.items():
        if "val_ssim" not in df.columns:
            continue
        ax.plot(
            df["epoch"], df["val_ssim"],
            color=strategy_color(name),
            label=strategy_label(name),
        )

    ax.set_title("Validation SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Wykres 4: Harmonogram LR
# ---------------------------------------------------------------------------

def plot_lr(histories: dict, ax: plt.Axes) -> None:
    for name, df in histories.items():
        if "lr" not in df.columns:
            continue
        ax.plot(df["epoch"], df["lr"],
                color=strategy_color(name),
                label=strategy_label(name))

    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Wykres 5: Składowe strat per strategia (siatka 2×2)
# ---------------------------------------------------------------------------

def plot_loss_components(histories: dict, output_dir: Path, fmt: str, dpi: int, show: bool) -> None:
    """Oddzielna figura: 2×2 siatka — każda strategia ze swoimi składowymi stratami."""
    strategies = list(histories.keys())
    n = len(strategies)
    if n == 0:
        return

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Loss Components — Training Progress",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, name in enumerate(strategies):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        df = histories[name]
        color = strategy_color(name)

        train_components = [c for c in ["l1", "fft", "gradient", "total"]
                            if f"train_{c}" in df.columns]

        for comp in train_components:
            ls, marker = LOSS_COMPONENT_STYLES.get(comp, ("-", None))
            col_name = f"train_{comp}"
            label = LOSS_COMPONENT_LABELS.get(comp, comp)
            markevery = max(1, len(df) // 8)
            ax.plot(
                df["epoch"], df[col_name],
                linestyle=ls,
                marker=marker,
                markevery=markevery,
                color=color if comp == "total" else None,
                alpha=0.9 if comp == "total" else 0.65,
                linewidth=2.2 if comp == "total" else 1.5,
                label=label,
            )

        ax.set_title(strategy_label(name), color=color, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Ukryj puste subploty
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    save_figure(fig, output_dir, "skladowe_strat", fmt, dpi, show)


# ---------------------------------------------------------------------------
# Wykres 6: Porównanie końcowe — wykresy słupkowe
# ---------------------------------------------------------------------------

def plot_test_comparison(test_results: pd.DataFrame, output_dir: Path, fmt: str, dpi: int, show: bool) -> None:
    """Trzy wykresy słupkowe: test_total, test_psnr, test_ssim."""
    metrics = [
        ("test_total", "Total Loss (test)", False),
        ("test_psnr",  "PSNR [dB] (test)", True),
        ("test_ssim",  "SSIM (test)",       True),
    ]

    available = [(col, label, higher) for col, label, higher in metrics
                 if col in test_results.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5),
                             constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    fig.suptitle("Strategy Comparison — Test Set",
                 fontsize=14, fontweight="bold")

    for ax, (col, label, higher_is_better) in zip(axes, available):
        strategies = test_results["strategy"].tolist()
        values = test_results[col].tolist()
        colors = [strategy_color(s) for s in strategies]
        xlabels = [strategy_label(s) for s in strategies]

        bars = ax.bar(range(len(strategies)), values, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)

        # Wyróżnij najlepszy słupek obramowaniem
        best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

        # Adnotacje wartości
        for i, (bar, val) in enumerate(zip(bars, values)):
            fmt_val = f"{val:.3f}" if col != "test_psnr" else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                fmt_val,
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if i == best_idx else "normal",
            )

        kierunek = "↑ higher = better" if higher_is_better else "↓ lower = better"
        ax.set_title(f"{label}\n({kierunek})", fontsize=11)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(label.split("(")[0].strip())

        margin = 0.12 * max(values)
        ax.set_ylim(0, max(values) + margin)
        ax.grid(axis="y", alpha=0.4)
        ax.grid(axis="x", visible=False)

    save_figure(fig, output_dir, "porownanie_testowe", fmt, dpi, show)


# ---------------------------------------------------------------------------
# Figura przeglądowa (overview)
# ---------------------------------------------------------------------------

def plot_overview(histories: dict, output_dir: Path, fmt: str, dpi: int, show: bool) -> None:
    """Figura 2×2: straty val + PSNR + SSIM + LR — zwięzły przegląd."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    plot_validation_loss(histories, axes[0, 0])
    plot_psnr(histories, axes[0, 1])
    plot_ssim(histories, axes[1, 0])
    plot_lr(histories, axes[1, 1])

    fig.suptitle("Training Overview", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "przeglad_treningu", fmt, dpi, show)


# ---------------------------------------------------------------------------
# Narzędzia
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, output_dir: Path, name: str,
                fmt: str, dpi: int, show: bool) -> None:
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Zapisano: {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary_table(histories: Dict, test_results: Optional[pd.DataFrame]) -> None:
    """Drukuje tabelę podsumowującą wyniki końcowe."""
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    if test_results is None or test_results.empty:
        print("No test_results.csv file found.")
        return

    display_cols = {
        "strategy":   "Strategy",
        "best_epoch": "Best epoch",
        "test_total": "Total Loss",
        "test_psnr":  "PSNR [dB]",
        "test_ssim":  "SSIM",
    }

    available = {k: v for k, v in display_cols.items() if k in test_results.columns}
    df = test_results[list(available.keys())].copy()
    df.columns = list(available.values())

    if "Strategy" in df.columns:
        df["Strategy"] = df["Strategy"].map(lambda x: STRATEGY_LABELS.get(x, x))

    for col in ["Total Loss", "PSNR [dB]", "SSIM"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if col == "Total Loss" else
                                               f"{x:.2f}"  if col == "PSNR [dB]" else
                                               f"{x:.4f}")

    print(df.to_string(index=False))

    if "PSNR [dB]" in df.columns:
        best_psnr_idx = test_results["test_psnr"].idxmax()
        best = test_results.loc[best_psnr_idx, "strategy"]
        print(f"\nBest strategy by PSNR: {STRATEGY_LABELS.get(best, best)}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza wyników treningu — generowanie wykresów publikacyjnych",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Katalog eksperymentu (np. output/20260206_140046)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(Path.home() / "Desktop"),
        help="Output directory for plots (default: ~/Desktop)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Format pliku wyjściowego",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rozdzielczość wykresów (DPI); dla druku zalecane 300",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Nie wyświetlaj okien — tylko zapisz pliki",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Ogranicz analizę do wybranych strategii (np. combined perceptual)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Błąd: katalog '{exp_dir}' nie istnieje.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_show
    fmt = args.format
    dpi = args.dpi

    print(f"\nEksperyment: {exp_dir}")
    print(f"Wykresy:     {output_dir}\n")

    histories, test_results = load_experiment(exp_dir)

    if not histories:
        print("Nie znaleziono żadnych plików *_history.csv.", file=sys.stderr)
        sys.exit(1)

    if args.strategies:
        histories = {k: v for k, v in histories.items() if k in args.strategies}
        if test_results is not None:
            test_results = test_results[test_results["strategy"].isin(args.strategies)]

    print_summary_table(histories, test_results)

    print("Generowanie wykresów...")

    plot_overview(histories, output_dir, fmt, dpi, show)
    plot_loss_components(histories, output_dir, fmt, dpi, show)

    if test_results is not None and not test_results.empty:
        plot_test_comparison(test_results, output_dir, fmt, dpi, show)

    print(f"\nWszystkie wykresy zapisane w: {output_dir}")


if __name__ == "__main__":
    main()
