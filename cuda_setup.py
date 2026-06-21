"""
Konfiguracja CUDA/cuDNN przed treningiem.

Niektóre kombinacje sterownik + PyTorch + cuDNN zwracają przy Conv2d:
RuntimeError: FIND/GET was unable to find an engine to execute this computation.
Wtedy jedynym niezawodnym obejściem jest torch.backends.cudnn.enabled = False
(konwolucje idą inną ścieżką na GPU, wolniej ale działa).
"""
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def configure_cuda_backends(*, no_cudnn: bool = False) -> None:
    """
    Wywołaj po stwierdzeniu torch.cuda.is_available(), zanim zbudujesz duży model.

    Args:
        no_cudnn: Wymuś wyłączenie cuDNN (np. --no_cudnn z CLI).
    """
    if not torch.cuda.is_available():
        return

    cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if no_cudnn:
        cudnn.enabled = False
        print(
            "cuDNN wyłączony (--no_cudnn). Konwolucje na GPU bez biblioteki cuDNN "
            "(zwykle wolniej, ale stabilnie)."
        )
        return

    device = torch.device("cuda:0")
    try:
        x = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float32)
        w = torch.randn(8, 3, 3, 3, device=device, dtype=torch.float32)
        F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
    except RuntimeError as e:
        err = str(e).lower()
        if "unable to find an engine" in err:
            cudnn.enabled = False
            print(
                "UWAGA: Testowa konwolucja cuDNN się nie powiodła "
                "(FIND/GET engine). Wyłączam cuDNN; trening nadal na GPU. "
                "Warto rozważyć aktualizację sterownika NVIDIA albo "
                "wersji PyTorch zgodnej z zainstalowanym CUDA.\n"
                "  → To nie jest błąd krytyczny: epoki lecą dalej, tylko konwolucje "
                "mogą być wolniejsze bez cuDNN."
            )
        else:
            raise
