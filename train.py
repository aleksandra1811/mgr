import argparse
from collections import OrderedDict
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
from dataset import PortraitsDataset
import pandas as pd
import ssl
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from simple_unet import SimpleUNet

"""
to są zmienne srodowiskowe, nie wiem czy ich potrzebujesz na windowsie + nie masz chyba gpu wiec i tak nie bedzie ci sie liczylo na karcie
if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
    ssl, "_create_unverified_context", None
):
    ssl._create_default_https_context = ssl._create_unverified_context

warnings.simplefilter("ignore", UndefinedMetricWarning)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
"""


def get_args():
    # to mozesz wyjebac ale moze ci sie kiedys przydac. to sa parametry do uruchamiania programu ale nieobowiazkowe
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint if any"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint", help="Checkpoint directory"
    )
    args = parser.parse_args()
    return args


def train(train_loader, model, optimizer, epoch, device):
    model.train()
    criterion = torch.nn.MSELoss()  # błąd średniokwadratowy, mozesz zmianiac pozniej
    running_loss = 0.0
    denominator = 0

    with tqdm(train_loader) as _tqdm:
        for i, (x, y) in enumerate(_tqdm):

            x = x.to(device)
            y = y.to(device)

            output = model(x)
   
            loss = criterion(output, y)
            running_loss += loss.item()

   
            denominator += 1
   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(
                OrderedDict(
                    stage="train",
                    epoch=epoch,
                    loss=running_loss / denominator,
                ),
            )

        return (
            running_loss / denominator,
        )


def validate(valid_loader, model, device):
    with torch.no_grad():
        model.eval()
        criterion = torch.nn.MSELoss()
        running_loss = 0.0
        denominator = 0

        with tqdm(valid_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = criterion(output, y)
                running_loss += loss.item()

                denominator += 1

                _tqdm.set_postfix(
                    OrderedDict(
                        stage="valid",
                        loss=running_loss / denominator,
                    ),
                )

        return (
            running_loss / denominator,
        )


def main():
    args = get_args()
    batch_size = 1  # zacznij od jeden bo chuj wie czy ci nawet 1 pojdzie, jak nie to zmniejsz wielkosc obrazkow zeby cokolwiek poszlo
    learning_rate = 0.0001  # przeczytaj jak sie ucza sieci neuronowe. te wartosci zaleza od modelu, danych itp, wiec mozesz je rozsadnie zmieniac
    lr_decay_rate = 0.5
    lr_decay_step = 10

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if device == torch.device("cuda:0"):
        cudnn.benchmark = True

    start_epoch = 0

    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model ...")
    print(device)

    train_dataset = PortraitsDataset(data_type="train")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    valid_dataset = PortraitsDataset(data_type="valid")
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    print("Training dataset: {}".format(len(train_dataset)))
    print("Validation dataset: {}".format(len(valid_dataset)))

    model = SimpleUNet(3, 3)  #  ładujesz model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # łądujesz optymalizator, nie przejmuj sie na poczatku a moze i w ogole
    
    model = model.to(device)
    
    scheduler = StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_rate,
        last_epoch=start_epoch - 1,
    )  #  to tez bardziej zaawansowane, nie usuwaj, nie musisz zmieniac

    training_info = pd.DataFrame()  # gowno, detal, przyda ci sie do wykresow do pracy
    best_loss = 10000000   # moze byc np nieskonczonosc, zalezy od funkcji straty, minimalizujesz to 

    for epoch in range(start_epoch, 10):

        train_loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            device,
        )

        print("Loss: {}".format(train_loss))
        
        valid_loss = validate(
            valid_loader,
            model,
            device,
        )

        print("Loss: {}".format(valid_loss))

        training_epoch = pd.DataFrame(
            {
                "epoch": [str(epoch)],
                "train_loss": [str(train_loss)],
                "valid_loss": [str(valid_loss)],
            }
        )

        training_info = pd.concat([training_info, training_epoch], axis=0)

        if train_loss < best_loss:  # to w sumie powinno byc validacyjne, mozesz zmienic pozniej
            print(
                "=> [epoch {}] best training loss was improved from {} to {}".format(
                    epoch, best_loss, train_loss
                )
            )
            model_state_dict = model.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                str(checkpoint_dir.joinpath("moj_sweet_model_e{}.pth".format(epoch))),
            )
            best_loss = train_loss  # nadpisujesz jak jest najlepsza
        else:
            print(
                "=> [epoch {}] best training loss was not improved from {} ({})".format(
                    epoch, best_loss, train_loss
                )
            )

        # adjust learning rate
        scheduler.step()

    training_info.to_csv("training_results.csv", index=False)


if __name__ == "__main__":
    main()
