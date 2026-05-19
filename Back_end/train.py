import torch


from Back_end.load_config import load_config
from Back_end.model_src.src import *


def main():
    cfg = load_config()

    model_cfg = cfg[cfg["model"]]

    x_train, x_val, y_train, y_val  = load_fer2013      (model_cfg["load_data"])
    train_loader, val_loader        = preprocessing     (model_cfg["preprocessing"], x_train, x_val, y_train, y_val)
    history                         = train             (model_cfg["train"], train_loader, val_loader)
    print(history)


main()