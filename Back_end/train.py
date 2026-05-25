from Back_end.load_config import load_config
from Back_end.model_src.src import load_fer2013, preprocessing, train_stream_packer_console


def main():
    cfg = load_config()

    model_cfg = cfg[cfg["model"]]

    x_train, x_val, y_train, y_val, ans     = load_fer2013      (cfg=model_cfg["load_data"])
    print("load完成")
    train_loader, val_loader ,ans           = preprocessing     (model_cfg["preprocessing"], x_train, x_val, y_train, y_val)
    print("preprocessing完成")
    history                                 = train_stream_packer_console             (model_cfg["train"], train_loader, val_loader)
    print(history)


main()