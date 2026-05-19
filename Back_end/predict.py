from Back_end.load_config import load_config
from Back_end.model_src.src import load_fer2013, preprocessing, predict


def main():
    cfg = load_config()

    model_cfg = cfg[cfg["model"]]

    result = predict(model_cfg["predict"])

    print(result)

main()